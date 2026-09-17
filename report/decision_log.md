# Decision Log

Non-obvious decisions made while building the AppleSupport customer support agent, with the
reasoning behind each. (in chronological order)

## 1. Brand: AppleSupport
Chosen after inspecting reply volume and follow-up rates across brands in the dataset
(`scripts/inspect_brands.py`): ~107k replies, ~34% customer-follow-up rate, and a narrower issue
surface than e-commerce brands like Amazon (device/software troubleshooting vs. open-ended order
issues). A narrower surface made a 7-intent taxonomy and a small hand-labeled golden set tractable
within the time budget.

## 2. LLM provider: switched four times, never touching calling code
Started against OpenAI, switched to Gemini after running out of OpenAI credits, then to AWS
Bedrock (Google's Gemma 3 27B) after Gemini's free tier became unusable and paid billing signup
failed across multiple providers, then to Groq after the Bedrock API key (borrowed from a
colleague's account) turned out to be short-lived and expired mid-session. Every switch touched
exactly one file (`call_llm` in `src/llm.py`) with zero changes to the classifier, drafter,
escalation, or judge modules — this is the single biggest practical payoff of isolating all LLM
calls behind one function, proven out for real across three unplanned provider changes rather than
just the one originally anticipated. See `DEVLOG.md`'s "LLM provider access" section for the full
incident-by-incident history, including two cases (Bedrock, Groq) where a commonly-referenced model
ID didn't actually exist in the account's catalog and had to be verified live before use.

## 3. Thread definition: triples, not pairs
Defined a "thread" as (customer message → brand reply → customer follow-up) rather than just
(customer → brand). The follow-up is what makes it possible to judge whether an issue actually
resolved — which is the entire basis of the golden-set escalation labeling standard (decision 9).
Pairs with no follow-up were kept as a secondary dataset for failure analysis rather than discarded.

## 4. 7-intent taxonomy, derived from data
`software_bug`, `battery_performance`, `account_security`, `billing_purchase`, `hardware_issue`,
`how_to_info`, `store_order_service` — derived by manually sampling ~120 real customer messages,
not guessed upfront. `battery_performance` was split out from `software_bug` specifically because
this is an iOS-11-era dataset dominated by update/battery complaints; keeping it merged would have
hidden the single largest issue category behind a generic label. Short conversational follow-ups
("Yes", "Thanks!") were explicitly scoped out of classification — they carry no intent signal on
their own.

## 5. Retrieval: TF-IDF + cosine similarity, not an embeddings API
Skipped using a Gemini embedding model here to avoid choking on rate limits. Retrieval scans the entire pool of triples across all data—not just per message—meaning it runs thousands of times more often than the main agent calls. Classic TF-IDF is free, runs locally in milliseconds, and gets the job done: it easily pulls matching complaints based on exact keywords, without the unnecessary overhead of semantic embeddings.

## 6. Escalation policy: rule-based + one narrow LLM signal, not a single LLM call
Needs to be interpretable and defensible — the assignment explicitly requires "a stated reason," and
a single opaque LLM call judging the whole decision would undermine that. The policy checks three
independent signals (unrecognized intent, an LLM-judged red-flag check on the message itself, and
retrieval-grounding availability) and reports which ones fired. Even after replacing the keyword-based red-flag check with an LLM call (decision 11), the policy stayed multi-signal and interpretable rather than becoming one black-box judgment.

**Scoped out: multi-turn conversation memory.** `decide_escalation` only ever sees the single
incoming customer message, matching its real call site — the agent decides per incoming message,
before any reply or follow-up exists yet. Messages that reference a failed prior attempt inline
(very common in this dataset, e.g. "I already tried restarting, still broken") are covered; tracking
risk signals that accumulate *across* separate messages in a longer conversation is not, and is
noted as future work (see the failure analysis on escalation recall in the report).

## 7. Simple baseline's escalation rule reuses a rejected rule, on purpose
`baseline_simple.py` escalates by topic category (always escalates `account_security`/
`billing_purchase`) — this is the exact rule that was tried and rejected for the real escalation
policy (see decision 8: it only scored 69% against ground truth because most of those threads
actually resolved fine). Reusing it in the simple baseline was deliberate: it's a realistic "naive
first attempt" a less careful implementation might ship. It later became directly relevant to a subtle finding (decision 13).

## 8. Escalation policy: intent-based rule tried and rejected
First version escalated automatically for `account_security`/`billing_purchase` intents. Tested
against 35 hand-labeled rows: only 24/35 (69%) matched ground truth — most of those threads actually
resolved cleanly with a standard reply. This directly informed decision 9 below: escalation has to
be judged per-thread (did this one resolve?), not inferred from topic category.

## 9. Golden-set labeling standard: per-thread resolution, not topic or tone
`auto_or_escalate` is decided by whether the thread's actual `brand_reply`/`customer_followup` shows
clean resolution — not by topic, and not by tone. Escalate when: a suggested fix is confirmed not to
have worked; there's account lockout/data loss/a financial-policy decision a bot can't grant; the
customer explicitly asks for something needing human judgment; or multiple channels/attempts already
failed. Do NOT escalate purely because a topic is common across many customers, the tone is
negative/profane without real severity, or the message is merely long. This standard was converged on
through iterative review and is what all label-quality checks were measured against.

**Also decided**: `eval/run_harness.py` defaults to `--n 25` for iteration speed rather than the
full golden set every run; `--full` runs all 175 rows for the numbers that actually go in the
report. Iterating against the full set on every change would have been far too slow given the LLM
call volume per row.

## 10. Rate limit discovery: two separate caps, not one
`gemini-3.6-flash`'s free tier turned out to have both a 5-requests/minute cap AND a *separate*
20-requests/DAY cap (different quotaId in the error response). The per-minute cap was fixed with a
pre-call throttle; the daily cap can't be fixed by throttling at all — it required switching model
(`gemini-3.1-flash-lite`, burst-tested at 30+ calls with zero errors on either axis). Worth recording
because the two caps produce visually identical 429 errors and are easy to conflate — the fix for one
does nothing for the other.

## 11. Escalation red-flag check: replaced keyword matching with an LLM judgment
The original `RED_FLAG_PHRASES` keyword list scored 32/35 (91%) on the 35 rows it was tuned against,
but 0/11 on a disjoint fresh sample — every true escalate case phrased "the fix didn't work" in words
that weren't on the list ("nope not on shuffle", "sadly that's not it"). This wasn't a coverage gap
fixable by adding more keywords: the real signal in this dataset is semantic (did the customer say a
fix failed, in any tone or wording), and no finite substring list generalizes to that. Replaced with
one `call_llm` judging the message directly. See `src/escalation.py` and the failure-analysis section
of the report for the full incident.

## 12. Escalation confidence-threshold signal: removed, not reworked
The policy also used to escalate when the classifier's self-reported confidence fell below a
threshold. Diagnosis showed this signal never once fired on a true escalate case (confidence was
always ≥85 on all 11 true-escalate rows in the sample used to find this). Root cause: that confidence
answers "how sure am I this is `battery_performance` vs. `software_bug`," not "does this need a
human" — a message can be 95% obviously about one topic while still needing escalation. These are
different questions, and conflating them made the signal structurally incapable of doing its job.
Removed entirely rather than reworked into a second self-reported number, since the LLM red-flag
check (decision 11) already answers the right question directly.

## 13. "Simple baseline beats the real agent" was a small-sample artifact — confirmed reversed at full scale
On an early small validation sample (n≤18), `baseline_simple.py`'s topic-based escalation rule
outperformed the fixed real-agent policy's precision/recall. At the time this looked concerning, but
the diagnosis was: the topic-based rule is the same rule already rejected in decision 8 for not
generalizing, and on a small sample where true escalate cases happen to cluster in
`account_security`/`billing_purchase`, a broad topic-based net gets lucky. **This was confirmed, not
just theorized**: the full 175-row `--full` harness run reverses the result outright — the real agent
beats the simple baseline on escalation precision (0.59 vs 0.46) and accuracy (67.4% vs 61.1%), with
recall close between the two (0.38 vs 0.36). Recorded here as concrete proof that small-sample
validation numbers during this project didn't just carry more noise than full-scale numbers — one
specific small-sample result pointed to the *opposite* conclusion from the full-scale truth. Exactly
the kind of misleading headline number the report is required to call out, and now backed by both the
small-sample number and the full-scale number that reverses it, not just the theory of why it might.

## 14. Reply drafting: rewrote the "default to DM" instruction after finding 100% of sampled replies mentioned DM
While sampling replies for the judge-vs-human agreement check, noticed all 40/40 sampled replies
mentioned "DM" — not visible in any aggregate metric, only by reading real output. Root cause was in
`src/drafter.py`'s own prompt: it instructed the LLM to direct customers to DM whenever an issue
"can't be resolved in a single public reply (needs account access, personal details, or a definitive
fix you can't verify)" — a condition broad enough to match nearly every real support message. Rewrote
it to prefer a concrete troubleshooting step or direct answer for general how-to/software/device
questions, reserving DM for genuinely account-specific cases. Validated on a stratified 14-row sample
(2 per intent, to avoid any one intent dominating): DM rate dropped from 100% (40/40) to 50% (7/14),
with the new non-DM replies giving real, specific troubleshooting steps rather than deflecting, and
no invented facts observed. Not yet re-validated at full 175-row scale — noted as next-week work
rather than assumed to hold at scale without checking, given decision 13's lesson about small-sample
numbers above.

## 15. Judge-vs-human check revealed the LLM judge's scores are compressed near the ceiling
Completed the required judge-vs-human agreement check: 40 rows, hand-scored against the same
grounded/correct/actionable rubric the LLM judge uses. Raw agreement looked fine (70-78% exact
match, 88-98% within one point), but linear-weighted Cohen's kappa was near zero or negative on two
of three dimensions (0.28 grounded, -0.05 correct, -0.08 actionable) — meaning real agreement,
after removing the part explainable by chance, was essentially absent. Cause: the judge gave 5/5 to
36-39 of 40 replies across all three dimensions; when one rater barely varies, a second rater
matching it happens easily by chance, which raw agreement % doesn't correct for but kappa does.
This is a positive finding for the agent and a useful one for the methodology at the same time: the
replies themselves are good (confirmed independently by the human rater's own scores and by reading
real output elsewhere in this project, see decision 14), but the judge's *absolute* scores
specifically shouldn't be trusted as a fine-grained quality signal — exactly the kind of
self-critical result the required check exists to surface. Not fixed today: recalibrating the
judge's prompt would require a fresh hand-scoring pass, since `draft_reply()` is non-deterministic
and a new `sample` run produces different reply text — scoped as next-week work given that cost.
