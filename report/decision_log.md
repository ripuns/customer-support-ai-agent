# Decision Log

Non-obvious decisions made while building the AppleSupport customer support agent, with the
reasoning behind each. (in chronological order)

## 1. Brand: AppleSupport
Chosen after inspecting reply volume and follow-up rates across brands in the dataset
(`scripts/inspect_brands.py`): ~107k replies, ~34% customer-follow-up rate, and a narrower issue
surface than e-commerce brands like Amazon (device/software troubleshooting vs. open-ended order
issues). A narrower surface made a 7-intent taxonomy and a small hand-labeled golden set tractable
within the time budget.

## 2. LLM provider: Gemini, not OpenAI
Ran out of OpenAI credits, so I temporarily switched to Gemini using the free tier in Google AI Studio. Since all LLM requests run through a single wrapper (`call_llm` in `src/llm.py`), this was a quick swap rather than a major architectural shift. The rest of the pipeline (classifier, drafter, escalation, and judge) is provider-agnostic.

## 3. Thread definition: triples, not pairs
Defined a "thread" as (customer message → brand reply → customer follow-up) rather than just
(customer → brand). The follow-up is what makes it possible to judge whether an issue actually
resolved — which is the entire basis of the golden-set escalation labeling standard (decision 10).
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
retrieval-grounding availability) and reports which ones fired. Even after replacing the keyword-based red-flag check with an LLM call (decision 12), the policy stayed multi-signal and interpretable rather than becoming one black-box judgment.

## 7. Simple baseline's escalation rule reuses a rejected rule, on purpose
`baseline_simple.py` escalates by topic category (always escalates `account_security`/
`billing_purchase`) — this is the exact rule that was tried and rejected for the real escalation
policy (see decision 9: it only scored 69% against ground truth because most of those threads
actually resolved fine). Reusing it in the simple baseline was deliberate: it's a realistic "naive
first attempt" a less careful implementation might ship. It later became directly relevant to a subtle finding (decision 14).

## 8. Harness default is a subsample, not the full golden set
`eval/run_harness.py` defaults to `--n 25` for iteration speed; `--full` runs all 175 rows for the
numbers that actually go in the report. Iterating against the full set on every change would have
been far too slow given the LLM call volume per row.

## 9. Escalation policy: intent-based rule tried and rejected
First version escalated automatically for `account_security`/`billing_purchase` intents. Tested
against 35 hand-labeled rows: only 24/35 (69%) matched ground truth — most of those threads actually
resolved cleanly with a standard reply. This directly informed decision 10 below: escalation has to
be judged per-thread (did this one resolve?), not inferred from topic category.

## 10. Golden-set labeling standard: per-thread resolution, not topic or tone
`auto_or_escalate` is decided by whether the thread's actual `brand_reply`/`customer_followup` shows
clean resolution — not by topic, and not by tone. Escalate when: a suggested fix is confirmed not to
have worked; there's account lockout/data loss/a financial-policy decision a bot can't grant; the
customer explicitly asks for something needing human judgment; or multiple channels/attempts already
failed. Do NOT escalate purely because a topic is common across many customers, the tone is
negative/profane without real severity, or the message is merely long. This standard was converged on
through iterative review and is what all label-quality checks were measured against.

## 11. Rate limit discovery: two separate caps, not one
`gemini-3.6-flash`'s free tier turned out to have both a 5-requests/minute cap AND a *separate*
20-requests/DAY cap (different quotaId in the error response). The per-minute cap was fixed with a
pre-call throttle; the daily cap can't be fixed by throttling at all — it required switching model
(`gemini-3.1-flash-lite`, burst-tested at 30+ calls with zero errors on either axis). Worth recording
because the two caps produce visually identical 429 errors and are easy to conflate — the fix for one
does nothing for the other.

## 12. Escalation red-flag check: replaced keyword matching with an LLM judgment
The original `RED_FLAG_PHRASES` keyword list scored 32/35 (91%) on the 35 rows it was tuned against,
but 0/11 on a disjoint fresh sample — every true escalate case phrased "the fix didn't work" in words
that weren't on the list ("nope not on shuffle", "sadly that's not it"). This wasn't a coverage gap
fixable by adding more keywords: the real signal in this dataset is semantic (did the customer say a
fix failed, in any tone or wording), and no finite substring list generalizes to that. Replaced with
one `call_llm` judging the message directly. See `src/escalation.py` and the failure-analysis section
of the report for the full incident.

## 13. Escalation confidence-threshold signal: removed, not reworked
The policy also used to escalate when the classifier's self-reported confidence fell below a
threshold. Diagnosis showed this signal never once fired on a true escalate case (confidence was
always ≥85 on all 11 true-escalate rows in the sample used to find this). Root cause: that confidence
answers "how sure am I this is `battery_performance` vs. `software_bug`," not "does this need a
human" — a message can be 95% obviously about one topic while still needing escalation. These are
different questions, and conflating them made the signal structurally incapable of doing its job.
Removed entirely rather than reworked into a second self-reported number, since the LLM red-flag
check (decision 12) already answers the right question directly.

## 14. Scoped out: multi-turn conversation memory for escalation
`decide_escalation` only ever sees the single incoming customer message, matching its real call
site — the agent decides per incoming message, before any reply or follow-up exists yet. Messages that reference a failed prior attempt inline (very common in this dataset, e.g. "I already tried restarting, still broken") are covered; tracking risk signals that accumulate *across* separate messages in a longer conversation is not, and is noted as future work

## 15. "Simple baseline beats the real agent" is a labeling artifact, not a real regression
On a small validation sample, `baseline_simple.py`'s topic-based escalation rule outperformed the
fixed policy's precision/recall. This looks bad at first glance, but the topic-based rule is the same
rule already rejected in decision 9 for not generalizing — on a small sample where true escalate
cases happen to cluster in `account_security`/`billing_purchase`, a broad topic-based net gets lucky.
The real policy is intentionally stricter (escalates only on an actual signal in the message), which
costs recall on small samples but is the defensible policy at scale. Recorded here because it's easy
to mistake "beats the real agent on this metric" for "the real agent regressed" without this context — exactly the kind of misleading headline number the report is required to call out.
