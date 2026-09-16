# AppleSupport Customer Support Agent — Report

## 1. Problem framing

### What the agent does
Given a single incoming customer message directed at AppleSupport on Twitter, the agent:
1. **Classifies** it into one of 7 intents derived from the data (`software_bug`,
   `battery_performance`, `account_security`, `billing_purchase`, `hardware_issue`, `how_to_info`,
   `store_order_service`).
2. **Drafts a reply**, grounded in how AppleSupport has actually resolved similar issues in the
   past — retrieved via TF-IDF/cosine similarity over ~5,000 real historical resolved threads.
3. **Decides** whether the message can be auto-handled or must be escalated to a human, with a
   stated reason.

### What "good" means here
"Good" is not "sounds like a helpful reply." It is measured against three specific, separately
scored properties: is the reply **grounded** in how this brand actually handles this kind of issue
(not generic boilerplate), is it **correct** (doesn't invent details it wasn't given), and is it
**actionable** (gives the customer a real next step). Separately, and just as important: does the
**escalation decision** match what a human reviewer, reading the same thread's actual outcome,
would decide? A confident, well-written reply to a message that needed a human is not a success —
it's the failure mode this system is specifically built to catch.

### What was deliberately not built
- **No semantic embeddings for retrieval.** TF-IDF + cosine similarity was used instead of an
  embeddings API, to avoid a second rate-limited LLM dependency running at ~5,000x the call volume
  of the agent's own per-message calls. See decision log #5.
- **No LLM call for the whole escalation decision.** The escalation policy checks a small number of
  interpretable signals and reports which ones fired, rather than asking an LLM to make the whole
  call in one opaque step — the assignment explicitly requires "a stated reason," and a single
  black-box judgment would undermine that even if it scored well. See decision log #6.
- **No multi-turn conversation memory.** The agent decides per incoming message, matching its real
  call site (before any reply or follow-up exists). A customer's accumulated history across
  multiple separate messages is not currently used. See "Next week" and decision log #14.
- **No live/served application.** This is a pipeline and evaluation harness, not a deployed service
  — appropriate for the scope of this assignment.

## 2. Methodology

### Data and thread definition
Source: the Kaggle "Customer Support on Twitter" dataset, filtered to AppleSupport. A "thread" is
defined as a triple: (customer message → brand reply → customer follow-up). The follow-up is what
makes it possible to judge whether an issue actually resolved, which is the basis for the
escalation ground truth below. Pairs with no follow-up were kept separately as a secondary dataset
for failure analysis (what reply patterns correlate with customer abandonment), not discarded.

### Golden evaluation set
175 examples, stratified 25 per intent, sampled from a keyword-heuristic-classified pool of ~5,000
threads (the heuristic is only used to build a balanced sampling pool — it is not treated as ground
truth anywhere in scoring). Each row was hand-labeled (by the project author, with 141 of 175 rows
initially assistant-drafted against an explicit, auditable standard and then human-reviewed and
corrected) for: the correct intent, whether the thread should have been auto-handled or escalated,
the reason if escalated, and a note on the actual historical reply's quality.

**Escalation labeling standard** (converged through iterative review — see `eval/README.md`):
decided per-thread by whether the actual `brand_reply`/`customer_followup` shows the issue
resolving cleanly, not by topic or tone. Escalate when: the customer confirms a suggested fix did
NOT work; the issue involves account lockout, data loss, or a financial/policy decision a bot can't
grant; the customer explicitly asks for something requiring human judgment; or multiple
channels/attempts already failed. Do NOT escalate purely because a topic is common across many
customers, the tone is negative/profane without real severity, or the message is merely long. This
standard exists specifically because an early, cruder rule (escalate by topic category) was tested
and rejected — see Section 4.

### Baselines
Two baselines, as required:
- **Trivial**: always predicts the majority intent (`software_bug`), always returns one fixed
  canned reply, always auto-handles. The floor every other approach must clear.
- **Simple**: a free keyword-substring classifier, 7 template replies per intent, and an
  escalation rule that escalates by topic category (always escalates `account_security` and
  `billing_purchase`). This rule was deliberately reused from an earlier version of the real
  policy that was tested and rejected (see decision log #7, #9) — a realistic "naive first
  attempt" for comparison, not a strawman built to lose.

### Evaluation harness and LLM-as-judge
`eval/run_harness.py` runs all three tiers (trivial, simple, real agent) against a sample of the
golden set and computes: intent classification accuracy; escalation accuracy, precision, and
recall against the hand labels; and LLM-judged reply quality (grounded / correct / actionable, each
1-5) via a separate judge prompt (`src/judge.py`), scored independently of the agent that produced
the reply.

### Judge-vs-human agreement
[PENDING — not yet built as of this draft. Requires hand-scoring a subset of `judge_reply` outputs
against the same 1-5 rubric and computing agreement (e.g. Cohen's kappa or simple % agreement)
against the LLM judge's scores. Blocked on working LLM API access — see Section 6.]

## 3. Results

**[PENDING — this section requires a `--full` (175-row) harness run, not yet completed as of this
draft.]**

What will go here: a results table (intent accuracy / escalation accuracy-precision-recall / mean
reply-quality scores) for trivial baseline, simple baseline, and the real agent, run against all
175 golden-set rows, plus the judge-vs-human agreement figure from Section 2.

Context available now: a 25-row preview run taken *before* the escalation-policy fix described in
Section 4 showed the real agent's escalation policy scoring 0 precision / 0 recall — a result now
understood to be a real overfitting bug (see Section 4), since fixed and validated (on small
samples only, pending full validation) to no longer sit at the 0/0 floor. That preview run's numbers
are stale and are not reported here as final — see `eval/results_preview.json` for the historical
record and `DEVLOG.md` for the full incident timeline.

## 4. Failure analysis

### Failure 1: the escalation policy was overfit to its own tuning set
The escalation policy's red-flag check was originally a hardcoded list of ~30 phrases (e.g.
"didn't work," "locked out," "speak to a human"), tuned by reading 35 hand-labeled rows until it
matched 32/35 (91%) of them. Run against a *different*, disjoint 25-row sample, it scored **0
precision and 0 recall** — it missed every single true escalate case. Direct diagnosis: on the 11
true-escalate rows in that sample, none contained any of the ~30 hardcoded phrases, and the
classifier's confidence score (also used as an escalation signal) was always ≥85, never triggering
its threshold. The real cause was not a bug in the code — it was that a keyword list built by
reading a small sample memorizes the specific sentences in that sample rather than the underlying
pattern. The messages that actually needed escalation phrased "the fix didn't work" in dozens of
different ways ("nope not on shuffle," "sadly that's not it," "still just gives me a link") that
share meaning but not vocabulary — no finite keyword list generalizes to that.

**Fix applied**: replaced the keyword list with a single LLM call that judges the message
semantically (does it show a failed prior fix, lockout/data-loss, or an explicit request for a
human), and removed the confidence-threshold signal entirely after confirming it was structurally
incapable of catching these cases (it measures certainty about *topic*, not need for a human — two
different questions). See decision log #12-13 and `src/escalation.py` for the full technical
writeup. Validated so far only on small samples (n≤18) due to LLM API access being blocked on
deadline day; full validation is pending (Section 6).

### Failure 2: "the simple baseline beats the real agent" is a misleading comparison
On small-sample validation, the simple baseline's escalation precision/recall looked better than
the fixed real-agent policy's. This is not a real regression — the simple baseline's rule (escalate
by topic category) is the exact rule already tested and rejected for the real policy (Section 2,
decision log #9: only 69% match on the 35 tuning rows, escalating many threads that actually
resolved fine). On a small sample where true escalate cases happen to cluster by topic, a crude
broad-net rule gets lucky. The real policy is intentionally *stricter* — it only escalates on an
actual signal found in the message, so it correctly abstains on ambiguous cases, which costs recall
specifically on small samples. This is included here because it is exactly the kind of headline
number that looks bad out of context but is misleading without it (see Section 5).

### Failure 3: retrieval eval leakage (found and fixed, but not the root cause it first looked like)
Every golden-set message exists verbatim in the retrieval index it is queried against (since the
golden set was sampled from the same pool). This meant the "no good retrieval match" escalation
signal could never fire during evaluation — a message always finds itself at similarity 1.00.
Fixed via an `exclude_exact_match` parameter threaded through retrieval, drafting, and escalation.
Worth including here specifically because fixing it did *not* fix the 0/0 escalation result above —
the same result persisted after the fix, which is what led to the deeper diagnosis in Failure 1.
Documented as a lesson: a plausible-sounding bug fix that doesn't change the output is itself a
useful signal that the wrong cause was diagnosed.

## 5. What's misleading about my headline number

If this report only reported "escalation accuracy: X%" or "the simple baseline beats the real
agent on escalation," both would be actively misleading without the context in Section 4:

- **A single escalation accuracy/precision/recall number hides *why* the policy fails.** 0/0 (or
  any low number) on a small out-of-sample set does not mean the agent is bad at judging severity —
  it means a keyword-based signal was overfit to 35 tuning rows and needed to be replaced with a
  semantic judgment. Reporting the number alone, without the overfitting diagnosis, would make the
  finding look like a dead end rather than a specific, fixable, and (partially) fixed problem.
- **"Simple baseline beats real agent" looks like a regression, but the simple baseline's number is
  inflated by a policy already known not to generalize** (Section 4, Failure 2). A reader
  comparing only the headline metric would conclude the wrong thing about which system is actually
  more trustworthy at scale.
- **Numbers from a 25-row (or smaller) sample carry real sampling noise.** With as few as 8-25 true
  escalate cases in a sample, a single row flipping prediction can swing precision or recall by
  10+ percentage points. The `--full` 175-row run (Section 3, pending) is meaningfully more
  trustworthy than any of the small-sample numbers referenced in this draft, and none of the small
  sample numbers above should be read as final.
- **Retrieval-grounding results still carry a smaller residual risk even after the exact-match
  leakage fix**: the 5,000-row retrieval pool may contain near-duplicate (not exact) versions of a
  golden-set message from the same era of complaints, which could make grounding-availability
  results somewhat optimistic versus a truly unseen message. This is a smaller, harder-to-fully-fix
  risk than the exact-match leakage bug, and is flagged rather than claimed as resolved.

## 6. Current status and next week

### Blocked as of this draft
LLM API access needed to complete Sections 2 (judge-vs-human agreement) and 3 (full results) was
blocked for most of the final day: the free-tier Gemini quota became unusable, and switching to a
paid provider hit unrelated account-verification issues across multiple providers (see `DEVLOG.md`
for the full timeline). This report is being finished with that section explicitly marked pending
rather than filled with stale or small-sample numbers presented as final.

### Next week
1. **Run the full 175-row harness** once LLM access is unblocked, and fill in Section 3 with real
   numbers, replacing every "pending" marker in this report.
2. **Build the judge-vs-human agreement check** — hand-score a meaningful subset of judge outputs
   and compute agreement, as required by the assignment.
3. **Fully validate the escalation-policy fix** at scale (only validated on samples ≤18 rows so
   far) — confirm the LLM red-flag judgment actually improves precision/recall on the full set, not
   just directionally on a small sample.
4. **Multi-turn conversation memory for escalation** (deliberately scoped out this cycle — decision
   log #14): extend the policy to consider a customer's prior messages in the same conversation, not
   just the single latest one, since risk signals can accumulate across turns that a single-message
   view misses.
5. **Address the residual retrieval near-duplicate risk** flagged in Section 5 — likely via a
   similarity ceiling (not just exact-match exclusion) when evaluating grounding availability.
6. **Reconsider the confidence-threshold removal's downstream effects** — it was removed because it
   never fired correctly, but a full-scale run may reveal other cases (e.g. genuinely low-confidence
   misclassifications) where some signal derived from classifier uncertainty is still useful, just
   not the one that was removed.
