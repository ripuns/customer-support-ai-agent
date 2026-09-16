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
Built via `eval/judge_agreement.py`: samples rows from the real agent's already-judged replies (no
new LLM calls — reuses judge scores computed during the `--full` harness run), writes a CSV for a
human rater to hand-score the same replies on the same 1-5 rubric, then computes exact-match %,
within-1-point %, and linear-weighted Cohen's kappa between the human and judge scores per
dimension. [Sampling and scoring not yet executed as of this draft — results to be added once the
hand-scoring pass is complete.]

## 3. Results

Full 175-row harness run (`eval/results_full.json`), all three tiers:

| Metric | Trivial | Simple | Real agent |
|---|---|---|---|
| Intent accuracy | 29.1% | 68.6% | **72.6%** |
| Escalation accuracy | 63.4% | 61.1% | **67.4%** |
| Escalation precision | 0.00 | 0.46 | **0.59** |
| Escalation recall | 0.00 | 0.36 | 0.38 |
| Reply quality — grounded (1-5) | 4.00 | 3.77 | **4.67** |
| Reply quality — correct (1-5) | 5.00 | 4.54 | **4.97** |
| Reply quality — actionable (1-5) | 3.00 | 3.33 | **4.13** |

The real agent beats both baselines on every metric except escalation recall, where it is close to
the simple baseline (0.38 vs 0.36) rather than clearly ahead. This is a materially different
picture from the small-sample validation referenced in earlier drafts of this report and in
`DEVLOG.md`, where the simple baseline appeared to outperform the real agent on escalation — see
Section 5 for why that small-sample comparison was misleading, and why this full-scale result is the
one that should be trusted.

Escalation recall (0.38) remains the weakest number here: the real agent still misses more than
6 in 10 true escalate cases (40 false negatives out of 64 true escalate cases in this set). The
fix described in Section 4 moved the policy from a 0/0 floor to a real, working signal, but did not
make it comprehensive — see Section 6 for what's next on this specifically.

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
writeup. **Validated at full scale** (Section 3): escalation precision went from 0.00 to 0.59 and
recall from 0.00 to 0.38 on the full 175-row set — a real, substantial fix, not just a small-sample
artifact. Recall (0.38) is still the weakest of the three metrics and is not treated as solved; see
Section 6.

### Failure 2: "the simple baseline beats the real agent" was a small-sample artifact, not a real result
Early validation of the fix above (on samples of n≤18 rows, run while LLM API access was still
constrained) showed the simple baseline's escalation precision/recall looking better than the fixed
real-agent policy's. **At full scale (Section 3), this reverses**: the real agent leads the simple
baseline on both escalation precision (0.59 vs 0.46) and accuracy (67.4% vs 61.1%), with recall
close between the two (0.38 vs 0.36). The small-sample result was misleading for a specific,
diagnosable reason: the simple baseline's rule (escalate by topic category) is the exact rule
already tested and rejected for the real policy (Section 2, decision log #9: only 69% match on the
35 tuning rows, escalating many threads that actually resolved fine). On a small sample where true
escalate cases happened to cluster by topic, that crude broad-net rule got lucky — a coincidence
that a larger, representative sample corrects. This is kept in the report as a concrete example of
why small-sample validation numbers should never be treated as a substitute for a full run — not
because the fix failed, but because the earlier comparison itself was the misleading part (see
Section 5).

### Failure 3: retrieval eval leakage (found and fixed, but not the root cause it first looked like)
Every golden-set message exists verbatim in the retrieval index it is queried against (since the
golden set was sampled from the same pool). This meant the "no good retrieval match" escalation
signal could never fire during evaluation — a message always finds itself at similarity 1.00.
Fixed via an `exclude_exact_match` parameter threaded through retrieval, drafting, and escalation.
Worth including here specifically because fixing it did *not* fix the 0/0 escalation result above —
the same result persisted after the fix, which is what led to the deeper diagnosis in Failure 1.
Documented as a lesson: a plausible-sounding bug fix that doesn't change the output is itself a
useful signal that the wrong cause was diagnosed.

### Failure 4: most of the remaining escalation recall gap is a structural information mismatch, not a model or data quality problem
Manually reading a sample of the 40 false-negative rows from the full run (true label: escalate;
agent decision: auto) shows a consistent pattern. Example: `customer_msg` "my 6S crashes on a daily
basis since iOS 11... When can I expect a newer software update to resolve this?" is labeled
escalate because the `customer_followup` reads "The same exact thing is happening to me since 11
and its getting worse with each update" (`escalate_reason`: "customer asking for ETA a bot can't
commit to"). Another: `customer_msg` "So, #ios11 made my 6s useless. Screen freezes, apps dont
respond, everything is very slow" is labeled escalate only because its `customer_followup` reads
"same here. literally turned the machine to a brick." In both cases, and in most of the sampled
false negatives, **the actual escalation signal is sitting in the follow-up message, not the
initial one** — the golden-set label was correctly determined by reading the full resolved thread
(message, brand reply, and follow-up together), which is the right way to establish ground truth.
But `decide_escalation` only ever receives the single incoming `customer_msg` (Section 1, "What was
deliberately not built" — this matches its real call site: the agent decides before any reply or
follow-up exists). The initial message alone often reads as a routine first-time bug report with no
obvious red flag; the signal that makes it escalation-worthy (recurrence, severity building,
"literally turned the machine to a brick") only appears once the customer responds a second time.
This is not a golden-set quality problem — the labels are correct and match the stated standard —
and it is not primarily a prompt-quality problem with `_llm_red_flag_check` either. It is a
structural mismatch: the ground truth was established with information the model does not have
access to at decision time, by design. Multi-turn conversation memory (Section 1, decision log #14,
already named as scoped-out and next-week work) is the direct fix for this specific gap, not
further tuning of the current single-message red-flag prompt.

## 5. What's misleading about my headline number

If this report only reported "escalation accuracy: 67.4%" or "the simple baseline beat the real
agent on escalation at one point during development," both would be misleading without the context
below:

- **A single escalation accuracy/precision/recall number hides *why* the policy used to fail.** 0/0
  on the original out-of-sample test did not mean the agent was bad at judging severity — it meant a
  keyword-based signal was overfit to 35 tuning rows and needed to be replaced with a semantic
  judgment. Reporting only the final 0.59/0.38 precision/recall, without that history, would make
  the fix look like routine tuning rather than what it actually was: catching and correcting a real
  overfitting bug that a naive precision/recall check on the tuning set alone (91%!) would never
  have revealed.
- **Small-sample validation actively pointed the wrong direction, and this report almost reported
  it that way.** While full LLM API access was blocked, the escalation fix was validated on samples
  of 3, 8, and 18 rows — the largest of which showed the simple baseline beating the real agent on
  escalation precision and recall. Had the deadline arrived before full access was restored, that
  would have been the headline number, and it would have been wrong: the full 175-row run reverses
  it (Section 4, Failure 2). This is the single clearest evidence in this project that a small
  validation sample is not just "less precise" than a full run — it can point to the opposite
  conclusion. Any number in this report drawn from fewer than the full golden set should be treated
  with that same skepticism.
- **Escalation recall (0.38) is still low, and headline accuracy (67.4%) hides that.** Because only
  64 of 175 rows are true escalate cases, a policy that leans toward "auto" scores reasonably on
  accuracy while still missing most of the cases that matter most (a missed escalation is a worse
  failure than an unnecessary one). Precision (0.59) and recall (0.38) are the numbers that actually
  describe this tradeoff — accuracy alone would understate how much room for improvement remains.
  Critically, reporting recall alone also invites the wrong fix: it looks like a model-quality
  problem (retrain, re-prompt), but Section 4's Failure 4 shows most of the remaining gap is a
  structural information mismatch (the ground truth was labeled using follow-up messages the model
  never sees at decision time), not a case of the model failing to recognize signals it could see.
- **Retrieval-grounding results still carry a smaller residual risk even after the exact-match
  leakage fix**: the 5,000-row retrieval pool may contain near-duplicate (not exact) versions of a
  golden-set message from the same era of complaints, which could make grounding-availability
  results somewhat optimistic versus a truly unseen message. This is a smaller, harder-to-fully-fix
  risk than the exact-match leakage bug, and is flagged rather than claimed as resolved.

## 6. Current status and next week

### Status as of this draft
LLM API access was blocked for most of the final day (the free-tier Gemini quota became unusable,
and paid billing signup failed across multiple providers due to account-verification issues — see
`DEVLOG.md` for the full timeline) and was restored via AWS Bedrock (Gemma 3 27B) late in the day.
The full 175-row harness run (Section 3) completed successfully on the restored access. The
judge-vs-human agreement check (`eval/judge_agreement.py`) is built and reuses already-computed
judge scores with no new API calls needed for sampling, but the hand-scoring pass itself had not
been completed as of this draft — see Section 2.

### Next week
1. **Complete the judge-vs-human agreement check** — run the hand-scoring pass and fill in
   Section 2/3 with the actual agreement numbers (exact-match %, within-1-point %, linear-weighted
   kappa) per reply-quality dimension.
2. **Multi-turn conversation memory for escalation is now the priority fix for recall specifically**
   (currently 0.38, the weakest of the three headline escalation metrics per Section 5) — Section 4's
   Failure 4 shows this directly: reading a sample of the 40 false-negative rows found the
   escalation signal usually sits in the customer's follow-up message, not the initial one the model
   sees. This was already scoped out this cycle for time reasons (decision log #14); the false-negative
   analysis now gives a concrete, evidence-backed reason to prioritize it over further single-message
   prompt tuning, which would likely show diminishing returns on this specific gap.
3. Re-check whether any residual recall gap remains **after** multi-turn memory is added — if so,
   that remainder (not the follow-up-dependent cases from Failure 4) is the right target for further
   single-message red-flag prompt tuning.
4. **Address the residual retrieval near-duplicate risk** flagged in Section 5 — likely via a
   similarity ceiling (not just exact-match exclusion) when evaluating grounding availability.
5. **Reconsider the confidence-threshold removal's downstream effects** — it was removed because it
   never fired correctly on the diagnosis sample, and the full run's intent accuracy (72.6%) suggests
   this was the right call, but worth re-examining specifically on the real agent's remaining intent
   misclassifications (27.4% of rows) to see if any recoverable signal was lost.
6. **Re-run with a larger/differently-sourced golden set** if time allows, to further stress-test
   whether the retrieval near-duplicate risk (Section 5) or any other dataset-era-specific pattern is
   inflating any of these numbers.
