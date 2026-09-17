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
Not "sounds like a helpful reply." Measured on:
- **Grounded** — reflects how this brand actually handles this kind of issue, not generic
  boilerplate.
- **Correct** — doesn't invent details it wasn't given.
- **Actionable** — gives the customer a real next step.
- **Escalation match** — does the escalation decision match what a human reviewer, reading the
  thread's actual outcome, would decide? A confident, well-written reply to a message that needed
  a human is not a success — it's the failure mode this system is built to catch.

### What was deliberately not built
- **No semantic embeddings for retrieval.** TF-IDF + cosine similarity instead, to avoid a second
  rate-limited LLM dependency running at ~5,000x the call volume of the agent's own calls.
  (Decision log #5.)
- **No LLM call for the whole escalation decision.** Checks a small number of interpretable
  signals and reports which fired, rather than one opaque LLM judgment — the assignment requires
  "a stated reason." (Decision log #6.)
- **No multi-turn conversation memory.** Decides per incoming message, matching its real call site
  (before any reply or follow-up exists). (Decision log #6, "Scoped out"; see Failure 4 below.)
- **No live/served application.** A pipeline and evaluation harness, not a deployed service —
  appropriate for this assignment's scope.

## 2. Methodology

**Data & thread definition** — Kaggle "Customer Support on Twitter," filtered to AppleSupport. A
"thread" = (customer message → brand reply → customer follow-up); the follow-up is what makes it
possible to judge whether an issue actually resolved (the basis for escalation ground truth).
No-follow-up pairs kept as a secondary failure-analysis dataset, not discarded.

**Golden evaluation set** — 175 examples, stratified 25/intent, sampled from a keyword-heuristic
pool (heuristic used only to build a balanced sampling pool, never treated as ground truth). Each
row hand-labeled (141 of 175 initially assistant-drafted against an explicit standard, then
human-reviewed/corrected) for: correct intent, auto-vs-escalate, escalate reason, reply-quality
note.

**Escalation labeling standard** (converged through iterative review — see `eval/README.md`):
decided per-thread by whether the reply/follow-up shows clean resolution, not topic or tone.
- Escalate when: a suggested fix is confirmed not to have worked; account lockout/data
  loss/financial-policy decisions a bot can't grant; the customer explicitly needs human judgment;
  or multiple channels/attempts already failed.
- Do NOT escalate purely because: the topic is common, the tone is negative/profane without real
  severity, or the message is merely long.
- This standard exists because an earlier, cruder rule (escalate by topic) was tested and
  rejected — see Section 4.

**Baselines** (both required):
- **Trivial** — always the majority intent, one fixed canned reply, always auto. The floor.
- **Simple** — keyword classifier, 7 template replies, escalates by topic category (always
  `account_security`/`billing_purchase`). This rule is deliberately reused from a rejected earlier
  version of the real policy (decision log #7, #9) — a realistic "naive first attempt," not a
  strawman.

**Evaluation harness + LLM-as-judge** — `eval/run_harness.py` runs all three tiers against the
golden set, computing intent accuracy, escalation accuracy/precision/recall, and LLM-judged reply
quality (grounded/correct/actionable, 1-5, via `src/judge.py`, scored independently of the agent
that drafted the reply).

**Judge-vs-human agreement** — `eval/judge_agreement.py` drafts and judges a fresh sample (so the
judge score always matches the exact reply text being scored), writes a CSV for a human rater to
hand-score the same rubric, then computes exact-match %, within-1-point %, and linear-weighted
Cohen's kappa per dimension. *[Sampling built; hand-scoring pass in progress as of this draft —
results to follow.]*

## 3. Results

Full 175-row harness run (`eval/results_full.json`):

| Metric | Trivial | Simple | Real agent |
|---|---|---|---|
| Intent accuracy | 29.1% | 68.6% | **72.6%** |
| Escalation accuracy | 63.4% | 61.1% | **67.4%** |
| Escalation precision | 0.00 | 0.46 | **0.59** |
| Escalation recall | 0.00 | 0.36 | 0.38 |
| Grounded (1-5) | 4.00 | 3.77 | **4.67** |
| Correct (1-5) | 5.00 | 4.54 | **4.97** |
| Actionable (1-5) | 3.00 | 3.33 | **4.13** |

- The real agent wins on every metric except escalation recall, where it's close to the simple
  baseline (0.38 vs 0.36) rather than clearly ahead.
- This **reverses** the small-sample picture reported earlier (where the simple baseline looked
  better on escalation) — see Section 5 for why that comparison was misleading, and why this
  full-scale result is the one to trust.
- **Escalation recall (0.38) is the weakest number**: the agent still misses 40 of 64 true
  escalate cases. The fix in Section 4 moved the policy off a 0/0 floor but didn't make it
  comprehensive — see Section 6.

## 4. Failure analysis

### Failure 1 — Escalation policy overfit to its own tuning set
- Red-flag check was originally ~30 hardcoded phrases, tuned on 35 hand-labeled rows to 32/35
  (91%). On a disjoint 25-row sample: **0 precision, 0 recall** — missed every true escalate case.
- Diagnosis: none of the 11 true-escalate rows contained any hardcoded phrase; classifier
  confidence (also used as a signal) was always ≥85, never triggering its threshold.
- Root cause: a keyword list built from 35 rows memorizes those sentences, not the underlying
  pattern. Real escalate signals ("nope not on shuffle," "sadly that's not it") share meaning, not
  vocabulary — no finite list generalizes to that.
- **Fix**: one LLM call judging the message semantically (failed fix / lockout / explicit human
  request); confidence-threshold signal removed entirely (it measures topic certainty, not need
  for a human — different questions). Decision log #12-13, `src/escalation.py`.
- **Validated at full scale**: precision 0.00 → 0.59, recall 0.00 → 0.38. Recall still the weakest
  metric — not treated as solved (Section 6).

### Failure 2 — "Simple beats real agent" was a small-sample artifact, reversed at scale
- Early validation (n≤18, while LLM access was constrained) showed the simple baseline
  outperforming the fixed policy on escalation.
- **At full scale this reverses**: real agent leads on precision (0.59 vs 0.46) and accuracy
  (67.4% vs 61.1%), close on recall (0.38 vs 0.36).
- Why the small sample misled: the simple baseline's rule is the exact topic-based rule already
  rejected for the real policy (decision log #9 — only 69% match on tuning rows). On a small
  sample where escalate cases happened to cluster by topic, that crude rule got lucky.
- Kept as a concrete example of why small-sample validation should never substitute for a full
  run — the earlier comparison was the misleading part, not the fix (Section 5).

### Failure 3 — Retrieval eval leakage (found, fixed, but not the actual cause of the 0/0 result)
- Every golden-set message exists verbatim in its own retrieval index, so the "no good match"
  escalation signal could never fire during eval — a message always finds itself at similarity
  1.00. Fixed via `exclude_exact_match`, threaded through retrieval/drafting/escalation.
- Fixing it did **not** fix the 0/0 escalation result — same result persisted, which is what led
  to the real diagnosis in Failure 1.
- Lesson: a plausible bug fix that doesn't change the output is itself a signal the wrong cause
  was diagnosed.

### Failure 4 — Most of the remaining recall gap is a structural information mismatch, not a model/data problem
- Reading the 40 false-negative rows (true label: escalate; agent said auto) shows a consistent
  pattern:
  - *"my 6S crashes daily... when can I expect an update?"* — escalate only because the
    **follow-up** reads "getting worse with each update."
  - *"iOS 11 made my 6s useless"* — escalate only because the follow-up reads "turned the machine
    to a brick."
- In most false negatives, **the escalation signal sits in the follow-up, not the initial
  message**. The golden-set label correctly used the full resolved thread; `decide_escalation`
  only ever sees the single incoming message (matching its real call site — no follow-up exists
  yet).
- Not a golden-set quality problem (labels are correct) and not primarily a prompt-quality problem
  with the red-flag check. It's a structural mismatch: ground truth used information the model
  doesn't have at decision time, by design.
- Fix: multi-turn conversation memory (decision log #6, already scoped out this cycle) — not
  further single-message prompt tuning.

### Failure 5 — Drafter defaulted to "DM us" on nearly every reply
- Found by inspecting real output, not aggregate metrics: sampling replies for the judge-agreement
  check showed **40/40 mentioned "DM."** Grounded/correct/actionable scores looked fine — this was
  invisible in any summary number.
- Root cause: `src/drafter.py`'s prompt told the LLM to defer to DM whenever an issue "can't be
  resolved in a single public reply (needs account access, personal details, or a definitive fix
  you can't verify)" — broad enough to match almost every real message. The LLM followed a too-eager
  instruction correctly.
- **Fix, validated**: rewrote to prefer a concrete troubleshooting step/direct answer for general
  how-to/software/device questions, reserving DM for genuinely account-specific cases. Stratified
  14-row sample (2/intent): **DM rate 100% (40/40) → 50% (7/14)**. Manually read all 14 replies —
  non-DM ones gave specific real steps (exact Settings paths, a phishing-report procedure, an
  Apple Music profile-sharing walkthrough), no invented facts. Remaining DM cases were genuinely
  account-specific or offered steps first with DM as fallback.
- Not yet re-validated at full 175-row scale — treat as a strong, evidenced improvement, not a
  fully proven one (Section 6).
- Included here because of **how** it was found: a system that always defers to a human on
  anything account-adjacent can still score well on the rubric while being meaningfully less
  useful. Reading real output caught what the aggregate score couldn't.

## 5. What's misleading about my headline number

- **A single escalation number hides *why* the policy used to fail.** 0/0 on the original
  out-of-sample test didn't mean the agent was bad at judging severity — a keyword signal was
  overfit to 35 rows. Reporting only the final 0.59/0.38, without that history, would make the fix
  look like routine tuning rather than catching a real overfitting bug a 91%-on-tuning-set check
  would never have revealed.
- **Small-sample validation pointed the wrong direction, and this report almost reported it that
  way.** Samples of 3/8/18 rows showed the simple baseline beating the real agent. Had the deadline
  landed before full access was restored, that would have been the headline number — and wrong:
  the full run reverses it. The clearest evidence here that a small sample isn't just noisier, it
  can point to the *opposite* conclusion. Every number in this report drawn from under the full
  golden set should be treated with that same skepticism.
- **Escalation recall (0.38) is low, and headline accuracy (67.4%) hides it.** With only 64/175
  true escalate cases, a policy leaning toward "auto" scores fine on accuracy while missing most of
  the cases that matter most (a missed escalation is worse than an unnecessary one). Reporting
  recall alone also invites the wrong fix — it looks like a model-quality problem, but Failure 4
  shows most of the gap is structural (the model never sees the follow-up the label was based on),
  not the model failing to recognize signals it could see.
- **4.67/4.97/4.13 reply-quality scores don't mean the replies were maximally useful.** Failure 5:
  100% of sampled replies deferred to DM — invisible in the aggregate score, very visible in the
  text. A system that always defers on anything account-adjacent can score well on this rubric
  while being less helpful than one that tries to answer directly. A general risk with aggregate
  LLM-judge scores: they can look healthy while masking a systematic behavioral pattern only
  visible by reading real output.
- **Retrieval-grounding results carry a smaller residual risk even after the exact-match leakage
  fix**: the 5,000-row pool may contain near-duplicate (not exact) versions of a golden-set
  message, which could make grounding-availability results somewhat optimistic vs. a truly unseen
  message. Flagged, not claimed resolved.

## 6. Current status and next week

### Status as of this draft
LLM API access was unstable across the final two days: the free-tier Gemini quota became unusable,
paid billing signup failed across multiple providers on account-verification issues, a
subsequently-obtained AWS Bedrock key (via a colleague's account) ran the full harness but then
expired mid-session unexpectedly, and access was finally stabilized on Groq (`DEVLOG.md` has the
complete four-provider timeline — every switch touched one file, `src/llm.py`, with zero changes
to any calling code). The full 175-row harness run (Section 3) completed on the Bedrock window.
`eval/judge_agreement.py` is built; the hand-scoring pass itself is still in progress. Failure 5
(the DM-default fix) was found and fixed after the harness run, validated on 14 rows but not yet
re-validated at full scale.

### Future Work
1. **Complete the judge-vs-human agreement check** — finish hand-scoring, fill in Sections 2/3
   with the real agreement numbers.
2. **Re-run `--full` with the DM-default fix applied** — confirm the 100%→50% improvement holds at
   full scale, the way the escalation fix was confirmed (Section 3) after initially looking weaker
   on small samples (Section 5).
3. **Multi-turn conversation memory for escalation — now the priority fix for recall** (0.38,
   the weakest headline escalation metric). Failure 4 gives a concrete, evidence-backed reason to
   prioritize this over further single-message prompt tuning, which would likely show diminishing
   returns.
4. Re-check any residual recall gap **after** multi-turn memory is added — that remainder (not the
   follow-up-dependent cases from Failure 4) is the right target for further prompt tuning.
5. **Address the residual retrieval near-duplicate risk** (Section 5) — likely a similarity
   ceiling, not just exact-match exclusion.
6. **Reconsider the confidence-threshold removal's downstream effects** — removed because it never
   fired correctly on the diagnosis sample, and full-run intent accuracy (72.6%) suggests this was
   right, but worth re-examining against the remaining 27.4% intent misclassifications.
7. **Re-run with a larger/differently-sourced golden set** if time allows, to stress-test whether
   the retrieval near-duplicate risk or any other dataset-era pattern is inflating these numbers.
