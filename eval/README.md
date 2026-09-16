# eval/

## What

The golden evaluation set (hand-labeled examples used to measure the agent's actual
classification/reply/escalation quality) and the automated evaluation harness (`run_harness.py`)
that scores the trivial baseline, simple baseline, and real agent against it.

## Why

Kept separate from `src/` (agent code) and `data/` (raw/processed dataset) because this directory
holds the project's source of truth for "is the agent good" — it must stay independent of both the
data pipeline and the agent implementation so it can catch regressions in either.

## How

`golden_set.csv` is produced by `scripts/sample_golden_set.py` (stratified sample from
`data/processed/apple_triples_classified.csv`) and then hand-labeled by me. `run_harness.py` loads
this file, runs the trivial baseline, simple baseline, and real agent against each `customer_msg`,
and scores classification accuracy, escalation accuracy/precision/recall, and LLM-judged reply
quality against the hand-labels, writing results to a JSON file.

**Labeling status**: 34 of 175 rows were originally labeled by hand by the project author, with the
labeling standard converging through iterative review (see below). The remaining 141 rows were
drafted by the assistant applying that same standard (see `scripts/_draft_labels.py` — every
drafted row's `intent_label`, `auto_or_escalate`, and `escalate_reason` is a literal, auditable
judgment call). As of this update, the project author has reviewed and confirmed spreadsheet rows
2-100 (df indices 0-98) — the `[DRAFT]` prefix has been removed from `reply_quality_note` for those
rows, correcting any the assistant got wrong along the way (see conversation history for the
specific corrections made during review, e.g. rows initially escalated on "many people report this"
or tone alone that were fixed to `auto`). **Spreadsheet rows 101-175 (76 rows) are still marked
`[DRAFT]` and not yet human-reviewed** — do not treat those rows as final for grading/reporting
purposes until reviewed.

## File responsibilities

### `golden_set.csv`
- **What it is**: 175 examples (25 per intent, stratified across the 7 `INTENT_LABELS`), sampled
  from `data/processed/apple_triples_classified.csv` by `scripts/sample_golden_set.py` (seeded,
  reproducible).
- **How it was sampled**: The classified pool (`apple_triples_classified.csv`, itself produced by
  a free keyword heuristic — see `src/keyword_classifier.py`) was split into buckets by
  `predicted_intent`; 25 rows were drawn from each bucket (all 7 buckets had at least 94 candidates,
  well above 25), then the combined set was shuffled. See `scripts/README.md`'s
  `sample_golden_set.py` entry for the full rationale, including why 25/intent (175 total) was
  chosen over a larger or uneven split.
- **Columns**:
  - `customer_tweet_id`, `customer_msg`, `brand_tweet_id`, `brand_reply`, `customer_followup` —
    carried over from `apple_triples.csv` (the real historical thread).
  - `suggested_intent` — the keyword heuristic's guess. **Not ground truth** — a starting
    suggestion only, expected to contain real classification noise.
  - `intent_label` — **to be filled in by hand**: the correct intent, confirming or correcting
    `suggested_intent`.
  - `auto_or_escalate` — **to be filled in by hand**: whether this message, given how it was
    actually resolved in `brand_reply`/`customer_followup`, should have been auto-handled or
    escalated to a human.
  - `escalate_reason` — **to be filled in by hand**: short reason, required whenever
    `auto_or_escalate` is "escalate".
  - `reply_quality_note` — **to be filled in by hand**: free-text note on whether `brand_reply` was
    actually a good resolution (used later for the reply-quality eval baseline/reference).
- **Note on empty cells**: The four hand-fill columns currently contain empty strings, which
  pandas/Excel may display as blank or `NaN` when reading the CSV back — this is normal round-trip
  behavior, not a data problem.
- **Encoding incident**: During hand-labeling, the file was re-saved by an editor with a mixed
  encoding — most of the file stayed UTF-8, but a smart-quote character typed into
  `reply_quality_note` was written as a single cp1252 byte (`\x92` for `'`), which broke UTF-8
  parsing entirely (`pd.read_csv` raised `UnicodeDecodeError`). Fixed by a one-off byte-level pass
  that decodes valid UTF-8 sequences as-is and falls back to cp1252 only for the specific bytes
  that aren't valid UTF-8, then rewrites the whole file as clean UTF-8 (see repair performed
  2026-09-11; no content was lost — verified 0 replacement characters and all 175 rows intact
  after the fix). If this recurs, avoid typing smart quotes/curly apostrophes directly in the
  spreadsheet editor, or save explicitly as UTF-8 CSV rather than the editor's default.
- **Depends on**: `data/processed/apple_triples_classified.csv` (produced by
  `scripts/classify_triples.py`).
- **Depended on by**: `run_harness.py`.
- **Labeling standard (converged through review of the first 34 rows)**: `auto_or_escalate` is
  decided per-thread based on whether `brand_reply`/`customer_followup` show the issue actually
  resolving cleanly on a standard/templatable reply, not on the topic or tone alone. Escalate when:
  the customer confirms a suggested fix did NOT work; the issue involves account lockout, data
  loss, or financial/policy decisions a bot can't grant; the customer explicitly asks for something
  requiring human judgment (ETA commitments, exceptions, replacement decisions); or multiple
  channels/attempts have already failed. Do NOT escalate purely because: the complaint is common
  across many customers (that's a topic-frequency signal, not a per-thread signal); the customer's
  tone is negative/uses profanity without the issue itself being severe or unresolved; or the
  message is merely long/detailed. This standard was arrived at iteratively — early drafts escalated
  based on "many people report this" or tone alone and were corrected during review; see
  conversation history / decision log for the specific examples that shaped this.

### `scripts/_draft_labels.py`
- **What it is**: Not a reusable pipeline script — a one-off record of the 141 assistant-drafted
  label decisions for `golden_set.csv`, kept (not deleted after running) as an audit trail of the
  reasoning behind each drafted `intent_label`/`auto_or_escalate`/`escalate_reason`. Applies the
  labeling standard above to a hardcoded `tweet_id -> (intent, decision, reason, note)` dict, then
  writes those values into `golden_set.csv` and prefixes each drafted `reply_quality_note` with
  `[DRAFT]`.
- **Purpose**: Traceability — if a drafted label looks wrong during review, this file shows exactly
  what reasoning produced it, without needing to reconstruct it from conversation history.
- **Depends on**: `eval/golden_set.csv` (reads and overwrites it in place).
- **Depended on by**: Nothing — already run once; re-running is idempotent for the same 141 rows
  but would overwrite any human corrections made to those rows since. Do not re-run after manual
  review has started.

### `run_harness.py`
- **What it does**: Loads `golden_set.csv`, samples `--n` rows (default 25, seeded) unless `--full`
  is passed (all 175 rows), then runs three tiers against that sample: `baseline_trivial`,
  `baseline_simple`, and the real agent (`classifier.classify_intent` + `escalation.decide_escalation`
  + `drafter.draft_reply`). For each tier, computes intent accuracy, escalation
  accuracy/precision/recall, and mean LLM-judge scores (`judge.judge_reply`, 1-5 on
  grounded/correct/actionable) across the sampled replies. The trivial baseline's fixed reply is
  judged once, not once per row, since it's identical every time. Writes results as JSON to
  `eval/results_preview.json` (subsample runs) or `eval/results_full.json` (`--full` runs), or a
  path given by `--out`.
- **Purpose**: This is the required "evaluation harness — automated metrics + an LLM-as-judge
  rubric for reply quality" deliverable, producing the numbers the report's baseline comparison and
  results sections are built from.
- **Why a default subsample**: The real agent's classifier and drafter each make one Gemini call
  per row, plus one judge call. With `gemini-3.6-flash` (the original default) this was throttled to
  ~12.5s/call, making a full 175-row run take ~2.5 hours — and that model turned out to also have a
  hard 20-requests/day cap on this key (see `src/llm.py`'s rate-limit finding), making even a 25-row
  preview run unreliable. After switching `DEFAULT_MODEL` to `gemini-3.1-flash-lite` (~3-4s/call,
  no observed daily cap), a 25-row run across all three tiers completes in under 10 minutes. `--n 25`
  remains the default; the actual report numbers come from a `--full` run, executed once and
  committed as `eval/results_full.json` so reproducing the report's headline numbers doesn't require
  re-running the full pass.
- **Verified**: Smoke-tested end-to-end at `--n 3`. A first `--n 25` run using `gemini-3.6-flash`
  died partway through the simple-baseline pass after hitting the (then-undiscovered) daily quota.
  After switching to `gemini-3.1-flash-lite`, a `--n 25` run completed successfully but revealed the
  eval leakage bug (real agent's escalation precision/recall both 0.0 — see below). After adding
  `exclude_exact_match=True` throughout, re-ran `--n 25` again for real preview numbers (see
  `eval/results_preview.json`).
- **Eval leakage bug found via this harness, fixed — but was NOT the real cause of the 0/0 result**:
  The first successful `--n 25` run showed the real agent's escalation policy scoring exactly
  **0 precision / 0 recall** (TP=0) despite 13 actual escalate cases in the sample. The retrieval
  leakage (every `golden_set.csv` message exists verbatim in the retrieval index, so
  `escalation.py`'s "no good retrieval match" signal could never fire — see `src/retrieval.py`) was
  fixed via `RetrievalIndex.query`'s new `exclude_exact_match` parameter, now passed as `True`
  everywhere in this harness. Verified the fix changes real retrieval behavior (similarities dropped
  from 1.00 to 0.21-0.49 on the same rows). **However, re-running after the fix produced the exact
  same 0 precision / 0 recall result** — so retrieval leakage, while a real bug worth fixing, was not
  actually what was causing the escalation misses. Direct diagnosis on the 11 true-escalate rows in
  this sample showed classifier confidence was always >=85 (the confidence signal never fires) and
  **none of the 11 messages contained any `RED_FLAG_PHRASES` substring** — the escalation policy's
  keyword list, tuned against the 35 hand-labeled rows (91% match there), does not generalize to a
  fresh random sample. See `src/README.md`'s `escalation.py` entry, "Overfitting confirmed on a
  fresh sample," for the full finding — this is likely the single most important result for the
  report's failure analysis. The harness output's `"caveats"` field documents the residual, smaller
  risk that near-duplicate (not exact) messages elsewhere in the 5,000-row pool could still make
  results somewhat optimistic.
- **Depends on**: `eval/golden_set.csv`; `src/baseline_trivial.py`, `src/baseline_simple.py`,
  `src/classifier.py`, `src/drafter.py`, `src/escalation.py`, `src/judge.py`, `src/retrieval.py`.
- **Depended on by**: Not yet consumed by other code — its output (`results_full.json`) will be
  referenced by `report/` once added.
