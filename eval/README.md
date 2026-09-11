# eval/

## What

The golden evaluation set (hand-labeled examples used to measure the agent's actual
classification/reply/escalation quality) and, eventually, the automated evaluation harness that
scores the agent against it.

## Why

Kept separate from `src/` (agent code) and `data/` (raw/processed dataset) because this directory
holds the project's source of truth for "is the agent good" — it must stay independent of both the
data pipeline and the agent implementation so it can catch regressions in either.

## How

`golden_set.csv` is produced by `scripts/sample_golden_set.py` (stratified sample from
`data/processed/apple_triples_classified.csv`) and then hand-labeled by me. Once
labeled, an evaluation harness (not yet added) will load this file, run the agent against each
`customer_msg`, and score classification accuracy, escalation correctness, and reply quality
against the hand-labels.

**Labeling status**: 34 of 175 rows were labeled by hand by the project author, with the labeling
standard converging through iterative review (see below). The remaining 141 rows were drafted by
the assistant applying that same standard (see `scripts/_draft_labels.py` — every drafted row's
`intent_label`, `auto_or_escalate`, and `escalate_reason` is a literal, auditable judgment call,
and each drafted `reply_quality_note` is prefixed `[DRAFT]`). **These 141 rows are not yet
human-reviewed** — the project author must read through and correct any the assistant got wrong
before this file can be considered the actual golden set for grading/reporting purposes. The
`[DRAFT]` prefix should be removed from `reply_quality_note` as each row is confirmed.

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
- **Depended on by**: Not yet consumed by other code — will be the input to the evaluation harness
  once added.
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
