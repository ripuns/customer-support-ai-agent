# report/

## What
Deliverables required by the assignment beyond the codebase itself: the decision log and the
6-page report.

## Why
The assignment requires these as separate artifacts from the code — a decision log of 10-15
non-obvious decisions with reasoning, and a report covering problem framing, results vs. baselines,
failure analysis, a mandatory "what's misleading about my headline number" section, and a next-week
plan. Kept in their own directory rather than mixed into the code READMEs so they read as standalone
submission documents.

## How
- `decision_log.md` — written directly from the project history recorded in `DEVLOG.md` (the
  session-by-session incident/decision log), condensed into the 10-15 decisions with the clearest
  non-obvious reasoning, in submission format rather than diary format.
- `report.md` — the full report.
  - Sections 1, 3, 4, 5, and 6 (problem framing, results, failure analysis, "what's misleading
    about my headline number," and next-week plan) are filled in with real numbers from
    `eval/results_full.json` (the completed `--full` 175-row harness run).
  - Section 2 (judge-vs-human agreement) is complete — `eval/judge_agreement.py` was run against a
    40-row hand-scored sample, with results committed in `eval/judge_agreement_results.json` (see
    `eval/README.md`).

## File responsibilities
- `decision_log.md` — 15 decisions, chronological, each with what was decided and why.
- `report.md` — the 6-page report. Results are real (from `eval/results_full.json`); judge-vs-human
  agreement is also real, from the completed hand-scoring pass (`eval/judge_agreement_results.json`).
  See "How" above.