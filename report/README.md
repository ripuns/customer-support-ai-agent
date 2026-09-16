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
- `report.md` — the full report. Sections 1, 2, 4, 5, and 6 (problem framing, methodology, failure
  analysis, "what's misleading about my headline number," and next-week plan) are written from
  `DEVLOG.md` and the code READMEs. **Section 3 (Results) and part of Section 2 (judge-vs-human
  agreement) are explicitly marked `[PENDING]`** — they require a `--full` 175-row harness run and a
  hand-scored judge-agreement check, both blocked on LLM API access as of this draft (see
  `DEVLOG.md`'s "Current blocker" entry). Do not treat this report as submission-final until those
  sections are filled in with real numbers.

## File responsibilities
- `decision_log.md` — 15 decisions, chronological, each with what was decided and why.
- `report.md` — the 6-page report. See "How" above for what's pending.