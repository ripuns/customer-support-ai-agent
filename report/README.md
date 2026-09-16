# report/

## What
Deliverables required by the assignment beyond the codebase itself: the decision log, and (not yet
written) the 6-page report.

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
- The report (not yet written) will similarly draw on `DEVLOG.md` and `eval/results_full.json` once
  the full harness run is available.