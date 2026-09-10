# scripts/

## What

Standalone, reproducible scripts that prepare and inspect the raw dataset before it reaches the
agent pipeline. Run directly with `python scripts/<name>.py` from the repo root.

## Why

Kept separate from `src/` because these are one-off/setup operations (download, exploration) run
manually during pipeline setup, not code imported by the agent at run time. Separating them keeps
`src/` focused purely on the agent's classify/draft/escalate logic.

## How

Scripts read/write relative to the repo root (`data/raw/...`) and are meant to be run in order:
`download_data.py` first, then `inspect_brands.py` (or any later prep script) against the file it
produces.

## File responsibilities

### `download_data.py`
- **What it does**: Downloads the `thoughtvector/customer-support-on-twitter` Kaggle dataset via
  `kagglehub.dataset_download`, which uses the user's local Kaggle API credentials. Locates
  `twcs/twcs.csv` in the downloaded/cached dataset and copies it to `data/raw/twcs.csv`.
- **Purpose**: Makes the dataset acquisition step reproducible from a clean clone without
  requiring the ~516MB CSV to be committed to git.
- **Depends on**: `kagglehub` package; valid Kaggle credentials available to the environment
  (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`).
- **Depended on by**: Every downstream script/pipeline step that reads `data/raw/twcs.csv`
  (currently `inspect_brands.py`).

### `inspect_brands.py`
- **What it does**: Loads `data/raw/twcs.csv`, isolates brand (non-inbound) tweets, and for the
  top brand accounts by tweet volume computes: total reply count, count of replies that received
  a customer follow-up (a rough proxy for "resolved/engaged" threads), and average reply length.
  Prints a ranked comparison to stdout.
- **Purpose**: Provided the data used to select the target brand (AppleSupport) for this project —
  see the root README's "Why AppleSupport" section.
- **Depends on**: `data/raw/twcs.csv` (produced by `download_data.py`); `pandas`.
- **Depended on by**: Nothing programmatically — its output informed a one-time manual decision
  recorded in the root README. Safe to re-run for reference but not part of the live pipeline.
- **Known dead code**: `id_to_row` (line 26) is computed but unused. Left as-is per scope rules —
  flagged for the user to decide whether to remove.
