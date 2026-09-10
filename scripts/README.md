# scripts/

## What

Standalone, reproducible scripts that prepare and inspect the raw dataset before it reaches the
agent pipeline. Run directly with `python scripts/<name>.py` from the repo root.

## Why

Kept separate from `src/` because these are one-off/setup operations (download, exploration) run
manually during pipeline setup, not code imported by the agent at run time. Separating them keeps
`src/` focused purely on the agent's classify/draft/escalate logic.

## How

Scripts read/write relative to the repo root (`data/raw/...`, `data/processed/...`) and are meant
to be run in order: `download_data.py` first, then `inspect_brands.py` and/or `build_threads.py`
against the file it produces.

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

### `build_threads.py`
- **What it does**: Loads `data/raw/twcs.csv` and reconstructs AppleSupport support threads. For
  every AppleSupport reply, looks up the customer message it responded to (via an indexed
  `tweet_id` lookup) and checks whether any customer tweet responded back to that brand reply
  (via a pre-grouped `in_response_to_tweet_id` -> rows map, built once up front). Text is cleaned
  (`@handles`/URLs stripped, whitespace collapsed) via `clean_text`. Produces two outputs:
  - `data/processed/apple_triples.csv` — customer message -> AppleSupport reply -> customer
    follow-up, for records where a follow-up exists. Deduplicated, then subsampled to
    `TRIPLE_SAMPLE_SIZE` (5,000, seeded) rows.
  - `data/processed/apple_no_followup.csv` — customer message -> AppleSupport reply pairs with no
    customer follow-up. Deduplicated, kept at full count (not subsampled).
- **Purpose**: `apple_triples.csv` is the primary dataset for grounding the agent's replies and
  sampling the golden evaluation set — a full 3-turn arc is the closest proxy in this dataset for
  "the brand's historical resolution actually landed with the customer." `apple_no_followup.csv`
  is intentionally kept aside (not used for grounding/retrieval) as a reference dataset for the
  report's failure-analysis discussion of reply patterns that may correlate with customer
  abandonment — see root README.
- **Depends on**: `data/raw/twcs.csv` (produced by `download_data.py`); `pandas`.
- **Depended on by**: Not yet consumed by other code — will be the input to the retrieval index
  and golden-set sampling steps once added.
- **Performance note**: An earlier version of this script scanned the full 2.8M-row dataframe
  inside the per-brand-reply loop to find follow-ups (O(n×m), effectively unbounded on this
  machine). It was killed mid-run and rewritten to pre-group inbound tweets by
  `in_response_to_tweet_id` once, turning the per-row follow-up lookup into an O(1) dict access.
  Output is identical in content to what the naive version would have produced; only the lookup
  strategy changed. Runtime: ~5 minutes for the full brand-reply loop.
