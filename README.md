# customer-support-ai-agent

An AI customer-support agent for **AppleSupport**, built from the [Customer Support on
Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.
Given an incoming customer message, the agent:

1. **Classifies** it into one of a small set of intents defined from the data.
2. **Drafts a reply**, grounded in how AppleSupport has historically resolved similar issues
   (retrieved from real past resolved threads).
3. **Decides** whether the message can be auto-handled or must be escalated to a human, with a
   stated reason.

This repo also contains the evaluation harness and hand-labelled golden set used to measure how
trustworthy the agent actually is (see `eval/` and `report/` once added).

## Why AppleSupport

Chosen after inspecting brand volumes in the dataset (`scripts/inspect_brands.py`): AppleSupport
has high tweet volume (~107k replies) and a strong customer-follow-up rate (~34%), meaning a large
number of threads that show a full resolution arc, not just a single unanswered reply. Its issue
surface (devices, iOS, account, App Store) is also narrower than a general e-commerce brand like
AmazonHelp, which makes a small, well-defined intent taxonomy easier to build and defend.

## Tech stack and why

| Choice | Reason |
|---|---|
| **Python** | Standard for data/ML pipelines; pandas + scikit-learn ecosystem fits the tabular/text nature of the dataset and baselines. |
| **kagglehub** | Downloads the Kaggle dataset programmatically using the user's existing Kaggle credentials, so the pipeline is reproducible from a clean clone without manually placing files. |
| **pandas** | Dataset is a single large CSV (~2.8M rows); pandas is sufficient without introducing a database. |
| **Gemini API** (`gemini-3.6-flash`) | Used for intent classification, grounded reply drafting, and the LLM-as-judge eval. Originally built against OpenAI, switched to Gemini per user preference; isolated behind a thin wrapper (`src/llm.py`) so the provider can be swapped again later by changing that one file. |
| **scikit-learn** | Provides the simple/trivial baselines (e.g. TF-IDF classifier) that the LLM agent is measured against. |

No database or web framework is used — this is a pipeline + evaluation harness, not a served
application, so a request/response server is out of scope unless a later step calls for a demo UI.

## Repository layout

- `data/` — raw and processed dataset files. `data/raw/` is gitignored (regenerate via
  `scripts/download_data.py`); `data/processed/` holds the reconstructed AppleSupport threads
  (`apple_triples.csv`, `apple_no_followup.csv`) produced by `scripts/build_threads.py`.
- `scripts/` — one-off/reproducible pipeline scripts (download, inspection, data prep).
- `src/` — agent source code. `src/intents.py` defines the 7-intent taxonomy derived from the
  data; `src/llm.py` wraps the Gemini API (with real rate-limiting, not just retry/backoff);
  `src/keyword_classifier.py` is a free heuristic classifier used to build the golden-set
  stratification pool; `src/classifier.py` is the agent's real LLM-based intent classifier;
  `src/retrieval.py` is a TF-IDF index over historical resolved threads for grounding drafted
  replies; `src/drafter.py` generates grounded replies from a classified intent + retrieved
  examples (escalation policy module not yet added).
- `eval/` — `eval/golden_set.csv` is the 175-example stratified golden evaluation set (34 rows
  hand-labeled, 141 drafted by the assistant and pending human review — see `eval/README.md`).
  Evaluation harness not yet added.
- `report/` — the written report (problem framing, baselines, failure analysis, decision log) —
  not yet added.
- `notebooks/` — exploratory notebooks — not yet added.

## Reproducing

1. `pip install -r requirements.txt`
2. Ensure Kaggle API credentials are configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`
   / `KAGGLE_KEY` env vars).
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (get one at
   [Google AI Studio](https://aistudio.google.com/)).
4. `python scripts/download_data.py` — downloads and caches `data/raw/twcs.csv`.
5. (optional) `python scripts/inspect_brands.py` — reproduces the brand-selection analysis above.
6. `python scripts/build_threads.py` — reconstructs AppleSupport threads into
   `data/processed/apple_triples.csv` (grounding/golden-set source) and
   `data/processed/apple_no_followup.csv` (reference set for failure analysis). Takes ~5 minutes.
7. `python scripts/classify_triples.py` — classifies all rows of `apple_triples.csv` using a free
   keyword heuristic (`src/keyword_classifier.py`) into `data/processed/apple_triples_classified.csv`,
   used to stratify-sample the golden evaluation set. Takes seconds (an earlier LLM-based version
   was abandoned after a live run hit an ~85% failure rate against the Gemini free-tier's
   5 requests/minute cap — see `src/llm.py`'s rate-limit finding).
8. `python scripts/sample_golden_set.py` — stratified-samples 25 examples per intent (175 total)
   into `eval/golden_set.csv`. This file is then hand-labeled (intent confirmation, auto/escalate
   decision, escalate reason, reply quality note) — see `eval/README.md`.

Further steps (agent pipeline, eval harness) will be documented here as they are added.
