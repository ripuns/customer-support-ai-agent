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
| **Gemini API** (`gemini-3.1-flash-lite`) | Used for intent classification, grounded reply drafting, and the LLM-as-judge eval. Originally built against OpenAI, switched to Gemini per user preference; isolated behind a thin wrapper (`src/llm.py`) so the provider can be swapped again later by changing that one file. Model was switched again from `gemini-3.6-flash` after discovering that model's 20-requests/day free-tier cap on this key — see `src/llm.py`'s rate-limit finding. |
| **scikit-learn** | Provides the simple/trivial baselines (e.g. TF-IDF classifier) that the LLM agent is measured against. |

No database or web framework is used — this is a pipeline + evaluation harness, not a served
application, so a request/response server is out of scope unless a later step calls for a demo UI.

## Repository layout

- `data/` — raw and processed dataset files. `data/raw/` is gitignored (regenerate via
  `scripts/download_data.py`); `data/processed/` holds the reconstructed AppleSupport threads
  (`apple_triples.csv`, `apple_no_followup.csv`) produced by `scripts/build_threads.py`.
- `scripts/` — one-off/reproducible pipeline scripts (download, inspection, data prep).
- `src/` — agent source code. All three required agent components are implemented:
  `src/classifier.py` (LLM-based intent classification), `src/drafter.py` (grounded reply drafting,
  using `src/retrieval.py`'s TF-IDF index over historical resolved threads), and
  `src/escalation.py` (rule-based auto-vs-escalate decision with a stated reason, 91% match against
  the hand-labeled golden set). Both required baselines are implemented too:
  `src/baseline_trivial.py` (fixed majority-class prediction, no learning) and
  `src/baseline_simple.py` (keyword classifier + template replies + a naive intent-risk escalation
  rule). Supporting modules: `src/intents.py` (the 7-intent taxonomy), `src/llm.py` (rate-limited
  Gemini wrapper), `src/keyword_classifier.py` (used by both the golden-set stratification pool and
  the simple baseline).
- `eval/` — `eval/golden_set.csv` is the 175-example stratified golden evaluation set (34 rows
  hand-labeled, 141 drafted by the assistant and pending human review — see `eval/README.md`).
  `eval/run_harness.py` scores the trivial baseline, simple baseline, and real agent against it
  (automated metrics + LLM-as-judge reply quality rubric), writing results as JSON. A 25-row preview
  run (`eval/results_preview.json`) surfaced an important finding: the real agent's escalation
  policy scores 0 precision/0 recall on this out-of-sample data despite 91% accuracy on the rows it
  was tuned against — a real overfitting result, not a bug, documented in `src/README.md`'s
  `escalation.py` entry.
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
9. `python eval/run_harness.py` — runs the trivial baseline, simple baseline, and real agent
   against a 25-row sample of `eval/golden_set.csv`, writing metrics to
   `eval/results_preview.json`. Takes under 10 minutes. Pass `--full` to run all 175 rows for the
   numbers reported in `report/` (longer, but no longer ~2.5 hours now that `DEFAULT_MODEL` is
   `gemini-3.1-flash-lite` — see `src/llm.py`'s rate-limit finding for the full history of both
   free-tier caps discovered on this key).

Further steps (report, decision log) will be documented here as they are added.
