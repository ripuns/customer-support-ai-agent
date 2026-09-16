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
| **AWS Bedrock** (`google.gemma-3-27b-it`) | Used for intent classification, grounded reply drafting, escalation red-flag judgment, and the LLM-as-judge eval. Originally built against OpenAI, then Gemini, then switched to AWS Bedrock after the Gemini free tier became unusable and paid billing signup failed across multiple providers on deadline day — isolated behind a thin wrapper (`src/llm.py`) so the provider could be swapped without touching any calling code. See `DEVLOG.md` for the full provider-switch history. |
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
  rule). Supporting modules: `src/intents.py` (the 7-intent taxonomy), `src/llm.py` (LLM provider
  wrapper, currently AWS Bedrock), `src/keyword_classifier.py` (used by both the golden-set
  stratification pool and the simple baseline).
- `eval/` — `eval/golden_set.csv` is the 175-example stratified golden evaluation set, fully
  hand-labeled and human-reviewed (see `eval/README.md`), and committed to the repo (see
  "Reproducing" below). `eval/run_harness.py` scores the trivial baseline, simple baseline, and real
  agent against it (automated metrics + LLM-as-judge reply quality rubric), writing results as JSON.
  A 25-row preview run (`eval/results_preview.json`, predating a since-applied escalation-policy
  fix) originally surfaced an important finding: the real agent's escalation policy scored 0
  precision/0 recall on out-of-sample data despite 91% accuracy on the rows it was tuned against —
  a real overfitting result, not a bug. See `report/report.md`'s failure analysis and
  `src/README.md`'s `escalation.py` entry for the finding and the fix applied since.
- `report/` — the written report (`report.md`: problem framing, methodology, results, failure
  analysis, "what's misleading about my headline number," next-week plan) and `decision_log.md`
  (15 non-obvious decisions with reasoning). The Results section of `report.md` is marked
  `[PENDING]` a full harness run and a judge-vs-human agreement check — see `report/README.md`.
- `notebooks/` — exploratory notebooks — not yet added.

## Reproducing the headline results (under 15 minutes)

`eval/golden_set.csv` (the 175-row hand-labeled golden set — the thing every headline number in
`report/report.md` is measured against) is committed to this repo, not regenerated at reproduce
time. Hand-labeling it took multiple review passes and is not something a fresh clone can redo in
15 minutes — so reproducing the *headline results* means re-running the evaluation harness against
this already-labeled file, not rebuilding the whole data pipeline from the raw Kaggle dataset.

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `AWS_BEARER_TOKEN_BEDROCK` (an AWS Bedrock API key with
   access to Google's Gemma 3 27B model) and `AWS_REGION`. See `src/llm.py` for the current
   provider — this project originally used the Gemini API and switched to AWS Bedrock; see
   `DEVLOG.md` for why.
3. `python eval/run_harness.py --full` — runs the trivial baseline, simple baseline, and real agent
   against all 175 rows of the committed `eval/golden_set.csv`, writing metrics to
   `eval/results_full.json`. This is the step that reproduces the numbers in `report/report.md`.

That's the full path to the headline numbers: install, set one API key, run one command. No Kaggle
account, no raw dataset download, no re-labeling required for this path.

**Note on timing**: this step makes ~4 LLM calls per row (700 total) and its wall-clock time
depends entirely on the configured provider's latency/rate limits — with a throttled free-tier key
this can take well over 15 minutes (see `src/llm.py`'s rate-limit finding for the two separate
Gemini free-tier caps discovered during development). The 15-minute target assumes a working,
reasonably-provisioned API key without aggressive throttling. `eval/results_full.json`, once
generated, is committed alongside the report so the headline numbers can also be read directly
without re-running anything, if a working key isn't available.

## Reproducing the full pipeline from raw data (optional, not required for headline results)

Only needed to inspect/regenerate the data pipeline that originally produced `golden_set.csv`
(brand selection, thread reconstruction, keyword-based sampling pool) — not needed to reproduce the
report's numbers, since the labeled golden set is already committed.

1. Ensure Kaggle API credentials are configured (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`
   / `KAGGLE_KEY` env vars).
2. `python scripts/download_data.py` — downloads and caches `data/raw/twcs.csv`.
3. (optional) `python scripts/inspect_brands.py` — reproduces the brand-selection analysis above.
4. `python scripts/build_threads.py` — reconstructs AppleSupport threads into
   `data/processed/apple_triples.csv` (grounding/golden-set source) and
   `data/processed/apple_no_followup.csv` (reference set for failure analysis). Takes ~5 minutes.
5. `python scripts/classify_triples.py` — classifies all rows of `apple_triples.csv` using a free
   keyword heuristic (`src/keyword_classifier.py`) into `data/processed/apple_triples_classified.csv`,
   used to stratify-sample the golden evaluation set. Takes seconds (an earlier LLM-based version
   was abandoned after a live run hit an ~85% failure rate against the Gemini free-tier's
   5 requests/minute cap — see `src/llm.py`'s rate-limit finding).
6. `python scripts/sample_golden_set.py` — stratified-samples 25 examples per intent (175 total)
   into a fresh `eval/golden_set.csv`, **overwriting the committed, already-labeled one**. Only run
   this if you intend to redo the hand-labeling process from scratch (see `eval/README.md`) — not
   part of the 15-minute reproduction path above.
