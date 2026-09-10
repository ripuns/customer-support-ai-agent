# src/

## What

The agent's source code: intent taxonomy, classifier, retrieval-grounded reply drafter, and
escalation policy. This is the code imported and run at agent inference time (as opposed to
`scripts/`, which holds one-off data preparation).

## Why

Separated from `scripts/` so the agent's actual classify/draft/escalate logic stays isolated from
setup/data-prep code that only runs once during pipeline setup. This also keeps `src/` the natural
place to import from when building the eval harness in `eval/`.

## How

Modules here will be composed by an eventual top-level agent entry point (not yet added): a
message comes in, is classified against `INTENT_LABELS`, relevant historical resolutions are
retrieved, a reply is drafted grounded in those, and the escalation policy decides
auto-handle vs. escalate.

## File responsibilities

### `intents.py`
- **What it does**: Defines the 7-intent taxonomy (`INTENTS` dict of label -> description,
  `INTENT_LABELS` list) used to classify incoming AppleSupport customer messages:
  `software_bug`, `battery_performance`, `account_security`, `billing_purchase`,
  `hardware_issue`, `how_to_info`, `store_order_service`.
- **Purpose**: Single source of truth for the taxonomy, so the classifier prompt, golden-set
  labeling, and report all reference the same definitions instead of drifting.
- **How it was derived**: By sampling and manually reading ~120 real `customer_msg` values from
  `data/processed/apple_triples.csv` (two samples of 60, different random seeds) rather than
  guessing categories upfront. The dataset is iOS-11-era and dominated by update-related bugs and
  battery complaints, which the taxonomy reflects (battery_performance is split out from
  software_bug specifically because of how frequent it was in the sample).
- **Scope note**: Classifies the first customer message in a thread only. Short conversational
  follow-up turns (e.g. "Yes", "Thanks!") that appear in the `customer_followup` field of
  `apple_triples.csv` are not new intents and are explicitly out of scope.
- **Depends on**: Nothing (pure data/constants module).
- **Depended on by**: Not yet consumed by other code — will be imported by the classifier module
  once added.

### `llm.py`
- **What it does**: Wraps the Gemini API behind a single `call_llm(system, user, model, temperature)`
  function using Google's native `google-genai` SDK. Lazily constructs and caches a `genai.Client`
  from `GEMINI_API_KEY` (loaded via `python-dotenv` from a repo-root `.env` file). Default model is
  `gemini-3.6-flash`. Retries up to `MAX_RETRIES` (4) times with exponential backoff
  (`RETRY_BASE_DELAY_SECONDS` doubling each attempt) on transient `429`/`503` API errors before
  raising.
- **Purpose**: Single choke point for all LLM calls (classifier, drafter, escalation reasoning, eval
  judge) so the rest of the codebase never imports the `google-genai` SDK directly — switching
  providers later means changing this file only, not every caller.
- **Provider history**: Originally written against the OpenAI API (per initial project decision),
  then switched to Gemini per user request. The `google-genai` package was added to
  `requirements.txt` in place of `openai`, which was removed since both providers are not
  supported simultaneously.
- **Rate-limit finding (important, still open)**: This API key's free tier caps `gemini-3.6-flash`
  at 5 requests/minute (`429 RESOURCE_EXHAUSTED`, confirmed empirically). The retry/backoff above
  was added to absorb this, but a real 300-call run against Google's own usage dashboard showed
  only a **14.78% success rate** (379 requests, ~85% failing with 429) — retry/backoff delays a
  failing call's *retries*, it does not throttle the *rate new calls are issued at*, so under
  sustained load the loop kept submitting faster than 5/min and mostly exhausted `MAX_RETRIES`
  before succeeding. The run was killed rather than left to finish on mostly-failed data. A
  candidate alternative, `gemini-3.1-flash-lite`, has no quota cap on this key but is slower
  per-call (~7s vs ~2-4s) and prone to transient `503` "high demand" errors — evaluated but not
  adopted. **Net effect**: this wrapper is not currently safe to call in a loop of more than a
  handful of requests without an actual rate limiter (e.g. a token-bucket enforcing >=12s between
  calls), which has not been built. `scripts/classify_triples.py` no longer uses this wrapper at
  all — see `keyword_classifier.py` below for what replaced it.
- **Known limitation**: The SDK prints a benign stderr warning about automatic function calling
  (AFC) on every call, recommending the Chat API pattern instead of `generate_content`. Left as-is
  since it doesn't affect correctness and switching to the Chat API is a larger change than this
  wrapper's current scope.
- **Depends on**: `google-genai`, `python-dotenv`; `GEMINI_API_KEY` must be set in `.env`.
- **Depended on by**: Not currently used by any script (see rate-limit finding above). Will be
  imported by the classifier, drafter, and escalation policy modules once added — those call sites
  will need real rate-limiting, not just retry/backoff, if they run in a loop over many examples.

### `keyword_classifier.py`
- **What it does**: `classify_keyword(msg)` — a substring/keyword-match heuristic that assigns one
  of the 7 `INTENT_LABELS` (or `"unknown"` for empty/NaN input) to a customer message, with
  `software_bug` as the fallback default. `KEYWORD_RULES` is an ordered list of
  `(label, [keywords])` pairs checked in order, first match wins.
- **Purpose**: Stand-in for LLM-based classification, used only to build a stratified sampling pool
  for the golden evaluation set. It is explicitly **not** a ground-truth classifier — every
  golden-set example's intent is confirmed or corrected by hand during labeling (see `eval/`), so
  this only needs to separate the 5,000 `apple_triples.csv` rows into roughly-populated buckets
  well enough that stratified sampling can draw enough candidates per intent.
- **Why it exists**: Replaces an LLM-based `scripts/classify_triples.py` that turned out to be
  infeasible on the free-tier Gemini key — see `llm.py`'s rate-limit finding above for the full
  investigation (a live 300-call run measured only 14.78% success). This heuristic runs in seconds
  over all 5,000 rows with zero API calls and zero cost.
- **How keywords were chosen**: By checking real match counts against
  `data/processed/apple_triples.csv` before committing to them (all 7 buckets found 100+ matches),
  not guessed blind. A manual spot-check of 5 examples per predicted intent showed generally
  correct groupings with some expected noise (e.g. a UI question about the lock screen was
  misclassified as `account_security` due to the word "locked") — acceptable for a stratification
  pre-pass, not acceptable as a final label.
- **Depends on**: `src/intents.py` (for `INTENT_LABELS`, and an assertion that every rule label is
  a real intent).
- **Depended on by**: `scripts/classify_triples.py`.
