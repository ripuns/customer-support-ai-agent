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
- **Rate-limit finding and fix**: This API key's free tier caps `gemini-3.6-flash` at 5
  requests/minute (`429 RESOURCE_EXHAUSTED`, confirmed empirically). A retry/backoff-only version of
  this wrapper was tried first and failed badly: a real 300-call run against Google's own usage
  dashboard showed only a **14.78% success rate** (379 requests, ~85% failing with 429), because
  backoff delays a failing call's *retries* without throttling the *rate new calls are issued at* —
  under sustained load the loop kept submitting faster than 5/min. The actual fix is
  `_throttle()`/`MIN_SECONDS_BETWEEN_CALLS` (12.5s, just over the 12s that 5/min implies): it blocks
  before every call so the request rate itself never exceeds the quota, with retry/backoff kept as
  a second line of defense for genuinely transient errors. Verified with a live 3-call test showing
  consistent ~12.6s spacing and no 429s. A candidate alternative, `gemini-3.1-flash-lite`, has no
  quota cap on this key but is slower per-call (~7s vs ~2-4s) and prone to transient `503` "high
  demand" errors — evaluated but not adopted. **Cost of the fix**: every call now takes at least
  12.5s, so a loop over the 175-row golden set takes ~37 minutes minimum — acceptable for a one-off
  eval run, not for anything latency-sensitive.
- **Known limitation**: The SDK prints a benign stderr warning about automatic function calling
  (AFC) on every call, recommending the Chat API pattern instead of `generate_content`. Left as-is
  since it doesn't affect correctness and switching to the Chat API is a larger change than this
  wrapper's current scope.
- **Depends on**: `google-genai`, `python-dotenv`; `GEMINI_API_KEY` must be set in `.env`.
- **Depended on by**: `classifier.py`. Will also be imported by the drafter and escalation policy
  modules once added.

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

### `classifier.py`
- **What it does**: `classify_intent(customer_msg)` — the agent's real intent classifier. Sends a
  few-shot prompt (built from `src.intents.INTENTS`) through `src.llm.call_llm` and parses a
  two-line `INTENT: <label>` / `CONFIDENCE: <0-100>` response via regex. Returns
  `{"intent": str, "confidence": int}`. Returns `{"intent": "unknown", "confidence": 0}` for
  empty/invalid input or if the model's response can't be parsed into a recognized label —
  callers should treat `"unknown"` as a signal to escalate rather than guess.
- **Purpose**: This is the actual LLM-based classifier the agent uses at inference time (as opposed
  to `keyword_classifier.py`, which only exists to build the golden-set sampling pool). Returning a
  confidence score alongside the label is deliberate: the escalation policy (not yet added) needs
  low classifier confidence as one of its inputs for deciding auto vs. escalate.
- **Verified against real data**: Spot-tested against 5 sampled `eval/golden_set.csv` rows —
  correct format parsing and confidence scores on every call, no 429s (throttling in `llm.py`
  held). 2/5 matched the hand-labeled `intent_label` on this tiny sample; the mismatches were on
  genuinely ambiguous messages (e.g. a complaint mixing a store-visit anecdote with a software
  complaint) rather than clear classifier errors. Real accuracy numbers come from the full
  evaluation harness (not yet added) against all 175 golden-set rows, not from this spot check.
- **Depends on**: `src/intents.py`, `src/llm.py`.
- **Depended on by**: Not yet consumed by other code — will be used by the (not yet added) drafter,
  escalation policy, and evaluation harness.

### `retrieval.py`
- **What it does**: `RetrievalIndex` loads `data/processed/apple_triples.csv` (dropping the 29 rows
  with NaN `customer_msg`), builds a TF-IDF matrix over `customer_msg` (scikit-learn
  `TfidfVectorizer`, English stopwords removed, capped at 5,000 features) at construction time, and
  exposes `.query(customer_msg, k=3)` — cosine similarity against the matrix, returning the top-k
  historical `{customer_msg, brand_reply, customer_followup, similarity}` triples, most similar
  first.
- **Purpose**: This is what "grounds" the agent's drafted replies in how AppleSupport has
  historically resolved similar issues, per the assignment's core requirement — the drafter (not
  yet added) will feed these retrieved triples to the LLM as context rather than letting it
  generate a reply from general knowledge alone.
- **Why TF-IDF instead of an embeddings API**: Embedding all 5,000 `apple_triples.csv` rows through
  the Gemini embeddings API would hit the same free-tier rate-limit wall documented in `llm.py`'s
  rate-limit finding, at 5,000x the scale that made `scripts/classify_triples.py` infeasible — would
  take many hours even with correct throttling. TF-IDF is local, free, and instant (index build:
  ~0.1s over ~5,000 rows) at the cost of missing paraphrases with no word overlap (a known,
  documented limitation, not an oversight).
- **Verified against real data**: A battery-drain query returned a near-identical historical
  complaint at 0.959 cosine similarity with a directly relevant reply. A less common
  billing/subscription-refund query returned lower but still topically relevant matches
  (0.36-0.39 similarity) — reflecting that billing intents are more sparsely represented in this
  dataset than the dominant battery/bug complaints (see `intents.py`'s taxonomy notes on data
  skew), not a retrieval bug.
- **Depends on**: `data/processed/apple_triples.csv` (produced by `build_threads.py`);
  `scikit-learn`.
- **Depended on by**: Not yet consumed by other code — will be used by the (not yet added) drafter.
