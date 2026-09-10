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
  `gemini-3.6-flash`.
- **Purpose**: Single choke point for all LLM calls (classifier, drafter, escalation reasoning, eval
  judge) so the rest of the codebase never imports the `google-genai` SDK directly — switching
  providers later means changing this file only, not every caller.
- **Provider history**: Originally written against the OpenAI API (per initial project decision),
  then switched to Gemini per user request. The `google-genai` package was added to
  `requirements.txt` in place of `openai`, which was removed since both providers are not
  supported simultaneously.
- **Known limitation**: The SDK prints a benign stderr warning about automatic function calling
  (AFC) on every call, recommending the Chat API pattern instead of `generate_content`. Left as-is
  since it doesn't affect correctness and switching to the Chat API is a larger change than this
  wrapper's current scope.
- **Depends on**: `google-genai`, `python-dotenv`; `GEMINI_API_KEY` must be set in `.env`.
- **Depended on by**: Not yet consumed by other code — will be imported by the classifier,
  drafter, escalation policy, and eval judge modules once added.
