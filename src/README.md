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
- **Rate-limit history (two separate caps found, one at a time)**:
  1. **Per-minute cap**: this key's free tier capped `gemini-3.6-flash` at 5 requests/minute (`429
     RESOURCE_EXHAUSTED`, confirmed empirically). A retry/backoff-only version of this wrapper was
     tried first and failed badly: a real 300-call run against Google's own usage dashboard showed
     only a **14.78% success rate** (379 requests, ~85% failing with 429), because backoff delays a
     failing call's *retries* without throttling the *rate new calls are issued at*. Fixed with a
     `_throttle()` that blocks before every call so the request rate itself never exceeds the quota.
  2. **Per-day cap (found later, more serious)**: `gemini-3.6-flash` on this key also has a hard
     cap of **20 requests/day total** (a *different* 429, quotaId
     `GenerateRequestsPerDayPerProjectPerModel-FreeTier`) — discovered when a 25-row harness
     evaluation run (`eval/run_harness.py`) died partway through after exhausting the day's quota.
     No amount of throttling or backoff can work around a daily cap. This made `gemini-3.6-flash`
     unusable as the default for anything beyond a handful of calls per day.
  - **Fix**: switched `DEFAULT_MODEL` to `gemini-3.1-flash-lite`. Verified with a live 30-call burst
    (on top of ~40+ other calls already made that same day across testing) — 30/30 succeeded, no
    per-minute or per-day quota errors, averaging ~3.7s/call (faster than the ~7s originally
    estimated in earlier testing). `MIN_SECONDS_BETWEEN_CALLS` was correspondingly relaxed from
    12.5s to 3.0s (no per-minute cap observed on this model; a small spacing kept as a courtesy
    margin, not because a specific limit was measured).
  - **Residual risk**: `gemini-3.1-flash-lite`'s daily quota was never hit in testing, but its exact
    limit is unknown (Google does not expose it directly) — a `--full` (175-row) harness run could
    still hit an undiscovered daily cap partway through. If it does, the fix is the same playbook
    used here: check `client.models.list()` for another candidate model, burst-test it, switch.
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
  to `keyword_classifier.py`, which only exists to build the golden-set sampling pool). The
  confidence score is kept in the return value (other callers/analysis may still want it), but as of
  the escalation-policy fix (see `escalation.py`) it is no longer used as an escalation signal —
  confidence in the *topic label* was found not to correlate with whether a message actually needs a
  human (see `escalation.py`'s "Fix 2" entry for the full diagnosis).
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
- **Depended on by**: `drafter.py`, `escalation.py`, `eval/run_harness.py`.
- **Eval leakage — found and fixed**: `eval/golden_set.csv` was sampled *from* `apple_triples.csv`,
  so every golden-set `customer_msg` exists verbatim inside this retrieval index — querying with a
  golden-set message would always find itself first at similarity 1.00. This was not just a
  cosmetic issue: an early `eval/run_harness.py` run showed the real agent's escalation policy
  scoring **0 precision / 0 recall** on a 25-row sample, because `escalation.py`'s "no sufficiently
  similar historical resolution found" signal could never fire against a message that was always
  its own top match — the leakage silenced one of the policy's three signals entirely. Fixed by
  adding `exclude_exact_match: bool` to `query()` — when `True`, any indexed row with a
  character-for-character identical `customer_msg` is masked out (similarity set to -1) before the
  top-k is taken. `drafter.py` and `escalation.py` both accept and forward this parameter;
  `eval/run_harness.py` passes `exclude_exact_match=True` on every call during evaluation. Verified
  the fix actually changes behavior: the same 5 escalate-labeled rows that showed similarity 1.00
  before the fix showed real similarities of 0.21-0.49 after — still topically related (the dataset
  is full of similar iOS-11-era complaints) but no longer an exact self-match.
- **Residual limitation (documented, not fixed)**: retrieval-based signals are still measured
  against the rest of the ~5,000-row `apple_triples.csv` pool, which may contain near-duplicate or
  paraphrased versions of a given golden-set message from the same era of complaints — so results
  may still be somewhat optimistic versus a truly unseen message in production. This is a smaller,
  harder-to-fully-eliminate version of the same underlying issue, worth carrying into the report's
  "what's misleading about my headline number" section rather than claiming it's fully solved.

### `drafter.py`
- **What it does**: `draft_reply(customer_msg, intent, retrieval_index, k=3, exclude_exact_match=False)`
  — queries the given `RetrievalIndex` for the top-k similar historical triples, filters them to
  `MIN_SIMILARITY_TO_USE` (0.15) so weak/irrelevant matches aren't used as grounding, builds a
  system prompt naming the classified intent (with its `INTENTS` description) and formatting the
  filtered examples as few-shot context, then calls `src.llm.call_llm` (temperature 0.3, some
  variation allowed since this is generative, unlike the classifier). Returns
  `{"reply": str, "grounded_on": list[dict]}` — `grounded_on` is the actual filtered examples used,
  so callers/eval code can see (and score) what grounding was available for this reply, including
  the empty-list case where nothing sufficiently similar was found. `exclude_exact_match` is
  forwarded straight to `RetrievalIndex.query` — pass `True` during evaluation (see
  `retrieval.py`'s eval leakage entry).
- **Purpose**: This is the "drafts a reply grounded in how the brand has historically resolved
  similar issues" requirement — the prompt explicitly instructs the model not to copy examples
  verbatim or invent unlisted specifics (case numbers, links), and to fall back to AppleSupport's
  real pattern of directing to DM when a public reply can't resolve the issue.
- **Verified against real data**: Tested end-to-end against golden-set row 1 (an AppleCare/Apple
  Store complaint) — with `exclude_exact_match=False`, retrieved 3 examples (top similarity 1.00,
  since this exact message exists in the source `apple_triples.csv` — the eval leakage case
  `retrieval.py` documents), and the drafted reply's opening closely mirrored the real historical
  reply's tone ("We'd like to look into this with you..."), then appropriately added a DM handoff.
- **Depends on**: `src/intents.py`, `src/llm.py`, `src/retrieval.py`.
- **Depended on by**: Not yet consumed by other code — will be used by the (not yet added)
  evaluation harness.

### `escalation.py`
- **What it does**: `decide_escalation(customer_msg, intent, retrieval_index,
  exclude_exact_match=False)` — a hybrid policy (rule-based signal + one LLM-judged signal)
  returning `{"decision": "auto" | "escalate", "reasons": [...]}`. Escalates when: intent is
  `"unknown"` (classifier couldn't parse a recognized label at all); the LLM judges `customer_msg`
  to show a red flag (prior fix attempt failed, lockout/data-loss language, explicit request for a
  human/exception — see `_llm_red_flag_check`); or `RetrievalIndex` finds no historical match above
  `MIN_SIMILARITY_FOR_GROUNDING` (0.15). `reasons` lists exactly which checks triggered, satisfying
  the assignment's "stated reason" requirement. `exclude_exact_match` is forwarded to
  `RetrievalIndex.query` — pass `True` during evaluation (see `retrieval.py`'s eval leakage entry;
  without this, the retrieval-grounding check above can never fire against a golden-set message,
  which is how the eval leakage bug was originally discovered).
- **Scope note**: `decide_escalation` only ever sees the single incoming `customer_msg`, matching
  its real call site (the agent decides per incoming message, before any reply exists — it has no
  access to `brand_reply`/`customer_followup`, which only exist for retrospective golden-set
  labeling). Messages that reference a failed prior attempt inline are covered ("I already tried
  restarting, still broken"); tracking risk signals across separate messages in a longer
  conversation is not — noted as future work below.
- **Purpose**: Deliberately rule-based rather than another LLM call — needs to be interpretable and
  defensible (a live-defense requirement of this assignment), and is built to directly encode the
  same standard used to hand-label `eval/golden_set.csv` (see that file's "Labeling standard"
  section) so the policy and the ground truth it's measured against share the same reasoning rather
  than being two independent guesses.
- **Design history — intent-based rule tried and rejected**: An earlier version escalated
  automatically for `account_security`/`billing_purchase` intents (reasoning: these routinely
  involve account access or money). Tested against the 35 hand-labeled `golden_set.csv` rows: only
  **24/35 (69%)** matched. Inspecting the mismatches showed most `account_security`/
  `billing_purchase` threads in this dataset actually resolved cleanly with a standard reply — the
  golden-set labeling standard explicitly says escalation should be decided per-thread (did it
  actually resolve?), not by topic category. Removed the intent-based rule and expanded
  `RED_FLAG_PHRASES` with the severity/failure-signal phrases that were actually present in the
  genuinely-escalated rows instead. Re-tested: **32/35 (91%)**.
- **Known remaining limitation (on the 35 tuning rows)**: 2 of the 3 still-mismatched rows are
  threads where a real severity signal exists but isn't expressible as a clean keyword (e.g. "killed
  my battery... freezes every few letters" — genuinely severe but phrased conversationally, no
  matching red-flag substring).
- **Overfitting confirmed on a fresh sample (important finding)**: `eval/run_harness.py`'s `--n 25`
  run — a random 25-row sample disjoint from the 35 rows this policy was tuned against — showed the
  real agent's escalation policy scoring **0 precision / 0 recall** (TP=0) against 11 true escalate
  cases. Diagnosed directly: classifier confidence on those 11 messages was always >=85 (never
  triggering `CONFIDENCE_ESCALATE_THRESHOLD`), and **none of the 11 contained any `RED_FLAG_PHRASES`
  substring**. This is not the eval-leakage bug (that was found and fixed separately — see
  `retrieval.py`) — it is the keyword-based policy genuinely failing to generalize past the specific
  phrasing of the rows it was iterated against. The 91% figure above describes fit to the 35 tuning
  rows, not real-world performance; the 25-row out-of-sample result is the more honest number. This
  is a central, concrete example for the report's failure analysis and "what's misleading about my
  headline number" section — a keyword list tuned by reading a small hand-labeled set will always
  risk this kind of overfitting.
- **Fix 1: `RED_FLAG_PHRASES` replaced with an LLM judgment call**. The keyword list was a ceiling,
  not a fixable bug — the real escalate triggers in this dataset ("that didn't do anything", "nope
  not on shuffle", "sadly that's not it") share meaning, not vocabulary, so no finite keyword list
  will generalize. `_llm_red_flag_check(customer_msg)` replaces `_find_red_flags`: one `call_llm`
  asking whether the message shows a failed prior fix, lockout/data-loss, or an explicit request for
  a human — judged semantically rather than by substring match.
- **Fix 2: `CONFIDENCE_ESCALATE_THRESHOLD` check removed entirely** (along with the `confidence`
  parameter from `decide_escalation`'s signature). Root cause: `classify_intent`'s confidence answers
  "how sure am I this is `battery_performance` vs. `software_bug`", not "does this need a human" — a
  message can be 95% obviously about one topic while still desperately needing escalation. Those are
  different questions; conflating them meant this signal was structurally incapable of catching
  escalation-worthy cases (confirmed: on the original 11 true-escalate rows, confidence was always
  >=85, never once triggering the threshold). Removed rather than reworked into a second
  self-reported number, since the LLM red-flag check already answers the right question directly.
- **Validated on small samples (n=3/8/18) using remaining free-tier quota** (full `--full` run
  blocked on LLM API access — see project status; do not treat these as final report numbers). On a
  fixed n=8 sample, results before and after removing the confidence check were identical
  (precision 0.5, recall 0.33, 1 TP/1 FP/2 FN), confirming the confidence signal genuinely never
  affected any outcome on that sample — consistent with the diagnosis above. On n=18: precision 0.5,
  recall 0.25 (2 TP, 2 FP, 6 FN) — a real improvement over the pre-fix 0/0 floor, but still behind
  `baseline_simple.py`'s topic-based escalation rule (0.8/0.5 on the same sample).
- **Why the simple baseline still "wins" on small samples — worth reporting explicitly**:
  `simple_escalate(intent)` escalates purely by topic category (always escalates
  `account_security`/`billing_purchase`) — this is the exact rule already tried and rejected for the
  real policy (see "Design history" above, only 69% on the 35 tuning rows, escalates many threads
  that resolved fine). On a small sample where true escalate cases happen to cluster in those
  topics, that broad net gets lucky. The LLM fix is intentionally stricter — it only escalates on a
  real signal in the message itself, so it correctly abstains on ambiguous cases rather than
  escalating a whole topic, costing recall on small samples. This "simple beats real agent" result
  is itself good material for the report's "what's misleading about my headline number" section —
  the simple baseline's number is inflated by a policy already known to be wrong at scale.
- **Depends on**: `src/retrieval.py` (uses it to compute the grounding-availability signal).
- **Depended on by**: `eval/run_harness.py`. This is also the last of the three core agent
  components (classify, draft, escalate) required by the assignment.

### `baseline_trivial.py`
- **What it does**: The "trivial baseline" required by the assignment ("results vs. at least two
  baselines: a trivial one and a simple one"). `trivial_classify` always returns
  `MAJORITY_INTENT` (`"software_bug"`, 77% of the classified `apple_triples.csv` pool);
  `trivial_draft` always returns one fixed `GENERIC_REPLY`; `trivial_escalate` always returns
  `FIXED_DECISION` (`"auto"`). No learning, no retrieval, no LLM calls — this is the floor every
  other component (simple baseline, real agent) needs to meaningfully beat.
- **Why always-auto rather than always-escalate**: Either is a valid trivial choice; always-auto
  was picked so the trivial baseline's escalation precision/recall are both meaningfully
  measurable against the golden set (always-escalate would make recall trivially 100% and give no
  useful comparison point).
- **Preview numbers (not final — golden set is still partially `[DRAFT]`)**: Against the current
  175-row `eval/golden_set.csv`: intent accuracy 38/175 (21.7%), escalation accuracy 105/175
  (60.0%). The escalation number looks deceptively decent only because `auto` happens to be the
  majority label in this labeled set (~58%) — flagged here as exactly the kind of number that
  needs the report's mandatory "what's misleading about my headline number" treatment: a real
  agent beating 60% by a small margin would not actually be demonstrating real escalation
  judgment.
- **Depends on**: Nothing (hardcoded constants only).
- **Depended on by**: Not yet consumed by other code — will be used by the (not yet added)
  evaluation harness.

### `baseline_simple.py`
- **What it does**: The "simple baseline" required by the assignment. `simple_classify` uses the
  real `keyword_classifier.classify_keyword` (not a fixed guess). `simple_draft` returns one of
  7 hand-written `TEMPLATE_REPLIES`, one per intent (plus an `"unknown"` fallback), rather than a
  single generic reply. `simple_escalate` uses `HIGH_RISK_INTENTS` (`account_security`,
  `billing_purchase` always escalate, everything else auto) — deliberately the same rule
  `escalation.py` tried first and rejected (see that file's design history), reused here on
  purpose as a realistic "naive first attempt" rather than a strawman.
- **Purpose**: Sits strictly between the trivial baseline (the floor) and the real agent, so results
  show a 3-point comparison (trivial / simple / real), not just two.
- **Preview numbers (not final — golden set is still partially `[DRAFT]`)**: Against the current
  175-row `eval/golden_set.csv`: intent accuracy 141/175 (80.6%) — notably strong, and verified not
  to be circular (the 35 hand-labeled rows alone score even higher, 30/35 = 85.7%, vs. 111/140 =
  79.3% on the assistant-drafted rows, so the keyword classifier isn't just agreeing with its own
  earlier `suggested_intent` guesses that seeded some of the drafts). Escalation accuracy 105/175
  (60.0%) — identical to the trivial baseline's escalation accuracy, but for a different and worse
  reason: precision 0.50 / recall 0.36 on the escalate class (TP=25, FP=25, TN=80, FN=45), meaning
  it's wrong half the time it does escalate and misses nearly two-thirds of real escalation cases,
  while `baseline_trivial.py`'s always-auto gets the same accuracy by design (it never even tries).
  This equal-accuracy-different-reasons result is a concrete, ready-made example for the report's
  "what's misleading about my headline number" section.
- **Depends on**: `src/keyword_classifier.py`, `src/intents.py`.
- **Depended on by**: Not yet consumed by other code — will be used by the (not yet added)
  evaluation harness.

### `judge.py`
- **What it does**: `judge_reply(customer_msg, agent_reply)` — LLM-as-judge scoring a drafted
  reply on 3 fixed dimensions, each 1-5: **grounded** (reflects real AppleSupport handling vs.
  generic boilerplate), **correct** (factually sound, doesn't invent specifics), **actionable**
  (gives a concrete next step). Returns `{"grounded": int|None, "correct": int|None,
  "actionable": int|None}` — `None` for a dimension only if the judge's response couldn't be
  parsed (missing data, not a failing score).
- **Purpose**: Satisfies the assignment's "LLM-as-judge rubric for reply quality" requirement. Kept
  to exactly 3 fixed dimensions (not open-ended scoring) so results are comparable across replies
  and so a human can be asked the same 3 questions for the judge-vs-human agreement check (not yet
  added) required by the assignment.
- **Verified against real data**: Tested on a clearly good vs. clearly bad synthetic reply pair for
  the same customer message. Good reply (asks for device/iOS version, AppleSupport's real pattern):
  5/5/5. Bad reply (generic acknowledgment, no substance): grounded=1, correct=5, actionable=1 —
  correctly low on the two dimensions the reply actually fails, while fairly not penalizing
  "correct" since the bad reply contains no false claims, just no substance. This directional
  sanity check is not the same as measuring real judge-vs-human agreement on golden-set examples,
  which the evaluation harness still needs to do.
- **Depends on**: `src/llm.py`.
- **Depended on by**: Not yet consumed by other code — will be used by the (not yet added)
  evaluation harness.
