# Hiver SDE Intern Assignment — Session Summary (as of 2026-09-16)

## What this is
Repo: `D:\customer-support-ai-agent` — AI customer support agent for **AppleSupport**, built from
the Kaggle "Customer Support on Twitter" dataset, for a Hiver SDE Intern take-home assignment.
Deadline: 6 days total, last 2 for QnA prep, so ~4 build days. We are currently on roughly day 3-4.

**Operating rules in effect** (user shared a rules.pdf early on — follow these strictly):
- One logical step at a time, then stop and wait for review. No batching, no silent continuation.
- No guessing — ask rather than invent scope/behavior.
- No unrequested refactoring or scope changes without approval.
- No commits/pushes unless explicitly asked (user has been committing manually themselves —
  git log shows commits like "feat(drafter): ...", "docs: update project and module documentation").
- Never rewrite git history.
- **Mandatory directory README rule**: every directory touched gets a README.md (What/Why/How/File
  Responsibilities) created/updated in the same step. Root README needs stack-choice rationale.
- Flag problems found outside scope rather than silently fixing them.
- User asked to keep answers SHORT/concise in chat once mid-session ("answer in short from now on").

## Assignment requirements (for reference)
Pick a brand from the dataset, build an agent that classifies intent, drafts a grounded reply, and
decides auto-handle vs escalate with a stated reason. Deliverables: reproducible repo (<15 min
repro), golden eval set (150-250 hand-labeled examples), eval harness with automated metrics + LLM-
judge + judge-vs-human agreement, a 6-page report (problem framing, baselines, failure analysis,
mandatory "what's misleading about my headline number" section, next-week plan), a decision log
(10-15 non-obvious decisions), submitted via a Notion form with repo link + report.

## Key decisions made (chronological, for the decision log)
1. **Brand: AppleSupport** — chosen after inspecting `scripts/inspect_brands.py` output: ~107k
   replies, ~34% customer-follow-up rate, narrower issue surface than e-commerce brands like
   Amazon.
2. **LLM provider: Gemini**, not OpenAI — user had no OpenAI billing credits, switched preference
   to Gemini (Google AI Studio). `src/llm.py` wraps it behind `call_llm()`.
3. **Thread definition**: triples (customer→brand→customer follow-up) as primary grounding/golden
   set source; no-followup pairs kept aside as a secondary dataset for failure-analysis ("what
   reply patterns correlate with customer abandonment").
4. **7-intent taxonomy** (`src/intents.py`), derived by manually sampling ~120 real customer
   messages, not guessed: `software_bug`, `battery_performance`, `account_security`,
   `billing_purchase`, `hardware_issue`, `how_to_info`, `store_order_service`.
   - Battery split out from software_bug specifically because of how common it was (iOS-11-era
     dataset, dominated by update/battery complaints).
   - Short conversational follow-ups ("Yes", "Thanks!") explicitly out of scope for classification.
5. **Retrieval: TF-IDF + cosine similarity**, not embeddings API — deliberately avoided another
   rate-limited Gemini dependency at 5000x the scale that already caused problems (see below).
6. **Escalation policy: rule-based**, not another LLM call — needs to be interpretable/defensible
   (assignment explicitly wants "a stated reason", and user will be asked to explain live).
7. **Simple baseline's escalation rule**: deliberately reused the "escalate all
   account_security/billing_purchase" rule that `escalation.py` itself tried and rejected — chosen
   as a realistic "naive first attempt" for comparison, not a strawman.
8. **Harness default subsample**: `--n 25` default (not full 175) for iteration speed; `--full`
   flag for the real report numbers, run once and committed as `eval/results_full.json`.

## Major incidents / non-obvious findings (important for decision log + report)

### Rate limits (multiple layers discovered)
- `gemini-3.6-flash` free tier: **5 requests/minute** cap. First discovered when a naive
  parallel-classification script got 14.78% success rate (verified via Gemini's own usage
  dashboard) because retry/backoff delays failed retries but doesn't throttle the *rate new calls
  are issued at*. Fixed properly in `src/llm.py` with a `_throttle()` that blocks *before* every
  call (`MIN_SECONDS_BETWEEN_CALLS`).
- `gemini-3.6-flash` ALSO has a **20 requests/DAY** cap (separate quotaId
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`) — discovered when a 25-row harness eval run
  died mid-way after exhausting the whole day's quota. No throttling fixes a daily cap.
- **Fix**: switched `DEFAULT_MODEL` to `gemini-3.1-flash-lite` — verified via burst-testing 30+
  calls with zero per-minute or per-day errors, ~3.7s/call (faster than expected). Current
  `MIN_SECONDS_BETWEEN_CALLS = 3.0` (courtesy margin, not a measured limit).
- Lesson pattern if it recurs: check `client.models.list()` for another model, burst-test it before
  committing, watch for BOTH per-minute AND per-day 429 errors (different quotaIds in the error
  JSON).

### Keyword-based pre-classification (replaced an LLM approach)
- `scripts/classify_triples.py` originally used the LLM (throttled, all 5000 rows) — infeasible at
  the 5/min cap (~17 hrs). Replaced with `src/keyword_classifier.py`, a free instant heuristic
  (substring match, ordered rules, `software_bug` fallback). NOT claimed as ground truth — only
  used to build a stratified sampling pool. Keywords were chosen by checking real match counts
  against the data (100+ matches per bucket) before committing.

### Golden set encoding corruption
- `eval/golden_set.csv` got mixed UTF-8/cp1252 encoding when edited in Excel (a smart-quote
  character got saved as a raw cp1252 byte `\x92`), breaking `pd.read_csv` entirely. Fixed via a
  byte-level decode pass (try UTF-8 first, fall back to cp1252 per-byte only where UTF-8 fails,
  rewrite as clean UTF-8). No content lost, verified 0 replacement chars, all 175 rows intact.
  Documented in `eval/README.md`. If it recurs: avoid typing curly quotes in the editor, or save
  explicitly as UTF-8 CSV.

### Escalation policy design — the most important finding so far

#### Round 1: topic-based rule tried and rejected
- First version escalated automatically for `account_security`/`billing_purchase` intents
  (topic-based rule).
- Tested against the 35 hand-labeled golden-set rows: only **24/35 (69%)** — most of those threads
  actually resolved fine with a standard reply; the golden-set labeling standard says escalate
  based on whether *this specific thread* resolved cleanly, not by topic category.
- Removed the intent-based rule, expanded `RED_FLAG_PHRASES` (prior-fix-failed language, severity
  signals, lockout/data-loss language) based on what was actually in the genuinely-escalated rows.
- Re-tested: **32/35 (91%)**.

#### Round 2: overfitting discovered on a fresh sample
- **BUT**: `eval/run_harness.py`'s `--n 25` run (a FRESH random sample, disjoint from the 35 tuning
  rows) showed the real agent's escalation policy scoring **0 precision / 0 recall** (TP=0) against
  11 true escalate cases.
- First suspected eval leakage (golden-set messages exist verbatim in the retrieval index used by
  `escalation.py`'s "no good retrieval match" signal) — found and fixed via
  `RetrievalIndex.query(..., exclude_exact_match=True)`, threaded through `drafter.py`,
  `escalation.py`, and the harness.
- **Re-ran after the fix: IDENTICAL 0/0 result.** So leakage was a real bug worth fixing but NOT
  the actual cause.
- **Direct diagnosis**: on the 11 true-escalate rows in this sample, classifier confidence was
  always >=85 (never triggers the 60 threshold) and **none** contained any `RED_FLAG_PHRASES`
  substring.
- **Conclusion**: the escalation policy is overfit to the 35 tuning rows and doesn't generalize —
  this is probably the single most important, most honest finding for the report's failure
  analysis and mandatory "what's misleading about my headline number" section.

#### Decision (2026-09-16, deadline day): fix it, not just report it
- Discussed fixing vs. leaning into the finding for the report.
- Decided to actually fix it — the finding is real and worth reporting either way, but a keyword
  list was diagnosed as fundamentally unable to generalize (real escalate triggers in this dataset
  are semantic — "that didn't do anything", "nope not on shuffle", "sadly that's not it" — not
  shared vocabulary), so patching more keywords would still be shallow.
- Replaced `RED_FLAG_PHRASES`/`_find_red_flags` with `_llm_red_flag_check`, one `call_llm` asking
  whether the single incoming `customer_msg` shows a failed prior fix, lockout/data-loss, or
  explicit request for a human — judged semantically.
- Confidence-threshold and retrieval-grounding checks were left unchanged (not diagnosed as the
  problem). See `src/escalation.py` and `src/README.md` for the full writeup.

**Scope decision**: `decide_escalation` only ever receives the single `customer_msg` (its real
call site has no `brand_reply`/`customer_followup` — those only exist for retrospective golden-set
labeling, not at live decision time).
- Considered extending to multi-turn conversation history (user's point: a customer who has
  messaged before should have that history considered) but explicitly scoped out for today given
  the deadline — noted as next-week plan work instead.

**Validated on small samples (n=3, 8, 18) using remaining free-tier quota**: fix moved escalation
off the hard 0/0 floor.
- n=18 result: precision 0.5, recall 0.25 (2 TP, 2 FP, 6 FN), vs. 0/0 before the fix.
- Real improvement, but recall still weak and still behind the simple baseline's 0.8/0.5 on this
  sample.

**Why the simple baseline still "beats" the LLM fix on small samples — important, non-obvious**:
- `simple_escalate(intent)` (`src/baseline_simple.py`) escalates purely by *topic category*
  (always escalates `account_security`/`billing_purchase`) — this is the exact topic-based rule
  that was tried and rejected for the real policy earlier (only 69% on the 35 tuning rows,
  escalates many threads that actually resolved fine).
- It's a crude, broad net: on a small random sample where true escalate cases happen to cluster in
  those topics, it gets lucky and looks good, without being a defensible policy.
- The LLM fix is intentionally *stricter* — it only escalates on a real signal in the single
  message itself (per the actual labeling standard), so it correctly abstains on ambiguous cases
  rather than escalating a whole topic.
- Net effect: higher precision-per-topic-guess isn't the same as being right for the right reason.
- This "simple baseline beats the real agent" result is itself a good candidate for the report's
  "what's misleading about my headline number" section — the comparison looks bad for the real
  agent on a small sample but the simple baseline's number is inflated by a policy already known to
  be wrong at scale.

Full validation still needs the `--full` 175-row harness run once LLM API access allows it (limited
free-tier quota only allowed small smoke tests so far) — do not treat n=18 numbers as final report
numbers.

### Golden-set labeling standard (converged through iterative review)
`auto_or_escalate` decided per-thread based on whether the actual `brand_reply`/`customer_followup`
show clean resolution, NOT by topic or tone:
- **Escalate when**: customer confirms a suggested fix did NOT work; account lockout/data
  loss/financial-policy decisions a bot can't grant; customer explicitly asks for something needing
  human judgment (ETA, exceptions, replacement); multiple channels/attempts already failed.
- **Do NOT escalate purely because**: topic is common across many customers (not a per-thread
  signal); tone is negative/profane without real severity; message is merely long/detailed.
This standard is documented in `eval/README.md` and directly encoded into `src/escalation.py`'s
`RED_FLAG_PHRASES` (though as noted above, that encoding doesn't generalize well).

## Current state of the codebase

### Pipeline (all built, working, documented)
- `scripts/download_data.py` → `data/raw/twcs.csv` (gitignored, ~516MB, kagglehub)
- `scripts/inspect_brands.py` → informed brand choice (no output file)
- `scripts/build_threads.py` → `data/processed/apple_triples.csv` (5000 rows, seeded sample) +
  `apple_no_followup.csv` (75167 rows, full). Had a perf bug (O(n×m) naive scan) fixed with
  pre-grouped dict lookup.
- `scripts/classify_triples.py` → `data/processed/apple_triples_classified.csv` (keyword-based now,
  instant, all 5000 rows)
- `scripts/sample_golden_set.py` → `eval/golden_set.csv` (175 rows, 25/intent stratified, seeded)
- `scripts/_draft_labels.py` — one-off, NOT a reusable pipeline script (underscore-prefixed,
  documented in eval/README.md not scripts/README.md). Contains the 141 assistant-drafted label
  decisions as an auditable dict. Currently in `.gitignore` (user added it there — flagged earlier
  as a possible inconsistency with eval/README.md treating it as documented project history; not
  yet resolved either way).

### Agent core (`src/`, all 3 required components done)
- `src/intents.py` — 7-intent taxonomy, `INTENTS` dict + `INTENT_LABELS` list
- `src/llm.py` — Gemini wrapper, `call_llm(system, user, model, temperature)`, throttled +
  retry/backoff, `DEFAULT_MODEL = "gemini-3.1-flash-lite"`
- `src/keyword_classifier.py` — free heuristic, used for golden-set stratification pool AND as the
  simple baseline's classifier
- `src/classifier.py` — real LLM classifier, `classify_intent(msg)` → `{intent, confidence}`
- `src/retrieval.py` — `RetrievalIndex` class, TF-IDF+cosine over `apple_triples.csv`, `.query(msg,
  k=3, exclude_exact_match=False)` — the exclude_exact_match param was added later to fix eval
  leakage
- `src/drafter.py` — `draft_reply(msg, intent, retrieval_index, k=3, exclude_exact_match=False)` →
  `{reply, grounded_on}`
- `src/escalation.py` — `decide_escalation(msg, intent, confidence, retrieval_index,
  exclude_exact_match=False)` → `{decision, reasons}`. Known overfitting issue, see above.
- `src/judge.py` — LLM-as-judge, `judge_reply(customer_msg, agent_reply)` → `{grounded, correct,
  actionable}` each 1-5
- `src/baseline_trivial.py` — fixed majority-intent (`software_bug`), one canned reply, always-auto
- `src/baseline_simple.py` — keyword_classifier + 7 template replies + intent-risk escalation rule
  (the rejected rule from escalation.py, reused deliberately)

### Eval (`eval/`)
- `eval/golden_set.csv` — 175 rows. **Labeling status: rows 2-100 (spreadsheet numbering, header=
  row1) reviewed/confirmed by user, `[DRAFT]` prefix removed from those. Rows 101-175 (76 rows)
  still `[DRAFT]`, NOT yet reviewed.** User was reviewing in batches; I was flagging inconsistencies
  for them to fix (e.g. escalating on "many people report this" or tone alone — both corrected).
- `eval/run_harness.py` — runs trivial/simple/real-agent against a golden-set sample, computes
  intent accuracy, escalation accuracy/precision/recall, LLM-judge reply quality. `--n` (default
  25) or `--full` (175). Writes to `eval/results_preview.json` or `eval/results_full.json`.
- `eval/results_preview.json` — LATEST real run (n=25, post-leakage-fix). Numbers:
  - Trivial: intent 20%, escalation 56%/0.0/0.0, reply quality 5.0/5.0/5.0 (fixed reply)
  - Simple: intent 88%, escalation 60%/0.56/0.45, reply quality 4.48/4.32/4.40
  - Real agent: intent 64%, escalation 48%/0.0/0.0, reply quality 5.0/4.96/5.0
  - **Simple baseline currently beats the real agent on intent accuracy AND escalation** — an
    honest, reportable result, not hidden.
- NOT YET RUN: `--full` (all 175 rows) — needed for actual report numbers, blocked on finishing
  golden-set review first (rows 101-175 still draft) and possibly on fixing the escalation policy.

### Docs
- Root `README.md`, `src/README.md`, `scripts/README.md`, `eval/README.md` all kept up to date
  after every step per the mandatory-README rule — these are thorough and contain most of the
  "why" reasoning already; worth reading directly rather than re-deriving in a new session.
- `.env` / `.env.example` — `GEMINI_API_KEY` (not `OPENAI_API_KEY`, provider was switched)
- `requirements.txt` — has `google-genai` not `openai`

## Current blocker (as of 2026-09-16, deadline day)
- Gemini free tier stopped being usable (rate limits — see "Rate limits" section above).
- Attempted to enable Google Cloud paid billing to switch to a paid Gemini tier with zero code
  changes; billing signup is failing with error `OR_BACR2_59` ("Billing setup can't be completed" /
  "we believe someone else may be trying to access your account") despite a verified payments
  profile and successful UPI mandate creation.
- This is a known Google-side fraud-check false-positive pattern for India signups, not something
  fixable by retrying the form repeatedly — retrying more may extend the lockout.
- **Next step if GCP retry (after a cooldown) still fails**: switch provider to Anthropic (Claude
  Haiku 4.5) or OpenAI (GPT-4o-mini) via `console.anthropic.com` / `platform.openai.com` — both are
  typically fast to set up from India and don't share this failure mode.
  - Would require a small change to `src/llm.py`'s `call_llm()` to swap the client; the rest of
    the pipeline (classifier, drafter, judge, escalation) is provider-agnostic since everything
    routes through that one function.
- **This blocks**: the `--full` harness run, judge-vs-human agreement check, and validating the
  escalation policy fix below — all need working LLM calls.

## What's NOT done yet (remaining work, roughly in priority order)
1. ~~Finish golden-set review~~ — **DONE 2026-09-16.** All 175 rows reviewed, `[DRAFT]` tags
   stripped.
   - One bug found and fixed during review: row for customer id 889104 had an `escalate_reason`
     copy-pasted from an unrelated purchase-related row (described a missing purchase instead of
     the actual Bluetooth connectivity issue) — corrected.
2. ~~Decide escalation policy fix vs. lean into finding~~ — **DECIDED 2026-09-16: fixed.** See
   "Escalation policy design" section above for the full writeup.
   - **Not yet validated** — blocked on LLM API access, see "Current blocker" above.
3. **Judge-vs-human agreement check** — required by the assignment, not yet built. Need to hand-
   score a subset of judge_reply outputs and compute agreement (e.g. Cohen's kappa or simple %).
   Blocked on LLM API access.
4. **Run `--full` harness** once LLM API access is restored, to get the actual report numbers
   (including validating the escalation policy fix above). Save as `eval/results_full.json`.
5. **Report** (`report/`, max 6 pages) — not started. Sections required: problem framing (what
   "good" means for AppleSupport, what was deliberately not built), results vs. both baselines,
   top-5 failure analysis with real examples, mandatory "what's misleading about my headline
   number" (the escalation-overfitting finding is a strong candidate here, plus the eval-leakage
   near-miss, plus TF-IDF vs embeddings tradeoff), next-week plan.
6. **Decision log** (10-15 non-obvious decisions + why) — lots of natural material already exists
   from this session (see "Key decisions" and "Major incidents" above), just needs compiling into
   the final format.
7. **README reproducibility pass** — confirm a clean-clone run actually completes in the stated
   time; current README's step-by-step reproduce section should already be accurate but hasn't been
   tested end-to-end from a truly clean clone.
8. **Submit** via the Notion form (URL was in the original assignment prompt) with repo link + report.

## How to resume in a new session
1. Read this file first.
2. Read the 4 README.md files (root, src/, scripts/, eval/) for full technical detail — they're
   comprehensive and were kept current throughout.
3. Check `eval/golden_set.csv` for current `[DRAFT]` status (`grep -c '\[DRAFT\]'` or similar) to
   confirm where review left off.
4. Check `git log` and `git status` to see what's been committed vs. still pending (user commits
   manually; agent has never committed).
5. Ask the user directly what they want to tackle next from the "not done yet" list above — don't
   assume; there were open decisions (e.g. fix escalation policy vs. lean into the finding) that
   were not resolved as of this summary.
6. Continue following the same operating rules (one step at a time, ask before scope changes, no
   commits, README-per-directory rule, etc.) — these are durable, not session-specific.
