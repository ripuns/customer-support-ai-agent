"""Judge-vs-human agreement check.

Required by the assignment: measure how well the LLM-as-judge's reply-quality
scores (grounded/correct/actionable, 1-5) agree with a human rater's scores
on the same replies, so the judge's scores aren't trusted blindly.

Two-step workflow:
  1. `python eval/judge_agreement.py sample --n 50` -- since the harness
     checkpoint (eval/.checkpoints/real_agent_full.jsonl) only ever saved the
     judge's scores and not the reply text itself, this re-runs draft_reply()
     AND judge_reply() together for each sampled row (2 LLM calls/row) so the
     judge score in the CSV always describes the exact same reply text a
     human will read and score -- draft_reply() is not deterministic
     (temperature=0.3), so reusing the harness's original judge score against
     a freshly-reconstructed reply would silently mismatch the two. Writes a
     CSV with the customer message, the reply, and the judge's scores, plus 3
     empty columns for a human rater to fill in (human_grounded,
     human_correct, human_actionable).
  2. Hand-score the sampled rows in eval/judge_agreement_sample.csv (1-5 per
     dimension, same rubric as src/judge.py's SYSTEM_PROMPT).
  3. `python eval/judge_agreement.py score` -- reads the filled-in CSV and
     computes agreement between the judge and human scores: exact-match %,
     within-1-point %, and linear-weighted Cohen's kappa (the standard
     chance-corrected metric for ordinal 1-5 scales, where a 4-vs-5
     disagreement counts as a smaller error than a 1-vs-5 disagreement).
     Writes results to eval/judge_agreement_results.json.

API calls: `sample` makes 2 LLM calls per sampled row (draft + judge). `score`
makes no API calls at all -- pure human-entered data vs. existing scores.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drafter import draft_reply
from src.judge import judge_reply
from src.retrieval import RetrievalIndex

CHECKPOINT_PATH = Path("eval/.checkpoints/real_agent_full.jsonl")
GOLDEN_SET_PATH = Path("eval/golden_set.csv")
SAMPLE_CSV_PATH = Path("eval/judge_agreement_sample.csv")
RESULTS_PATH = Path("eval/judge_agreement_results.json")
RANDOM_SEED = 42

DIMENSIONS = ["grounded", "correct", "actionable"]


def _load_real_agent_checkpoint() -> pd.DataFrame:
    """Load real_agent_full.jsonl rows joined back to their customer_msg.

    real_agent_full.jsonl stores {index, intent, escalate_decision,
    judge_score} per row -- index matches the row position in the full
    (unsampled) golden_set.csv, since --full doesn't shuffle/sample. Only
    row_index/intent/customer_msg are used here -- the checkpoint's own
    judge_score is not reused (see cmd_sample: draft_reply/judge_reply are
    re-run together so the reply and its score always describe the same text).
    """
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found. Run `python eval/run_harness.py --full` "
            "(it can be partially complete -- this only needs finished rows) before sampling."
        )

    records = []
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    golden = pd.read_csv(GOLDEN_SET_PATH)

    rows = []
    for rec in records:
        idx = rec["index"]
        rows.append({
            "row_index": idx,
            "intent": rec["intent"],
            "customer_msg": golden.iloc[idx]["customer_msg"],
        })
    return pd.DataFrame(rows)


def cmd_sample(n: int) -> None:
    df = _load_real_agent_checkpoint()

    if len(df) < n:
        print(
            f"WARNING: only {len(df)} rows available in the checkpoint so far "
            f"(requested {n}). Sampling all {len(df)} available -- re-run this command later "
            "once eval/run_harness.py --full has more rows checkpointed, if you want the full amount."
        )
        n = len(df)

    sample = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Drafting + judging {len(sample)} reply(ies) (2 LLM calls each)...")
    retrieval_index = RetrievalIndex()
    replies, judge_grounded, judge_correct, judge_actionable = [], [], [], []
    for _, row in tqdm(list(sample.iterrows()), total=len(sample), desc="draft + judge"):
        drafted = draft_reply(
            row["customer_msg"], row["intent"], retrieval_index, exclude_exact_match=True
        )
        reply = drafted["reply"]
        score = judge_reply(row["customer_msg"], reply)
        replies.append(reply)
        judge_grounded.append(score.get("grounded"))
        judge_correct.append(score.get("correct"))
        judge_actionable.append(score.get("actionable"))

    sample["agent_reply"] = replies
    sample["judge_grounded"] = judge_grounded
    sample["judge_correct"] = judge_correct
    sample["judge_actionable"] = judge_actionable

    # Drop rows where the judge's response couldn't be parsed -- can't compare
    # agreement against a missing score.
    before = len(sample)
    sample = sample.dropna(subset=["judge_grounded", "judge_correct", "judge_actionable"])
    if len(sample) < before:
        print(f"Dropped {before - len(sample)} row(s) where the judge's response was unparseable.")

    for d in DIMENSIONS:
        sample[f"human_{d}"] = ""

    # Reorder so agent_reply sits next to customer_msg, readable for hand-scoring.
    cols = ["row_index", "customer_msg", "agent_reply"] + [
        c for c in sample.columns if c not in ("row_index", "customer_msg", "agent_reply", "intent")
    ]
    sample = sample[cols]

    sample.to_csv(SAMPLE_CSV_PATH, index=False)
    print(f"Wrote {len(sample)} rows -> {SAMPLE_CSV_PATH}")
    print(
        "Next: hand-score human_grounded / human_correct / human_actionable (1-5 each) "
        "for every row, reading customer_msg + agent_reply, using the same rubric as "
        "src/judge.py's SYSTEM_PROMPT, then run `python eval/judge_agreement.py score`."
    )


def _weighted_agreement_within_one(human: list[int], judge: list[int]) -> float:
    matches = sum(abs(h - j) <= 1 for h, j in zip(human, judge))
    return matches / len(human)


def cmd_score() -> None:
    if not SAMPLE_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLE_CSV_PATH} not found. Run `python eval/judge_agreement.py sample` first."
        )

    df = pd.read_csv(SAMPLE_CSV_PATH)

    human_cols = [f"human_{d}" for d in DIMENSIONS]
    missing = df[human_cols].isna().any(axis=1) | (df[human_cols] == "").any(axis=1)
    if missing.any():
        raise ValueError(
            f"{missing.sum()} row(s) in {SAMPLE_CSV_PATH} still have an empty human_* score. "
            "Fill in all rows (1-5, every dimension) before scoring."
        )

    for col in human_cols:
        df[col] = df[col].astype(int)

    results = {"n": len(df), "dimensions": {}}
    for d in DIMENSIONS:
        human = df[f"human_{d}"].tolist()
        judge = df[f"judge_{d}"].astype(int).tolist()

        exact_match = sum(h == j for h, j in zip(human, judge)) / len(human)
        within_one = _weighted_agreement_within_one(human, judge)
        kappa = cohen_kappa_score(human, judge, weights="linear")

        results["dimensions"][d] = {
            "exact_match_pct": exact_match,
            "within_one_point_pct": within_one,
            "linear_weighted_kappa": kappa,
        }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote results -> {RESULTS_PATH}")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample", help="Generate the hand-scoring CSV.")
    sample_parser.add_argument("--n", type=int, default=50, help="Number of rows to sample (default 50).")

    subparsers.add_parser("score", help="Compute agreement from the filled-in CSV.")

    args = parser.parse_args()
    if args.command == "sample":
        cmd_sample(args.n)
    elif args.command == "score":
        cmd_score()


if __name__ == "__main__":
    main()
