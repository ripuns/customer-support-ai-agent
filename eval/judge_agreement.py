"""Judge-vs-human agreement check.

Required by the assignment: measure how well the LLM-as-judge's reply-quality
scores (grounded/correct/actionable, 1-5) agree with a human rater's scores
on the same replies, so the judge's scores aren't trusted blindly.

Two-step workflow:
  1. `python eval/judge_agreement.py sample --n 50` -- reads the real agent's
     already-judged replies from eval/.checkpoints/real_agent_full.jsonl (no
     new LLM calls) and writes a CSV with the customer message, the agent's
     reply, and the judge's scores, plus 3 empty columns for a human rater to
     fill in (human_grounded, human_correct, human_actionable).
  2. Hand-score the sampled rows in eval/judge_agreement_sample.csv (1-5 per
     dimension, same rubric as src/judge.py's SYSTEM_PROMPT).
  3. `python eval/judge_agreement.py score` -- reads the filled-in CSV and
     computes agreement between the judge and human scores: exact-match %,
     within-1-point %, and linear-weighted Cohen's kappa (the standard
     chance-corrected metric for ordinal 1-5 scales, where a 4-vs-5
     disagreement counts as a smaller error than a 1-vs-5 disagreement).
     Writes results to eval/judge_agreement_results.json.

No new API calls at either step -- sampling reuses judge scores already
computed during the eval/run_harness.py --full run; scoring is pure
human-entered data vs. those existing scores.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    (unsampled) golden_set.csv, since --full doesn't shuffle/sample.
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
        judge_score = rec["judge_score"]
        rows.append({
            "row_index": idx,
            "customer_msg": golden.iloc[idx]["customer_msg"],
            "judge_grounded": judge_score.get("grounded"),
            "judge_correct": judge_score.get("correct"),
            "judge_actionable": judge_score.get("actionable"),
        })
    return pd.DataFrame(rows)


def cmd_sample(n: int) -> None:
    df = _load_real_agent_checkpoint()
    # Only sample rows where the judge actually returned parseable scores on
    # all 3 dimensions -- a row with a None score can't be compared for agreement.
    df = df.dropna(subset=[f"judge_{d}" for d in DIMENSIONS])

    if len(df) < n:
        print(
            f"WARNING: only {len(df)} fully-judged rows available in the checkpoint so far "
            f"(requested {n}). Sampling all {len(df)} available -- re-run this command later "
            "once eval/run_harness.py --full has judged more rows, if you want the full 50."
        )
        n = len(df)

    sample = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    for d in DIMENSIONS:
        sample[f"human_{d}"] = ""

    sample.to_csv(SAMPLE_CSV_PATH, index=False)
    print(f"Wrote {len(sample)} rows -> {SAMPLE_CSV_PATH}")
    print(
        "Next: hand-score human_grounded / human_correct / human_actionable (1-5 each) "
        "for every row using the same rubric as src/judge.py's SYSTEM_PROMPT, then run "
        "`python eval/judge_agreement.py score`."
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
