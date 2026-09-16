"""Evaluation harness: runs trivial baseline, simple baseline, and the real
agent (classifier + drafter + escalation policy) against a sample of
golden_set and reports automated metrics for all three.

Default run is a small subsample (25) to fit a reasonable reproduction
time -- the real agent's classifier and drafter both make throttled Gemini
API calls, so scoring all 175 golden-set rows takes a while (2 agent calls +
up to 2 judge calls per row). See src/llm.py's rate-limit finding for the
two separate free-tier caps discovered on this key (per-minute and, more
seriously, per-day) and why DEFAULT_MODEL is gemini-3.1-flash-lite.

Metrics reported:
  - Intent classification accuracy (trivial / simple / real agent)
  - Escalation decision accuracy, precision, recall (trivial / simple / real)
  - Reply quality: LLM-judge scores (grounded/correct/actionable, 1-5) for
    real agent and simple baseline replies (trivial baseline's reply is
    fixed, so it is judged once, not once per row)

Eval leakage fix: every golden_set.csv row exists verbatim in the retrieval
index (it was sampled from the same apple_triples.csv the index is built
from), so the real agent's draft_reply/decide_escalation calls below pass
exclude_exact_match=True -- without it, an early run showed the real agent's
escalation policy scoring 0 precision/0 recall, because the "no good
retrieval match" signal could never fire against a message that was always
its own top match (see src/retrieval.py's RetrievalIndex.query docstring
and src/README.md for the full discovery).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_simple import simple_classify, simple_draft, simple_escalate
from src.baseline_trivial import trivial_classify, trivial_draft, trivial_escalate
from src.classifier import classify_intent
from src.drafter import draft_reply
from src.escalation import decide_escalation
from src.judge import judge_reply
from src.retrieval import RetrievalIndex

GOLDEN_SET_PATH = Path("eval/golden_set.csv")
DEFAULT_N = 25
RANDOM_SEED = 42


def _intent_metrics(predictions: list[str], labels: list[str]) -> dict:
    correct = sum(p == l for p, l in zip(predictions, labels))
    return {"accuracy": correct / len(labels), "n": len(labels)}


def _escalation_metrics(predictions: list[str], labels: list[str]) -> dict:
    tp = sum(p == "escalate" and l == "escalate" for p, l in zip(predictions, labels))
    fp = sum(p == "escalate" and l == "auto" for p, l in zip(predictions, labels))
    tn = sum(p == "auto" and l == "auto" for p, l in zip(predictions, labels))
    fn = sum(p == "auto" and l == "escalate" for p, l in zip(predictions, labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
             "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _mean_score(scores: list[dict], dimension: str) -> float | None:
    values = [s[dimension] for s in scores if s.get(dimension) is not None]
    return sum(values) / len(values) if values else None


def run_trivial(df: pd.DataFrame) -> dict:
    intent_preds = [trivial_classify(m)["intent"] for m in df["customer_msg"]]
    escalate_preds = [trivial_escalate(m)["decision"] for m in df["customer_msg"]]

    # trivial's reply is fixed: judge it once, not once per row
    judge_score = judge_reply(df["customer_msg"].iloc[0], trivial_draft(df["customer_msg"].iloc[0])["reply"])

    return {
        "intent": _intent_metrics(intent_preds, df["intent_label"].tolist()),
        "escalation": _escalation_metrics(escalate_preds, df["auto_or_escalate"].tolist()),
        "reply_quality": {
            "grounded": judge_score.get("grounded"),
            "correct": judge_score.get("correct"),
            "actionable": judge_score.get("actionable"),
            "note": "fixed reply, judged once for all rows",
        },
    }


def run_simple(df: pd.DataFrame) -> dict:
    intent_preds = []
    escalate_preds = []
    judge_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="simple baseline"):
        intent = simple_classify(row["customer_msg"])["intent"]
        intent_preds.append(intent)
        escalate_preds.append(simple_escalate(intent)["decision"])
        reply = simple_draft(row["customer_msg"], intent)["reply"]
        judge_scores.append(judge_reply(row["customer_msg"], reply))

    return {
        "intent": _intent_metrics(intent_preds, df["intent_label"].tolist()),
        "escalation": _escalation_metrics(escalate_preds, df["auto_or_escalate"].tolist()),
        "reply_quality": {
            "grounded": _mean_score(judge_scores, "grounded"),
            "correct": _mean_score(judge_scores, "correct"),
            "actionable": _mean_score(judge_scores, "actionable"),
        },
    }


def run_real_agent(df: pd.DataFrame, retrieval_index: RetrievalIndex) -> dict:
    intent_preds = []
    escalate_preds = []
    judge_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="real agent"):
        classification = classify_intent(row["customer_msg"])
        intent_preds.append(classification["intent"])

        escalation = decide_escalation(
            row["customer_msg"], classification["intent"], retrieval_index,
            exclude_exact_match=True,
        )
        escalate_preds.append(escalation["decision"])

        drafted = draft_reply(
            row["customer_msg"], classification["intent"], retrieval_index, exclude_exact_match=True
        )
        judge_scores.append(judge_reply(row["customer_msg"], drafted["reply"]))

    return {
        "intent": _intent_metrics(intent_preds, df["intent_label"].tolist()),
        "escalation": _escalation_metrics(escalate_preds, df["auto_or_escalate"].tolist()),
        "reply_quality": {
            "grounded": _mean_score(judge_scores, "grounded"),
            "correct": _mean_score(judge_scores, "correct"),
            "actionable": _mean_score(judge_scores, "actionable"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                         help=f"Number of golden-set rows to sample (default {DEFAULT_N}). Ignored if --full.")
    parser.add_argument("--full", action="store_true", help="Run all 175 golden-set rows (~2.5 hours).")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path.")
    args = parser.parse_args()

    df = pd.read_csv(GOLDEN_SET_PATH)
    if not args.full:
        n = min(args.n, len(df))
        df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Evaluating against {len(df)} golden-set rows ({'full set' if args.full else f'subsample of {len(df)}'})")

    print("\n== Trivial baseline ==")
    trivial_results = run_trivial(df)

    print("\n== Simple baseline ==")
    simple_results = run_simple(df)

    print("\n== Real agent ==")
    retrieval_index = RetrievalIndex()
    real_agent_results = run_real_agent(df, retrieval_index)

    results = {
        "n": len(df),
        "full_run": args.full,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trivial": trivial_results,
        "simple": simple_results,
        "real_agent": real_agent_results,
        "caveats": [
            "Every golden_set.csv row exists verbatim in the retrieval index the real agent "
            "queries (it was sampled from the same source data). This harness passes "
            "exclude_exact_match=True to draft_reply/decide_escalation to prevent a row from "
            "matching itself at similarity 1.00 -- without that fix, an early run showed the "
            "real agent's escalation policy scoring 0 precision/0 recall. Retrieval-based "
            "signals are still measured against the rest of the 5,000-row apple_triples.csv "
            "pool (which may itself contain near-duplicate/paraphrased versions of a golden-set "
            "message from the same era of complaints), so results may still be somewhat "
            "optimistic versus a truly unseen message -- see report/ for discussion.",
        ],
    }

    out_path = args.out or ("eval/results_full.json" if args.full else "eval/results_preview.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote results -> {out_path}")
    print(json.dumps({k: v for k, v in results.items() if k not in ("caveats",)}, indent=2))


if __name__ == "__main__":
    main()
