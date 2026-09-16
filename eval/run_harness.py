"""Evaluation harness: runs trivial baseline, simple baseline, and the real
agent (classifier + drafter + escalation policy) against a sample of
golden_set and reports automated metrics for all three.

Default run is a small subsample (25) to fit a reasonable reproduction
time -- the real agent's classifier, escalation check, and drafter all make
LLM API calls (currently AWS Bedrock, Gemma 3 27B -- see src/llm.py), so
scoring all 175 golden-set rows still takes a while even without the
throttling the original Gemini free-tier key required (see src/llm.py's
provider-switch history for that incident).

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

Checkpointing: each tier writes one JSON line per completed row to
eval/.checkpoints/<tier>_<n or full>.jsonl as it goes (flushed immediately,
not buffered), keyed by row index within this run's sample. On startup each
tier skips indices already present in its checkpoint file, so a crash,
API failure, or manual interrupt partway through a `--full` run (700+ LLM
calls, real cost/time on a live API key) can be resumed by simply re-running
the same command -- already-completed rows are not re-sent to the API.
Metrics are computed once by reading the completed checkpoint file at the
end, so a resumed run's final numbers are identical to an uninterrupted run.
Pass --fresh to ignore any existing checkpoint and start over.
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
CHECKPOINT_DIR = Path("eval/.checkpoints")
DEFAULT_N = 25
RANDOM_SEED = 42


def _checkpoint_path(tier: str, run_tag: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{tier}_{run_tag}.jsonl"


def _load_checkpoint(path: Path) -> dict[int, dict]:
    """Load completed rows already checkpointed, keyed by row index."""
    if not path.exists():
        return {}
    completed = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed[row["index"]] = row
    return completed


def _append_checkpoint(path: Path, index: int, record: dict) -> None:
    record = {"index": index, **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


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


def run_trivial(df: pd.DataFrame, run_tag: str, fresh: bool = False) -> dict:
    # Trivial's classify/escalate are free (no LLM call) -- only its one fixed
    # reply's judge call is worth checkpointing, and only 1 call total either way.
    checkpoint_path = _checkpoint_path("trivial", run_tag)
    completed = {} if fresh else _load_checkpoint(checkpoint_path)

    if 0 in completed:
        judge_score = completed[0]
    else:
        judge_score = judge_reply(
            df["customer_msg"].iloc[0], trivial_draft(df["customer_msg"].iloc[0])["reply"]
        )
        _append_checkpoint(checkpoint_path, 0, judge_score)

    intent_preds = [trivial_classify(m)["intent"] for m in df["customer_msg"]]
    escalate_preds = [trivial_escalate(m)["decision"] for m in df["customer_msg"]]

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


def run_simple(df: pd.DataFrame, run_tag: str, fresh: bool = False) -> dict:
    checkpoint_path = _checkpoint_path("simple", run_tag)
    completed = {} if fresh else _load_checkpoint(checkpoint_path)
    if completed:
        print(f"  resuming simple baseline: {len(completed)}/{len(df)} rows already checkpointed")

    for i, row in tqdm(list(df.iterrows()), total=len(df), desc="simple baseline"):
        if i in completed:
            continue
        intent = simple_classify(row["customer_msg"])["intent"]
        escalate_decision = simple_escalate(intent)["decision"]
        reply = simple_draft(row["customer_msg"], intent)["reply"]
        judge_score = judge_reply(row["customer_msg"], reply)
        record = {"intent": intent, "escalate_decision": escalate_decision, "judge_score": judge_score}
        _append_checkpoint(checkpoint_path, i, record)
        completed[i] = record

    ordered = [completed[i] for i in range(len(df))]
    intent_preds = [r["intent"] for r in ordered]
    escalate_preds = [r["escalate_decision"] for r in ordered]
    judge_scores = [r["judge_score"] for r in ordered]

    return {
        "intent": _intent_metrics(intent_preds, df["intent_label"].tolist()),
        "escalation": _escalation_metrics(escalate_preds, df["auto_or_escalate"].tolist()),
        "reply_quality": {
            "grounded": _mean_score(judge_scores, "grounded"),
            "correct": _mean_score(judge_scores, "correct"),
            "actionable": _mean_score(judge_scores, "actionable"),
        },
    }


def run_real_agent(df: pd.DataFrame, retrieval_index: RetrievalIndex, run_tag: str, fresh: bool = False) -> dict:
    checkpoint_path = _checkpoint_path("real_agent", run_tag)
    completed = {} if fresh else _load_checkpoint(checkpoint_path)
    if completed:
        print(f"  resuming real agent: {len(completed)}/{len(df)} rows already checkpointed")

    for i, row in tqdm(list(df.iterrows()), total=len(df), desc="real agent"):
        if i in completed:
            continue
        classification = classify_intent(row["customer_msg"])
        escalation = decide_escalation(
            row["customer_msg"], classification["intent"], retrieval_index,
            exclude_exact_match=True,
        )
        drafted = draft_reply(
            row["customer_msg"], classification["intent"], retrieval_index, exclude_exact_match=True
        )
        judge_score = judge_reply(row["customer_msg"], drafted["reply"])
        record = {
            "intent": classification["intent"],
            "escalate_decision": escalation["decision"],
            "judge_score": judge_score,
        }
        _append_checkpoint(checkpoint_path, i, record)
        completed[i] = record

    ordered = [completed[i] for i in range(len(df))]
    intent_preds = [r["intent"] for r in ordered]
    escalate_preds = [r["escalate_decision"] for r in ordered]
    judge_scores = [r["judge_score"] for r in ordered]

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
    parser.add_argument("--full", action="store_true", help="Run all 175 golden-set rows.")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path.")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing checkpoint for this run and start over.")
    args = parser.parse_args()

    df = pd.read_csv(GOLDEN_SET_PATH)
    if not args.full:
        n = min(args.n, len(df))
        df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    # run_tag identifies this specific run's checkpoint files -- a --full run and a
    # --n 25 run never share or collide with each other's saved progress.
    run_tag = "full" if args.full else f"n{len(df)}"

    print(f"Evaluating against {len(df)} golden-set rows ({'full set' if args.full else f'subsample of {len(df)}'})")
    print(f"Checkpoints: {CHECKPOINT_DIR}/*_{run_tag}.jsonl "
          f"(re-running the same command resumes from here; pass --fresh to restart)")

    print("\n== Trivial baseline ==")
    trivial_results = run_trivial(df, run_tag, fresh=args.fresh)

    print("\n== Simple baseline ==")
    simple_results = run_simple(df, run_tag, fresh=args.fresh)

    print("\n== Real agent ==")
    retrieval_index = RetrievalIndex()
    real_agent_results = run_real_agent(df, retrieval_index, run_tag, fresh=args.fresh)

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
