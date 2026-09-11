import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intents import INTENT_LABELS  # noqa: E402

IN_PATH = Path("data/processed/apple_triples_classified.csv")
OUT_PATH = Path("eval/golden_set.csv")
PER_INTENT_SAMPLE_SIZE = 25
RANDOM_SEED = 42


def main():
    df = pd.read_csv(IN_PATH)

    samples = []
    for intent in INTENT_LABELS:
        bucket = df[df["predicted_intent"] == intent]
        n = min(PER_INTENT_SAMPLE_SIZE, len(bucket))
        if n < PER_INTENT_SAMPLE_SIZE:
            print(f"WARNING: {intent} only has {len(bucket)} candidates, sampling all of them")
        samples.append(bucket.sample(n=n, random_state=RANDOM_SEED))

    golden = pd.concat(samples, ignore_index=True)
    golden = golden.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle

    golden = golden.rename(columns={"predicted_intent": "suggested_intent"})
    golden["intent_label"] = ""
    golden["auto_or_escalate"] = ""
    golden["escalate_reason"] = ""
    golden["reply_quality_note"] = ""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(OUT_PATH, index=False)

    print(f"\nWrote {len(golden)} rows -> {OUT_PATH}")
    print("\nPer-intent counts in the sample (by keyword-heuristic suggestion):")
    print(golden["suggested_intent"].value_counts())


if __name__ == "__main__":
    main()
