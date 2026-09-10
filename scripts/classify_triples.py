"""Pre-classify all AppleSupport triples using the keyword heuristic.

Runs every customer_msg in data/processed/apple_triples.csv through
keyword_classifier (instant, no API calls) and writes the predicted 
intent back out. This classified pool is what the golden-set
sampler stratifies over.

This is a heuristic pre-pass, not a ground-truth labeling: every golden-set
example still gets its intent confirmed or corrected by hand during
labeling.

Output:
    data/processed/apple_triples_classified.csv
    (all columns from apple_triples.csv, plus a `predicted_intent` column)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.keyword_classifier import classify_keyword  # noqa: E402

IN_PATH = Path("data/processed/apple_triples.csv")
OUT_PATH = Path("data/processed/apple_triples_classified.csv")


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Classifying {len(df)} customer messages with keyword heuristic...")

    df["predicted_intent"] = df["customer_msg"].apply(classify_keyword)
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} classified rows -> {OUT_PATH}")
    print("\nIntent distribution:")
    print(df["predicted_intent"].value_counts())


if __name__ == "__main__":
    main()
