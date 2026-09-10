"""Reconstruct AppleSupport threads from the raw twcs.csv dataset.

Extracts two datasets from data/raw/twcs.csv, scoped to the AppleSupport brand:

1. Triples: customer message -> AppleSupport reply -> customer follow-up.
   A full 3-turn arc, used as the primary grounding/golden-set source since it
   shows an actual resolution attempt landing with the customer.

2. No-followup pairs: customer message -> AppleSupport reply where the customer
   never responded again. Not fed into the agent's grounding/retrieval index;
   kept aside as a reference dataset for the report's failure-analysis /
   "what reply patterns correlate with customer abandonment" discussion.

Output:
    data/processed/apple_triples.csv     (subsampled to TRIPLE_SAMPLE_SIZE)
    data/processed/apple_no_followup.csv (full count, not subsampled)
"""
import re
from pathlib import Path

import pandas as pd

RAW = "data/raw/twcs.csv"
BRAND = "AppleSupport"
TRIPLE_SAMPLE_SIZE = 5000
RANDOM_SEED = 42

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    # strip @handles, urls, and collapse whitespace from tweet text
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(RAW, dtype=str)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # index by tweet_id for O(1) lookups instead of scanning the full frame per row
    rows = df.set_index("tweet_id", drop=False)
    rows = rows[~rows.index.duplicated(keep="first")]

    # group inbound tweets by what they're responding to, so "does this brand
    # reply have a customer follow-up" is an O(1) dict lookup instead of an
    # O(n) scan of the full 2.8M-row frame per brand reply
    inbound = df[df["inbound"] == "True"]
    followups_by_parent = {
        parent_id: group for parent_id, group in inbound.groupby("in_response_to_tweet_id")
    }

    brand_replies = df[df["author_id"] == BRAND].copy()
    print(f"AppleSupport replies total: {len(brand_replies)}")

    triples = []
    no_followup = []

    for _, brand_row in brand_replies.iterrows():
        in_response_to = brand_row["in_response_to_tweet_id"]
        if pd.isna(in_response_to) or in_response_to not in rows.index:
            continue
        customer_msg_row = rows.loc[in_response_to]
        if customer_msg_row["inbound"] != "True":
            continue

        record = {
            "customer_tweet_id": customer_msg_row["tweet_id"],
            "customer_msg": clean_text(customer_msg_row["text"]),
            "brand_tweet_id": brand_row["tweet_id"],
            "brand_reply": clean_text(brand_row["text"]),
        }

        followups = followups_by_parent.get(brand_row["tweet_id"])
        if followups is not None and len(followups) > 0:
            followup_row = followups.iloc[0]
            record["customer_followup"] = clean_text(followup_row["text"])
            triples.append(record)
        else:
            no_followup.append(record)

    triples_df = pd.DataFrame(triples).drop_duplicates(subset=["customer_msg", "brand_reply"])
    no_followup_df = pd.DataFrame(no_followup).drop_duplicates(subset=["customer_msg", "brand_reply"])

    print(f"Triples (customer->brand->customer): {len(triples_df)}")
    print(f"No-followup pairs (customer->brand only): {len(no_followup_df)}")

    if len(triples_df) > TRIPLE_SAMPLE_SIZE:
        triples_df = triples_df.sample(n=TRIPLE_SAMPLE_SIZE, random_state=RANDOM_SEED)

    triples_df.to_csv(OUT_DIR / "apple_triples.csv", index=False)
    no_followup_df.to_csv(OUT_DIR / "apple_no_followup.csv", index=False)
    print(f"Wrote {len(triples_df)} triples -> {OUT_DIR / 'apple_triples.csv'}")
    print(f"Wrote {len(no_followup_df)} no-followup pairs -> {OUT_DIR / 'apple_no_followup.csv'}")


if __name__ == "__main__":
    main()
