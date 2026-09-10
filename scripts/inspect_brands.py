"""Inspect the twcs dataset to help pick a target brand.

Prints: top brand accounts by tweet volume, how many of their tweets are
inbound-response threads (customer -> brand -> customer), and average
reply length, to gauge which brand has enough clean, resolvable threads
to ground an agent on.
"""
import pandas as pd

RAW = "data/raw/twcs.csv"


def main():
    df = pd.read_csv(RAW, dtype=str)
    print(f"Total rows: {len(df)}")
    print(df.columns.tolist())

    brands = df[df["inbound"] == "False"]
    # return unique counts of authors with a limit of 25
    counts = brands["author_id"].value_counts().head(25)
    print("\nTop 25 brand accounts by tweet volume:")
    print(counts)

    # For a shortlist, estimate thread depth: customer msg -> brand reply -> customer follow-up
    df["response_tweet_id"] = df["response_tweet_id"]
    id_to_row = df.set_index("tweet_id")

    for brand in counts.head(8).index:
        brand_replies = df[(df["author_id"] == brand)]
        n = len(brand_replies)
        # how many brand replies have a customer follow-up (i.e. someone responded to them)
        brand_ids = set(brand_replies["tweet_id"])
        follow_ups = df[df["in_response_to_tweet_id"].isin(brand_ids)]
        n_followup = len(follow_ups)
        avg_len = brand_replies["text"].dropna().str.len().mean()
        print(f"{brand}: replies={n}, with_customer_followup={n_followup}, avg_reply_chars={avg_len:.0f}")


if __name__ == "__main__":
    main()
