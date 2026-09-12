"""Retrieval index over historical resolved AppleSupport threads.

Given an incoming customer message, finds the most similar historical
customer_msg values from data/processed/apple_triples.csv and returns
their (customer_msg, brand_reply, customer_followup) triples -- these are
what the drafter grounds its reply generation in.

Uses TF-IDF + cosine similarity (scikit-learn) rather than an embeddings
API: free, instant, no rate limits, and avoids repeating the Gemini
free-tier throttling problem at 5,000x the scale (see src/llm.py's rate-limit
finding). Trades off some semantic matching quality (won't catch
paraphrases with no word overlap) for full local reproducibility.
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TRIPLES_PATH = Path("data/processed/apple_triples.csv")


class RetrievalIndex:
    """TF-IDF index over apple_triples.csv's customer_msg column."""

    def __init__(self, triples_path: Path = TRIPLES_PATH):
        df = pd.read_csv(triples_path)
        df = df.dropna(subset=["customer_msg"]).reset_index(drop=True)
        self.df = df

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix = self.vectorizer.fit_transform(df["customer_msg"])

    def query(self, customer_msg: str, k: int = 3) -> list[dict]:
        """Return the top-k most similar historical triples to customer_msg.

        Returns fewer than k results only if the index itself has fewer than k rows.
        """
        query_vec = self.vectorizer.transform([customer_msg])
        similarities = cosine_similarity(query_vec, self.matrix)[0]

        top_k_idx = similarities.argsort()[::-1][:k] # returns top k similarities

        results = []
        for idx in top_k_idx:
            row = self.df.iloc[idx]
            results.append({
                "customer_msg": row["customer_msg"],
                "brand_reply": row["brand_reply"],
                "customer_followup": row.get("customer_followup"),
                "similarity": float(similarities[idx]),
            })
        return results
