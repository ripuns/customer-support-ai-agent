"""Retrieval index over historical resolved AppleSupport threads.

Given an incoming customer message, finds the most similar historical
customer_msg values from apple_triples.csv and returns their 
(customer_msg, brand_reply, customer_followup) triples -- these are
what the drafter grounds its reply generation in.

Uses TF-IDF + cosine similarity (scikit-learn) rather than an embeddings API: 
free, instant, no rate limits, and avoids repeating the gemini free-tier throttling problem 
(see src/llm.py's rate-limit finding).Trades off some semantic matching quality 
(won't catch paraphrases with no word overlap) for full local reproducibility.
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

    def query(self, customer_msg: str, k: int = 3, exclude_exact_match: bool = False) -> list[dict]:
        """Return the top-k most similar historical triples to customer_msg.

        Returns fewer than k results only if the index itself has fewer than k
        (matching, if exclude_exact_match) rows.

        exclude_exact_match: when True, drops any indexed row whose customer_msg
        is character-for-character identical to the query before taking the
        top-k. Needed during evaluation: eval/golden_set.csv was sampled from
        the same apple_triples.csv this index is built from, so without this,
        querying with a golden-set message always finds itself at similarity
        1.00 -- making grounding and any retrieval-based signal look
        artificially perfect. See src/README.md's retrieval.py entry for the
        discovery (this made the real agent's escalation policy score 0
        precision/recall on a harness run, since the "no good retrieval match"
        signal could never fire against a message that was always its own
        top match).
        """
        query_vec = self.vectorizer.transform([customer_msg])
        similarities = cosine_similarity(query_vec, self.matrix)[0]

        if exclude_exact_match:
            self_mask = (self.df["customer_msg"] == customer_msg).to_numpy()
            similarities = similarities.copy()
            similarities[self_mask] = -1.0

        top_k_idx = similarities.argsort()[::-1][:k] # returns top k similarities
        top_k_idx = [idx for idx in top_k_idx if similarities[idx] > -1.0]

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
