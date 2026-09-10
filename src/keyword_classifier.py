"""Keyword-based intent pre-classifier.

A cheap, free, instant heuristic classifier used only to build a stratified
sampling pool for the golden evaluation set -- NOT a claim of ground-truth
accuracy. Every golden-set example gets its intent confirmed/corrected by
hand during labeling (see eval/), so this only needs to be good enough to
separate the 7 intents into roughly-populated buckets rather than perfectly
correct.

Replaces an earlier LLM-based approach (scripts/classify_triples.py, since
removed) that turned out to be infeasible: the Gemini free-tier key is
capped at 5 requests/minute, and classifying enough of the ~5,000
apple_triples.csv rows for reliable stratification would have taken hours
and burned most of the quota on retries (see src/llm.py's rate-limit note
and scripts/README.md for the full investigation).

Keywords were chosen by checking real match counts against
data/processed/apple_triples.csv (all 7 buckets found 100+ matches) rather
than guessed blind.
"""
import re

from src.intents import INTENT_LABELS

# Checked in this order; first match wins. Order matters for overlapping
# terms (e.g. "screen" alone could be a software glitch or a cracked
# hardware screen -- hardware-specific phrasing is checked before the
# generic software_bug fallback catches it).
KEYWORD_RULES = [
    ("account_security", [
        "apple id", "password", "verification code", "2fa", "two-factor",
        "two factor", "locked", "login", "log in", "sign in", "signed in",
        "hacked", "phishing", "icloud account", "reset my account",
        "compromised",
    ]),
    ("billing_purchase", [
        "charged", "charge me", "refund", "subscription", "unauthorized",
        "billing", "invoice", "unsubscribe", "cancel my", "price", "pricing",
        "plan cost", "storage plan", "purchase",
    ]),
    ("battery_performance", [
        "battery", "drain", "draining", "% by", "charge level", "slow",
        "lagg", "sluggish", "degraded",
    ]),
    ("hardware_issue", [
        "screen crack", "cracked screen", "touch screen", "touchscreen",
        "won't charge", "wont charge", "charger", "charging cable",
        "speaker", "camera lens", "button", "physical damage", "screen is",
    ]),
    ("store_order_service", [
        "apple store", "applecare", "apple care", "appointment", "schedule",
        "delivery", "order", "shipped", "shipping", "warranty", "genius bar",
    ]),
    ("how_to_info", [
        "how do i", "how to", "is there a way", "can i ", "is it possible",
        "any way to", "best way to",
    ]),
]

DEFAULT_INTENT = "software_bug"


def classify_keyword(msg: str) -> str:
    """Return a heuristic intent label for a customer message.

    Case-insensitive substring match against KEYWORD_RULES, first match
    wins. Falls back to DEFAULT_INTENT (software_bug) since that is the
    dominant real category in this dataset and most unmatched messages are
    in fact bug/crash/glitch complaints (see src/intents.py taxonomy notes).
    """
    if not isinstance(msg, str) or not msg.strip():
        return "unknown"
    text = msg.lower()
    for label, keywords in KEYWORD_RULES:
        if any(kw in text for kw in keywords):
            return label
    return DEFAULT_INTENT


assert all(label in INTENT_LABELS for label, _ in KEYWORD_RULES), (
    "KEYWORD_RULES label not found in src.intents.INTENT_LABELS"
)
