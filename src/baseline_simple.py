"""Simple baseline: a naive-but-real approach, the second required comparison point.

Uses classify_keyword (real keyword matching, not a fixed guess) for
intent, a template reply per intent (not one fixed generic reply), and an
intent-category escalation rule (escalate account_security/billing_purchase,
auto otherwise) -- the kind of first-pass approach a developer might ship
without doing careful per-thread analysis. This is meant to sit strictly
between src/baseline_trivial.py (the floor) and the real agent
(src/classifier.py + src/drafter.py + src/escalation.py).
"""
from src.keyword_classifier import classify_keyword
from src.intents import INTENT_LABELS

TEMPLATE_REPLIES = {
    "software_bug": "Thanks for reporting this. Could you let us know your device model and iOS version so we can look into this bug?",
    "battery_performance": "We'd like to help with your battery concern. Which device and iOS version are you using?",
    "account_security": "We take account security seriously. Please DM us so we can help you securely.",
    "billing_purchase": "We can help with this billing question. Please DM us your order or purchase details.",
    "hardware_issue": "Sorry to hear about this hardware issue. Which device model are you using, and when did this start?",
    "how_to_info": "Great question! Here's an article that should help: [link]. Let us know if you need more assistance.",
    "store_order_service": "We'd like to help with your order/store question. Could you DM us more details?",
    "unknown": "Thanks for reaching out. Could you DM us more details so we can look into this together?",
}

# The rule this project's own escalation.py tried first and rejected after
# testing against the golden set (see src/README.md's escalation.py design
# history: 69% accuracy, and here it also produces poor precision/recall --
# 50%/36% against the current golden set -- despite matching the trivial
# baseline's 60% accuracy). Used here deliberately as the "simple" baseline
# specifically because it's a realistic naive first attempt, and comparing
# it against the trivial baseline and the real policy is informative: equal
# accuracy to doing nothing, but for the wrong reasons.
HIGH_RISK_INTENTS = {"account_security", "billing_purchase"}


def simple_classify(customer_msg: str) -> dict:
    intent = classify_keyword(customer_msg)
    return {"intent": intent, "confidence": None}


def simple_draft(customer_msg: str, intent: str) -> dict:
    reply = TEMPLATE_REPLIES.get(intent, TEMPLATE_REPLIES["unknown"])
    return {"reply": reply, "grounded_on": []}


def simple_escalate(intent: str) -> dict:
    decision = "escalate" if intent in HIGH_RISK_INTENTS else "auto"
    reason = (
        f"intent '{intent}' is in the fixed high-risk category list"
        if decision == "escalate"
        else "intent not in fixed high-risk category list"
    )
    return {"decision": decision, "reasons": [reason]}


assert all(label in INTENT_LABELS or label == "unknown" for label in TEMPLATE_REPLIES), (
    "TEMPLATE_REPLIES key not found in src.intents.INTENT_LABELS"
)
