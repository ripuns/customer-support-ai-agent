"""LLM-based intent classifier for incoming AppleSupport customer messages.

Classifies a single customer_msg into one of the 7 intents defined in
src/intents.py, using a few-shot prompt sent through call_llm.
Returns both the predicted label and a confidence self-estimate, since the
escalation policy needs confidence as one of its inputs.
"""
import re

from src.intents import INTENTS, INTENT_LABELS
from src.llm import call_llm

SYSTEM_PROMPT = (
    "You are an intent classifier for AppleSupport customer messages on Twitter. "
    "Classify the customer's message into exactly one of the following intents:\n\n"
    + "\n".join(f"- {label}: {desc}" for label, desc in INTENTS.items())
    + "\n\nRespond in exactly this format, two lines:\n"
    "INTENT: <one of the labels above, exactly as written>\n"
    "CONFIDENCE: <a number from 0 to 100, your confidence that this is the correct intent>"
)

_INTENT_LINE_RE = re.compile(r"INTENT:\s*(\w+)", re.IGNORECASE)
_CONFIDENCE_LINE_RE = re.compile(r"CONFIDENCE:\s*(\d+)", re.IGNORECASE)

DEFAULT_CONFIDENCE_ON_PARSE_FAILURE = 0


def classify_intent(customer_msg: str) -> dict:
    """Classify a customer message. Returns {"intent": str, "confidence": int}.

    intent is "unknown" if the message is empty/invalid, or if the model's
    response could not be parsed into a recognized label. callers should
    treat "unknown" as a signal to escalate rather than guess.
    """
    if not isinstance(customer_msg, str) or not customer_msg.strip():
        return {"intent": "unknown", "confidence": DEFAULT_CONFIDENCE_ON_PARSE_FAILURE}

    response = call_llm(SYSTEM_PROMPT, customer_msg, temperature=0.0)

    intent_match = _INTENT_LINE_RE.search(response)
    confidence_match = _CONFIDENCE_LINE_RE.search(response)

    intent = intent_match.group(1).lower() if intent_match else None
    if intent not in INTENT_LABELS:
        intent = "unknown"

    if confidence_match:
        confidence = max(0, min(100, int(confidence_match.group(1))))
    else:
        confidence = DEFAULT_CONFIDENCE_ON_PARSE_FAILURE

    return {"intent": intent, "confidence": confidence}
