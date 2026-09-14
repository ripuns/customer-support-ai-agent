"""Rule-based escalation policy.

Decides whether an incoming customer message should be auto-handled or
escalated to a human, with a stated reason.

Escalate when:
  - classifier confidence is low or intent is "unknown" (can't trust the
    classification enough to draft a safe auto-reply)
  - the message contains red-flag content: signs a prior fix attempt already
    failed, explicit requests for a human/exception, account lockout/data-loss
    language, or a severity signal (brand-new device already failing,
    recurring/repeated failures, device completely dead)
  - retrieval found no sufficiently similar historical resolution to ground
    a reply in (a proxy for "this is not a well-understood, repeatable case")

Do NOT escalate purely on:
  - negative tone / profanity
  - topic frequency ("many people report this" is not a per-message signal)
  - the message's intent/topic category alone (e.g. all account_security or
    billing_purchase messages)
"""
from src.retrieval import RetrievalIndex

RED_FLAG_PHRASES = [
    "didn't work", "doesn't work", "still not working", "still doesn't",
    "did not help", "doesn't help", "none of the above", "already tried",
    "restart didn't", "restarting didn't", "tried both", "tried everything",
    "locked out", "can't access", "cannot access", "lost all my",
    "permanently", "unauthorized", "when will you fix",
    "when can i expect", "speak to a human", "speak to someone",
    "escalate", "supervisor", "every single time", "again and again",
    "second time", "brick", "completely dead", "won't turn on",
    "wont turn on", "haven't even had", "havent even had",
    "disappeared", "backup failed", "recover missing",
]

CONFIDENCE_ESCALATE_THRESHOLD = 60

MIN_SIMILARITY_FOR_GROUNDING = 0.15


def _find_red_flags(customer_msg: str) -> list[str]:
    text = customer_msg.lower()
    return [phrase for phrase in RED_FLAG_PHRASES if phrase in text]


def decide_escalation(
    customer_msg: str,
    intent: str,
    confidence: int,
    retrieval_index: RetrievalIndex,
) -> dict:
    """Decide auto vs. escalate for a classified customer message.

    Returns {"decision": "auto" | "escalate", "reasons": list[str]}.
    "reasons" lists only the checks that actually triggered -- it is empty
    when decision is "auto" (no risk signals found), and always non-empty
    when decision is "escalate" (at least one signal triggered it).
    """
    reasons = []

    if intent == "unknown":
        reasons.append("classifier could not confidently determine intent")
    elif confidence < CONFIDENCE_ESCALATE_THRESHOLD:
        reasons.append(f"classifier confidence ({confidence}) below threshold ({CONFIDENCE_ESCALATE_THRESHOLD})")

    red_flags = _find_red_flags(customer_msg)
    if red_flags:
        reasons.append(f"red-flag language found: {red_flags}")

    retrieved = retrieval_index.query(customer_msg, k=3)
    best_similarity = max((r["similarity"] for r in retrieved), default=0.0)
    if best_similarity < MIN_SIMILARITY_FOR_GROUNDING:
        reasons.append(
            f"no sufficiently similar historical resolution found (best similarity {best_similarity:.2f})"
        )

    decision = "escalate" if reasons else "auto"
    return {"decision": decision, "reasons": reasons}
