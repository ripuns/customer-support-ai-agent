"""Trivial baseline: the floor the real agent and the simple baseline must beat.

No learning, no retrieval, no LLM calls. Always predicts the majority intent
from the source data, always replies with one fixed generic message, and
always makes the same escalation decision regardless of input. This is the
reference point required by the assignment ("results vs. at least two baselines:
a trivial one and a simple one").
"""
MAJORITY_INTENT = "software_bug"  # 3,829 / 4,971 (77%) of classified apple_triples.csv rows

GENERIC_REPLY = (
    "Thanks for reaching out. We'd like to help. Could you DM us more details "
    "so we can look into this together?"
)

# Always-auto is the trivial choice for escalation: it requires no signal at
# all, unlike always-escalate (which would trivially "solve" recall but is
# an equally uninformed rule -- both are valid trivial choices; auto was
# picked so the trivial baseline's auto/escalate precision and recall are
# both meaningfully measurable against the golden set rather than one being
# trivially 100% or undefined).
FIXED_DECISION = "auto"


def trivial_classify(customer_msg: str) -> dict:
    return {"intent": MAJORITY_INTENT, "confidence": None}


def trivial_draft(customer_msg: str) -> dict:
    return {"reply": GENERIC_REPLY, "grounded_on": []}


def trivial_escalate(customer_msg: str) -> dict:
    return {"decision": FIXED_DECISION, "reasons": ["trivial baseline: fixed decision, no signal used"]}
