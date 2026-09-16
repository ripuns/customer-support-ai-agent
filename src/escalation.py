"""Escalation policy: rule-based signals + one LLM-judged signal.

Decides whether an incoming customer message should be auto-handled or
escalated to a human, with a stated reason.

Escalate when:
  - intent is "unknown" (classifier couldn't parse a recognized label at all --
    genuinely can't trust classification enough to draft a safe auto-reply)
  - the LLM judges the message itself to show a red flag: a prior fix
    attempt already failed, account lockout/data-loss, an explicit request
    for a human/exception, or another signal needing human judgment
  - retrieval found no sufficiently similar historical resolution to ground
    a reply in (a proxy for "this is not a well-understood, repeatable case")

Do NOT escalate purely on:
  - negative tone / profanity
  - topic frequency ("many people report this" is not a per-message signal)
  - the message's intent/topic category alone (e.g. all account_security or
    billing_purchase messages)

History: the red-flag check was originally a hardcoded substring list
(RED_FLAG_PHRASES). Tuned against 35 hand-labeled rows it scored 32/35
(91%), but on a disjoint 25-row sample it caught 0/11 true escalate cases.
Replaced with an LLM judgment call (_llm_red_flag_check) that
reads the message for meaning rather than exact wording.

Confidence-threshold signal removed: decide_escalation used to also escalate
when classify_intent's self-reported confidence fell below
CONFIDENCE_ESCALATE_THRESHOLD (60). Diagnosis showed this never fired on
true escalate cases -- on the 11 true-escalate rows in the original 25-row
sample, confidence was always >=85. Root cause: classify_intent's confidence
answers "how sure am I this is battery_performance vs. software_bug", not
"does this need a human" -- a message can be 95% obviously about one topic
while still desperately needing escalation (e.g. "I already tried
everything, still draining"). Those are different questions; conflating them
meant this signal was structurally incapable of catching escalation-worthy
cases. Removed rather than reworked, since the LLM red-flag check above is
already answering the right question directly.

Scope note: this only ever sees the single incoming customer_msg, not prior
conversation turns
Messages that reference a failed prior attempt inline (very common in this
dataset, e.g. "I already tried restarting, still broken") are covered;
tracking risk signals that accumulate *across* separate messages in a
longer conversation is not, and is noted as future work.
"""
from src.llm import call_llm
from src.retrieval import RetrievalIndex

MIN_SIMILARITY_FOR_GROUNDING = 0.15

_RED_FLAG_SYSTEM_PROMPT = """You are screening a single customer support message from a \
Twitter conversation with AppleSupport to decide if it needs a human agent rather than an \
automated reply.

Answer YES (needs a human) if the message itself shows any of:
- the customer says a previously suggested fix did not work (in any wording/tone, including \
sarcasm or minimal replies like "nope" or "still broken")
- account lockout, data loss, or being unable to access something they need
- an explicit request to speak to a human, a supervisor, or for an exception/refund/policy \
decision
- the customer says they've already tried multiple things or contacted support multiple times \
without success

Answer NO if the message is a first-time report of an issue, a routine question, or provides \
information without any of the above signals -- even if the tone is negative, frustrated, or \
profane. Negative tone alone is NOT a reason to answer YES.

Respond in exactly this format, two lines:
ANSWER: YES or NO
REASON: <one short phrase, empty if NO>"""


def _llm_red_flag_check(customer_msg: str) -> str | None:
    """Ask the LLM whether customer_msg shows an escalation-worthy red flag.

    Returns a short reason string if YES, None if NO or unparseable (fails
    open to "no red flag" rather than escalating on a malformed response).
    """
    response = call_llm(_RED_FLAG_SYSTEM_PROMPT, customer_msg, temperature=0.0)
    lines = response.strip().splitlines()
    answer_line = next((l for l in lines if l.upper().startswith("ANSWER:")), "")
    if "YES" not in answer_line.upper():
        return None
    reason_line = next((l for l in lines if l.upper().startswith("REASON:")), "")
    reason = reason_line.split(":", 1)[1].strip() if ":" in reason_line else ""
    return reason or "LLM judged this message as escalation-worthy"


def decide_escalation(
    customer_msg: str,
    intent: str,
    retrieval_index: RetrievalIndex,
    exclude_exact_match: bool = False,
) -> dict:
    """Decide auto vs. escalate for a classified customer message.

    Returns {"decision": "auto" | "escalate", "reasons": list[str]}.
    "reasons" lists only the checks that actually triggered

    exclude_exact_match: pass True during evaluation against golden_set.csv. Without this, the "no good retrieval
    match" signal below can never fire for a golden-set message, since it
    always finds itself at similarity 1.00.
    """
    reasons = []

    if intent == "unknown":
        reasons.append("classifier could not confidently determine intent")

    red_flag_reason = _llm_red_flag_check(customer_msg)
    if red_flag_reason:
        reasons.append(f"red flag: {red_flag_reason}")

    retrieved = retrieval_index.query(customer_msg, k=3, exclude_exact_match=exclude_exact_match)
    best_similarity = max((r["similarity"] for r in retrieved), default=0.0)
    if best_similarity < MIN_SIMILARITY_FOR_GROUNDING:
        reasons.append(
            f"no sufficiently similar historical resolution found (best similarity {best_similarity:.2f})"
        )

    decision = "escalate" if reasons else "auto"
    return {"decision": decision, "reasons": reasons}
