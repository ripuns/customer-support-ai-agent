"""LLM-as-judge for reply quality.

Scores a drafted reply against the customer's message on 3 dimensions
required to be evaluable and defensible: grounded, correct, actionable.
Each scored 1-5.
"""
import re

from src.llm import call_llm

SYSTEM_PROMPT = """You are evaluating a customer support reply from AppleSupport on Twitter.

Score the AGENT REPLY on these 3 dimensions, each from 1 (poor) to 5 (excellent):

GROUNDED: Does the reply reflect how AppleSupport actually handles this kind of issue \
(e.g. asking for device/iOS version, directing to DM for account-specific help), rather than \
generic customer-service boilerplate with no real substance?

CORRECT: Is the reply factually sound? Does it avoid inventing specifics (order numbers, case \
numbers, policy claims) that weren't given? Does it not contradict anything known about the issue?

ACTIONABLE: Does the reply give the customer a concrete next step (a question to answer, a \
setting to check, a place to go), rather than a vague acknowledgment with no path forward?

Respond in exactly this format, three lines:
GROUNDED: <1-5>
CORRECT: <1-5>
ACTIONABLE: <1-5>"""

_SCORE_RES = {
    "grounded": re.compile(r"GROUNDED:\s*(\d)", re.IGNORECASE),
    "correct": re.compile(r"CORRECT:\s*(\d)", re.IGNORECASE),
    "actionable": re.compile(r"ACTIONABLE:\s*(\d)", re.IGNORECASE),
}


def judge_reply(customer_msg: str, agent_reply: str) -> dict:
    """Score agent_reply for customer_msg. Returns {"grounded", "correct",
    "actionable"} each 1-5, or None for a dimension if it couldn't be
    parsed from the judge's response (callers should treat None as missing
    data, not as a 0/failing score).
    """
    user_prompt = f"CUSTOMER MESSAGE:\n{customer_msg}\n\nAGENT REPLY:\n{agent_reply}"
    response = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)

    scores = {}
    for dimension, pattern in _SCORE_RES.items():
        match = pattern.search(response)
        scores[dimension] = int(match.group(1)) if match else None
    return scores
