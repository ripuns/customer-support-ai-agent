"""Grounded reply drafter.

Given an incoming customer message and its classified intent, retrieves
similar historical AppleSupport resolutions and prompts the LLM to draft a 
reply grounded in how the brand has actually handled similar issues before.
"""
from src.intents import INTENTS
from src.llm import call_llm
from src.retrieval import RetrievalIndex

MIN_SIMILARITY_TO_USE = 0.15

SYSTEM_PROMPT_TEMPLATE = (
    "You are an AppleSupport customer support agent replying on Twitter. "
    "Write a short, helpful reply to the customer's message below.\n\n"
    "The customer's message has been classified as intent: {intent} ({intent_description})\n\n"
    "Here are real past examples of how AppleSupport has replied to similar customer messages. "
    "Use them as a guide for tone, structure, and typical next steps (e.g. asking for device/iOS "
    "version, directing to DM for account-specific help) -- do not copy them verbatim, and do not "
    "invent specific facts (order numbers, case numbers, links) that aren't given to you.\n\n"
    "{examples}\n\n"
    "Reply in AppleSupport's style: brief, polite, and helpful. If the issue can't be resolved in "
    "a single public reply (needs account access, personal details, or a definitive fix you "
    "can't verify), direct the customer to DM as AppleSupport typically does."
)


def _format_examples(retrieved: list[dict]) -> str:
    if not retrieved:
        return "(No closely similar historical examples were found.)"
    blocks = []
    for i, r in enumerate(retrieved, 1):
        blocks.append(
            f"Example {i} (similarity {r['similarity']:.2f}):\n"
            f"  Customer: {r['customer_msg']}\n"
            f"  AppleSupport reply: {r['brand_reply']}"
        )
    return "\n\n".join(blocks)


def draft_reply(
    customer_msg: str,
    intent: str,
    retrieval_index: RetrievalIndex,
    k: int = 3,
    exclude_exact_match: bool = False,
) -> dict:
    """Draft a grounded reply for customer_msg, classified as intent.

    Returns {"reply": str, "grounded_on": list[dict]} where grounded_on is
    the list of retrieved historical examples actually used as context
    (filtered to similarity >= MIN_SIMILARITY_TO_USE; may be empty if
    nothing sufficiently similar was found, in which case the reply is
    drafted without grounding examples and this should be treated as a
    weaker-confidence draft by callers).

    exclude_exact_match: pass True during evaluation against golden_set.csv,
    since those messages exist verbatim in the retrieval index -- see
    RetrievalIndex.query's docstring for why.
    """
    retrieved = retrieval_index.query(customer_msg, k=k, exclude_exact_match=exclude_exact_match)
    grounded_on = [r for r in retrieved if r["similarity"] >= MIN_SIMILARITY_TO_USE]

    intent_description = INTENTS.get(intent, "No description available.")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        intent=intent,
        intent_description=intent_description,
        examples=_format_examples(grounded_on),
    )

    reply = call_llm(system_prompt, customer_msg, temperature=0.3)

    return {"reply": reply.strip(), "grounded_on": grounded_on}
