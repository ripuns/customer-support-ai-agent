"""Thin LLM provider wrapper.

Isolates the Groq API behind a single call_llm function so callers (the
classifier, drafter, escalation policy, and eval judge) don't depend on the
groq SDK directly.
"""
import os
import time

from dotenv import load_dotenv
from groq import Groq
from groq import APIError, APIStatusError

load_dotenv()

# Provider history: OpenAI -> Gemini -> AWS Bedrock (Gemma 3 27B) -> Groq, all on deadline
# day. Gemini's free tier became unusable (rate limits, see DEVLOG.md); AWS Bedrock access
# (via a friend's account) worked but its API key was short-lived and expired mid-validation;
# switched to Groq (free tier, no billing/card signup friction) as the fourth provider.
# See DEVLOG.md for the full incident timeline across all four.
# Model ID verified via a real client.models.list() call, not guessed -- Groq's catalog on
# this key does not include llama-3.3-70b-versatile; largest available general-purpose
# chat model is openai/gpt-oss-120b.
DEFAULT_MODEL = "openai/gpt-oss-120b"

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to a .env file in the repo root "
                "(see .env.example)."
            )
        _client = Groq(api_key=api_key)
    return _client


MAX_RETRIES = 4
RETRY_BASE_DELAY_SECONDS = 5


def call_llm(system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """Send a single system+user turn to the LLM and return the text response.

    Retries with backoff on throttling (429) and transient 5xx errors up to
    MAX_RETRIES times.
    """
    client = _get_client()

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except APIStatusError as e:
            last_error = e
            if e.status_code not in (429, 500, 502, 503):
                raise
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
        except APIError as e:
            last_error = e
            raise
    raise last_error
