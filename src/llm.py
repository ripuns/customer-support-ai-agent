"""Thin LLM provider wrapper.

Isolates the Gemini API behind a single call_llm function so callers (the
classifier, drafter, escalation policy, and eval judge) don't depend on the
google-genai SDK directly.
"""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

load_dotenv()

# gemini-3.6-flash's free tier caps at 5 requests/minute (429 resource_exhausted errors).
# Acceptable for small pool sizes(a few hundred calls) especially with
# the retry/backoff below turning 429s into a slow-but-successful call
# instead of a hard failure. gemini-3.1-flash-lite has no such quota cap on
# this key but is slower per-call (~7s) and prone to transient 503s.
DEFAULT_MODEL = "gemini-3.6-flash"

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to a .env file in the repo root "
                "(see .env.example)."
            )
        _client = genai.Client(api_key=api_key)
    return _client


MAX_RETRIES = 4
RETRY_BASE_DELAY_SECONDS = 5


def call_llm(system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """Send a single system+user turn to the LLM and return the text response.

    Retries with 503 high-demand, 429 rate limit up to MAX_RETRIES times before raising.
    """
    client = _get_client()
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                ),
            )
            return response.text
        except genai_errors.APIError as e:
            last_error = e
            if getattr(e, "code", None) not in (429, 503):
                raise
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
    raise last_error
