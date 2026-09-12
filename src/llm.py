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

# gemini-3.6-flash's free tier caps at 5 requests/minute (429 resource_exhausted errors)
# a live run only achieved a 14.78% success rate because backoff delays a failed call's retries 
# but does not throttle the rate new calls are issued at. MIN_SECONDS_BETWEEN_CALLS below is the
# actual fix: it enforces the request rate itself, before any call is made, so this
# wrapper is now safe to use in a loop over many examples

DEFAULT_MODEL = "gemini-3.6-flash"
MIN_SECONDS_BETWEEN_CALLS = 12.5  # slightly over 60/5=12s to leave margin

_client = None
_last_call_time = None


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


def _throttle():
    """Block until at least MIN_SECONDS_BETWEEN_CALLS has passed since the last call."""
    global _last_call_time
    if _last_call_time is not None:
        elapsed = time.monotonic() - _last_call_time
        remaining = MIN_SECONDS_BETWEEN_CALLS - elapsed
        if remaining > 0:
            time.sleep(remaining)
    _last_call_time = time.monotonic()


MAX_RETRIES = 4
RETRY_BASE_DELAY_SECONDS = 5


def call_llm(system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """Send a single system+user turn to the LLM and return the text response.

    Throttles to MIN_SECONDS_BETWEEN_CALLS between requests (enforcing the actual
    5-requests/minute free-tier limit) before every call, then retries with backoff
    on transient 429/503 errors up to MAX_RETRIES times as a second line of defense.
    """
    client = _get_client()
    last_error = None
    for attempt in range(MAX_RETRIES):
        _throttle()
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
