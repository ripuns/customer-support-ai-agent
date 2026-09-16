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

# gemini-3.6-flash has TWO separate free-tier caps on this key, discovered one at a time:
#   1. 5 requests/minute (429 RESOURCE_EXHAUSTED) -- fixed by the throttle below.
#   2. 20 requests/DAY total (a *different* 429, quotaId
#      "GenerateRequestsPerDayPerProjectPerModel-FreeTier") -- discovered when a 25-row
#      harness eval run died partway through after exhausting the day's quota. No amount
#      of throttling or backoff can work around a daily cap; the only fixes are enabling
#      billing or switching model. Switched DEFAULT_MODEL to gemini-3.1-flash-lite, which
#      empirically handles bursts of 20+ calls without hitting either limit on this key
#      (per-minute or per-day) -- see src/README.md for the verification history. It is
#      slower per-call (~7s) and occasionally returns transient 503s, both already handled
#      by the retry logic below.
DEFAULT_MODEL = "gemini-3.1-flash-lite"
MIN_SECONDS_BETWEEN_CALLS = 3.0  # gemini-3.1-flash-lite has no observed per-minute cap on
# this key; a small spacing is still kept as a courtesy/safety margin rather than firing
# requests back-to-back with zero delay.

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
