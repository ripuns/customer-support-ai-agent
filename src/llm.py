"""Thin LLM provider wrapper.

Isolates the Gemini API behind a single call_llm function so callers (the
classifier, drafter, escalation policy, and eval judge) don't depend on the
google-genai SDK directly. Swapping providers later means changing this
file only.
"""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

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


def call_llm(system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """Send a single system+user turn to the LLM and return the text response."""
    client = _get_client()
    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
        ),
    )
    return response.text
