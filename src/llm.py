"""Thin LLM provider wrapper.

Isolates the AWS Bedrock API behind a single call_llm function so callers
(the classifier, drafter, escalation policy, and eval judge) don't depend
on the boto3/Bedrock request shape directly.
"""
import os
import time

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Switched from Gemini to AWS Bedrock (Gemma 3 27B) on deadline day after the Gemini free
# tier became unusable (see DEVLOG.md's "Rate limits" and "Current blocker" entries) and
# paid billing signup failed across GCP, AWS console billing, and several other providers
# due to account-verification issues unrelated to this project. Bedrock access was obtained
# via a friend's already-verified AWS account using Bedrock's API key (bearer token) auth,
# which needs no IAM user/access-key setup -- see DEVLOG.md for the full incident timeline.
# Model ID verified via a real list_foundation_models() call, not guessed -- Bedrock's
# catalog name for this model is "google.gemma-3-27b-it", not the "amazon.gemma-3-27b-..."
# name it's colloquially referred to as.
DEFAULT_MODEL = "google.gemma-3-27b-it"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not api_key:
            raise RuntimeError(
                "AWS_BEARER_TOKEN_BEDROCK not set. Add it to a .env file in the repo root "
                "(see .env.example)."
            )
        # boto3 picks up AWS_BEARER_TOKEN_BEDROCK from the environment automatically for
        # Bedrock's API-key auth mode -- no explicit credentials need to be passed here.
        _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _client


MAX_RETRIES = 4
RETRY_BASE_DELAY_SECONDS = 5


def call_llm(system: str, user: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """Send a single system+user turn to the LLM and return the text response.

    Uses Bedrock's Converse API rather than InvokeModel -- Converse has one standardized
    request/response shape across all Bedrock models, avoiding the need to know each
    model provider's own native JSON format. Retries with backoff on throttling
    (429-equivalent) errors up to MAX_RETRIES times.
    """
    client = _get_client()

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.converse(
                modelId=model,
                messages=[{"role": "user", "content": [{"text": user}]}],
                system=[{"text": system}],
                inferenceConfig={"temperature": temperature},
            )
            return response["output"]["message"]["content"][0]["text"]
        except ClientError as e:
            last_error = e
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code not in ("ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException"):
                raise
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
    raise last_error
