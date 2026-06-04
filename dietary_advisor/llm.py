"""Resolve pydantic-ai model identifiers with provider-appropriate HTTP settings."""

from __future__ import annotations

import httpx
from groq import AsyncGroq
from pydantic_ai.models import Model
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

from dietary_advisor.config import get_api_keys, get_settings


def resolve_llm_model(model: str | Model | None = None) -> str | Model:
    """Return a model suitable for agent construction.

    Groq's SDK defaults to a 60s httpx timeout. On the free tier, 429 rate-limit
    responses trigger multi-second backoffs (often 15-20s per retry), so a single
    multi-turn agent run can exceed 60s and surface as a timeout.
    """
    settings = get_settings()
    model_id = model or settings.llm_model
    if isinstance(model_id, Model):
        return model_id
    if not model_id.startswith("groq:"):
        return model_id
    groq_model_name = model_id.removeprefix("groq:")
    api_key = get_api_keys().groq_api_key
    timeout = httpx.Timeout(settings.llm_request_timeout_s, connect=10.0)
    client = AsyncGroq(
        api_key=api_key,
        timeout=timeout,
        max_retries=settings.llm_max_retries,
    )
    provider = GroqProvider(groq_client=client)
    return GroqModel(groq_model_name, provider=provider)
