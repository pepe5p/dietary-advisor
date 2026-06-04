"""LLM model resolution helpers."""

from __future__ import annotations

import httpx
import pytest
from pydantic_ai.models.groq import GroqModel

from dietary_advisor.llm import resolve_llm_model


def test_resolve_groq_model_uses_extended_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from dietary_advisor.config import get_api_keys

    get_api_keys.cache_clear()
    model = resolve_llm_model("groq:llama-3.3-70b-versatile")
    assert isinstance(model, GroqModel)
    assert model.model_name == "llama-3.3-70b-versatile"
    timeout = model.client.timeout
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read is not None
    assert timeout.read >= 300.0
    get_api_keys.cache_clear()


def test_resolve_passes_through_non_groq_string() -> None:
    assert resolve_llm_model("openai:gpt-4o-mini") == "openai:gpt-4o-mini"
