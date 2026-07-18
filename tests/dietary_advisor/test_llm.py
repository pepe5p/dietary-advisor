"""LLM model resolution helpers."""

from __future__ import annotations

import httpx
from pydantic_ai.models import Model
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIChatModel

from dietary_advisor.llm import resolve_llm_model


def test_resolve_groq_model_uses_extended_timeout() -> None:
    model = resolve_llm_model("groq:llama-3.3-70b-versatile", groq_api_key="test-key")
    assert isinstance(model, GroqModel)
    assert model.model_name == "llama-3.3-70b-versatile"
    timeout = model.client.timeout
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read is not None
    assert timeout.read >= 300.0


def test_resolve_infers_non_groq_string_to_a_model() -> None:
    model = resolve_llm_model("openai:gpt-4o-mini")
    assert isinstance(model, Model)
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "gpt-4o-mini"
