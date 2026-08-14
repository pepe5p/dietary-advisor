"""Tests for LlmSpec parsing, naming and model settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dietary_advisor.config.llm import LlmSpec


def test_parse_round_trip_without_suffix() -> None:
    spec = LlmSpec.parse("openrouter:openai/gpt-5.6-luna")
    assert spec == LlmSpec(model="openrouter:openai/gpt-5.6-luna")
    assert str(spec) == "openrouter:openai/gpt-5.6-luna"
    assert spec.name == "openrouter-openai-gpt-5.6-luna"


def test_parse_round_trip_with_suffix() -> None:
    spec = LlmSpec.parse("openrouter:openai/gpt-5.6-luna#high")
    assert spec == LlmSpec(model="openrouter:openai/gpt-5.6-luna", reasoning="high")
    assert str(spec) == "openrouter:openai/gpt-5.6-luna#high"
    assert spec.name == "openrouter-openai-gpt-5.6-luna-high"


def test_parse_rejects_unknown_effort() -> None:
    with pytest.raises(ValidationError):
        LlmSpec.parse("openrouter:openai/gpt-5.6-luna#bogus")


def test_model_settings_empty_when_no_effort() -> None:
    assert LlmSpec(model="openrouter:openai/gpt-5.6-luna").model_settings == {}


def test_model_settings_openrouter_passthrough() -> None:
    settings = LlmSpec(model="openrouter:openai/gpt-5.6-luna", reasoning="minimal").model_settings
    assert settings == {"openrouter_reasoning": {"effort": "minimal"}}


def test_model_settings_non_openrouter_uses_thinking() -> None:
    settings = LlmSpec(model="gemini-3.5-flash-lite", reasoning="minimal").model_settings
    assert settings == {"thinking": "minimal"}
