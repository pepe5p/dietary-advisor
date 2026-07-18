"""The blueprint agent: a tool-free brainstorming pass ahead of the nutrition agent."""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from dietary_advisor.agents.blueprint_agent import build_blueprint_agent
from dietary_advisor.config import get_settings
from dietary_advisor.schemas.blueprint import MealConcept


def _tool_names(agent: object) -> set[str]:
    return set(agent._function_toolset.tools)  # type: ignore[attr-defined]


def test_blueprint_agent_registers_no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(get_settings().__dict__, "resolved_llm_model", TestModel())

    agent = build_blueprint_agent()

    assert _tool_names(agent) == set()
    assert agent.output_type == list[MealConcept]
