"""`run_agent_logged`: the shared runner, including its warning on rejected outputs."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dietary_advisor.agents.runner import run_agent_logged


class _Answer(BaseModel):
    value: int


def _output_call(args: dict[str, object]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart("final_result", args)])


def _scripted_model(*responses: ModelResponse) -> FunctionModel:
    """A model that replays `responses` in order, one per request."""
    calls = iter(responses)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return next(calls)

    return FunctionModel(respond)


def _output_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.asyncio()
async def test_schema_rejection_of_final_result_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    agent = Agent(
        _scripted_model(_output_call({"value": "not-an-int"}), _output_call({"value": 7})),
        output_type=_Answer,
        retries=2,
    )

    with caplog.at_level(logging.WARNING, logger="dietary_advisor.agents.runner"):
        result = await run_agent_logged(agent, "go", deps=None, label="probe")

    assert result.output == _Answer(value=7)
    warnings = _output_warnings(caplog)
    assert len(warnings) == 1
    assert "[probe] final_result rejected" in warnings[0]
    assert "value:" in warnings[0]


@pytest.mark.asyncio()
async def test_output_validator_rejection_of_final_result_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    agent = Agent(
        _scripted_model(_output_call({"value": 1}), _output_call({"value": 2})),
        output_type=_Answer,
        retries=2,
    )

    @agent.output_validator
    def only_even(answer: _Answer) -> _Answer:
        if answer.value % 2:
            raise ModelRetry("value must be even")
        return answer

    with caplog.at_level(logging.WARNING, logger="dietary_advisor.agents.runner"):
        result = await run_agent_logged(agent, "go", deps=None, label="probe")

    assert result.output == _Answer(value=2)
    assert _output_warnings(caplog) == ["[probe] final_result rejected, asking the model to retry: value must be even"]


@pytest.mark.asyncio()
async def test_accepted_output_and_tool_retries_are_not_warned(caplog: pytest.LogCaptureFixture) -> None:
    agent = Agent(
        _scripted_model(
            ModelResponse(parts=[ToolCallPart("flaky", {})]),
            _output_call({"value": 1}),
        ),
        output_type=_Answer,
        retries=2,
    )

    @agent.tool_plain
    def flaky() -> str:
        raise ModelRetry("tool says try again")

    with caplog.at_level(logging.WARNING, logger="dietary_advisor.agents.runner"):
        await run_agent_logged(agent, "go", deps=None, label="probe")

    assert _output_warnings(caplog) == []


@pytest.mark.asyncio()
async def test_plain_text_instead_of_final_result_is_not_reported_as_a_failed_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing output call has no `tool_name`, so it isn't attributed to `final_result`."""
    agent = Agent(
        _scripted_model(ModelResponse(parts=[TextPart("here you go")]), _output_call({"value": 1})),
        output_type=_Answer,
        retries=2,
    )

    with caplog.at_level(logging.WARNING, logger="dietary_advisor.agents.runner"):
        await run_agent_logged(agent, "go", deps=None, label="probe")

    assert _output_warnings(caplog) == []
