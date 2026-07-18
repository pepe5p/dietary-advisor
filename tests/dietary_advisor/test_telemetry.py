"""Tests for run telemetry (tool-call counts + token usage)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.usage import RunUsage

from dietary_advisor.telemetry import collect_from_result, RunTelemetry


@dataclass
class _FakeResult:
    """Minimal stand-in for `AgentRunResult` (only the two methods we read)."""

    messages: list[ModelMessage]
    run_usage: RunUsage = field(default_factory=RunUsage)

    def all_messages(self) -> list[ModelMessage]:
        return self.messages

    def usage(self) -> RunUsage:
        return self.run_usage


def test_collect_from_result_counts_tool_calls_by_name() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name="lookup_food", args={"query": "rice"}, tool_call_id="1"),
                ToolCallPart(tool_name="lookup_food", args={"query": "chicken"}, tool_call_id="2"),
                ToolCallPart(tool_name="total_meal_plan", args={}, tool_call_id="3"),
            ],
        ),
        ModelResponse(parts=[TextPart(content="done")]),
    ]
    result = _FakeResult(messages=messages, run_usage=RunUsage(requests=2, input_tokens=120, output_tokens=40))

    telemetry = collect_from_result(result)

    assert telemetry.tool_calls == {"lookup_food": 2, "total_meal_plan": 1}
    assert telemetry.total_tool_calls == 3
    assert telemetry.requests == 2
    assert telemetry.input_tokens == 120
    assert telemetry.output_tokens == 40
    assert telemetry.total_tokens == 160


def test_collect_from_result_handles_no_tool_calls() -> None:
    messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content="no tools needed")])]
    result = _FakeResult(messages=messages, run_usage=RunUsage(requests=1, input_tokens=10, output_tokens=5))

    telemetry = collect_from_result(result)

    assert telemetry.tool_calls == {}
    assert telemetry.total_tool_calls == 0


def test_run_telemetry_merge_combines_counts_and_usage() -> None:
    a = RunTelemetry(tool_calls=Counter({"lookup_food": 2}), requests=1, input_tokens=100, output_tokens=20)
    b = RunTelemetry(
        tool_calls=Counter({"lookup_food": 1, "total_meal_plan": 3}),
        requests=2,
        input_tokens=50,
        output_tokens=10,
    )

    merged = a.merge(b)

    assert merged.tool_calls == {"lookup_food": 3, "total_meal_plan": 3}
    assert merged.requests == 3
    assert merged.input_tokens == 150
    assert merged.output_tokens == 30
    assert merged.total_tokens == 180
    # merge() must not mutate either operand.
    assert a.tool_calls == {"lookup_food": 2}
    assert b.tool_calls == {"lookup_food": 1, "total_meal_plan": 3}


def test_run_telemetry_as_dict_is_json_friendly() -> None:
    telemetry = RunTelemetry(tool_calls=Counter({"lookup_food": 2}), requests=1, input_tokens=100, output_tokens=20)
    payload = telemetry.as_dict()

    assert payload == {
        "tool_calls": {"lookup_food": 2},
        "total_tool_calls": 2,
        "requests": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
