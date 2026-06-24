"""Run telemetry: tool-call and token-usage counters for `--verbose` debugging.

Collected unconditionally (cheap - it just walks messages already held in
memory) but only rendered by the CLI when `--verbose` is set, so normal runs
stay uncluttered while debugging an agent's tool-use behaviour stays possible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.usage import RunUsage


class _AgentRunResultLike(Protocol):
    def all_messages(self) -> list[ModelMessage]: ...
    def usage(self) -> RunUsage: ...


@dataclass
class RunTelemetry:
    """Aggregated tool-call counts and LLM usage for one or more agent runs."""

    tool_calls: Counter[str] = field(default_factory=Counter)
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merge(self, other: RunTelemetry) -> RunTelemetry:
        """Return a new `RunTelemetry` combining `self` and `other`."""
        merged = Counter(self.tool_calls)
        merged.update(other.tool_calls)
        return RunTelemetry(
            tool_calls=merged,
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "tool_calls": dict(self.tool_calls),
            "total_tool_calls": self.total_tool_calls,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def collect_from_result(result: _AgentRunResultLike) -> RunTelemetry:
    """Build a `RunTelemetry` from a finished pydantic-ai agent run result."""
    tool_calls: Counter[str] = Counter()
    for message in result.all_messages():
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolCallPart):
                tool_calls[part.tool_name] += 1

    usage = result.usage()
    return RunTelemetry(
        tool_calls=tool_calls,
        requests=usage.requests,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
