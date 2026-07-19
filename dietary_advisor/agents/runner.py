"""Iterating agent runner that logs why each model request happens.

`Agent.run` hides the request/tool-call graph behind a single await, which
makes the flood of `generateContent` HTTP calls in the logs unreadable: there
is no way to tell a tool round-trip apart from the final answer. This wraps
`Agent.iter` so callers get identical behaviour (same `AgentRunResult`) plus
an INFO-level narration of each request. Per-tool call/return detail is
logged by the tool implementations themselves (food_db, retriever, totaller,
...), not here, so this only names which tool was invoked rather than dumping
its (possibly large) arguments.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ToolCallPart

log = logging.getLogger(__name__)

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


async def run_agent_logged(
    agent: Agent[DepsT, OutputT],
    prompt: str,
    *,
    deps: DepsT,
    label: str,
) -> AgentRunResult[OutputT]:
    """Run `agent` via `Agent.iter`, logging each node at INFO level.

    Equivalent to `await agent.run(prompt, deps=deps)` but narrates, per
    node, whether the model is being called or which tool it asked for - so
    each `httpx` request line in the logs can be attributed to a cause.
    """
    async with agent.iter(prompt, deps=deps) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                log.info("[%s] -> model request", label)
            elif Agent.is_call_tools_node(node):
                for part in node.model_response.parts:
                    if isinstance(part, ToolCallPart):
                        log.info("[%s] tool call: %s", label, part.tool_name)
    assert run.result is not None  # run.iter always yields a result once the context exits cleanly
    return run.result
