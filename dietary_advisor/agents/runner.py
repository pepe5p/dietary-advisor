"""Iterating agent runner that logs why each model request happens.

`Agent.run` hides the request/tool-call graph behind a single await, which
makes the flood of `generateContent` HTTP calls in the logs unreadable: there
is no way to tell a tool round-trip apart from the final answer. This wraps
`Agent.iter` so callers get identical behaviour (same `AgentRunResult`) plus
an INFO-level narration of each request. Per-tool call/return detail is
logged by the tool implementations themselves (food_db, retriever, totaller,
...), not here, so this only names which tool was invoked rather than dumping
its (possibly large) arguments.

Rejected structured outputs are the exception: because every agent goes
through this runner, it is also the one place that can warn - for all of them
at once - when a `final_result` call is sent back for a retry (bad schema, or
an output validator like the nutrition agent's `_validate_codes`). Those
retries are otherwise invisible, since a run that eventually succeeds looks
identical to one that got it right first time.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolCallPart

log = logging.getLogger(__name__)

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")

# pydantic-ai names the output tool "final_result", suffixed with the type name
# when an agent declares several output types (`DEFAULT_OUTPUT_TOOL_NAME`).
_OUTPUT_TOOL_PREFIX = "final_result"


def _warn_on_rejected_output(request: ModelRequest, label: str) -> None:
    """Warn about every rejected output tool call this outgoing request carries.

    A rejection reaches the model as a `RetryPromptPart` on the next request,
    which is why the outgoing request is where it becomes observable. A call
    rejected on the *last* allowed attempt gets no next request: it surfaces to
    the caller as `UnexpectedModelBehavior` instead.
    """
    for part in request.parts:
        if isinstance(part, RetryPromptPart) and (part.tool_name or "").startswith(_OUTPUT_TOOL_PREFIX):
            log.warning("[%s] %s rejected, asking the model to retry: %s", label, part.tool_name, _retry_reason(part))


def _retry_reason(part: RetryPromptPart) -> str:
    if isinstance(part.content, str):
        return part.content
    return "; ".join(
        f"{'.'.join(str(loc) for loc in error['loc']) or '<root>'}: {error['msg']}" for error in part.content
    )


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
                _warn_on_rejected_output(node.request, label)
                log.info("[%s] -> model request", label)
            elif Agent.is_call_tools_node(node):
                for part in node.model_response.parts:
                    if isinstance(part, ToolCallPart):
                        log.info("[%s] tool call: %s", label, part.tool_name)
    assert run.result is not None  # run.iter always yields a result once the context exits cleanly
    return run.result
