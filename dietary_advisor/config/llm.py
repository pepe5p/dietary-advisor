from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter
from pydantic_ai.models.openrouter import OpenRouterModelSettings
from pydantic_ai.settings import ModelSettings, ThinkingEffort

_EFFORT: TypeAdapter[ThinkingEffort] = TypeAdapter(ThinkingEffort)


@dataclass(frozen=True)
class LlmSpec:
    model: str
    reasoning: ThinkingEffort | None = None

    @classmethod
    def parse(cls, value: str) -> LlmSpec:
        """Parse `model` or `model#effort` (e.g. `openrouter:openai/gpt-5.6-luna#high`)."""
        model, _, effort = value.partition("#")
        return cls(model, _EFFORT.validate_python(effort) if effort else None)

    def __str__(self) -> str:
        return self.model if self.reasoning is None else f"{self.model}#{self.reasoning}"

    @property
    def name(self) -> str:
        """Filesystem/key-safe identifier, used in output filenames."""
        return str(self).replace(":", "-").replace("/", "-").replace("#", "-")

    @property
    def model_settings(self) -> ModelSettings:
        """`Agent(model_settings=...)` value for this spec; empty when no effort is set.

        OpenRouter's own `reasoning.effort` is used instead of the unified
        `thinking` setting because the latter collapses `minimal`->`low` and
        `xhigh`->`high`, which would make a `#minimal` spec send a different
        effort than the one recorded.
        """
        if self.reasoning is None:
            return ModelSettings()
        if self.model.startswith("openrouter:"):
            return OpenRouterModelSettings(openrouter_reasoning={"effort": self.reasoning})
        return ModelSettings(thinking=self.reasoning)
