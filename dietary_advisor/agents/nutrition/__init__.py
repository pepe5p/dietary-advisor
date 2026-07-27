"""Nutrition agent package: factory, prompts, and (shared) refiner factory.

The refiner lives here because it is the same agent with a different prompt -
same tools, same output validator, same `AgentMealPlan` output type.
"""

from dietary_advisor.agents.nutrition.agent import build_nutrition_agent, build_refiner_agent

__all__ = [
    "build_nutrition_agent",
    "build_refiner_agent",
]
