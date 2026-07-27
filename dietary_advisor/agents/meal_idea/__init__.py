"""Meal-idea agent package: brainstorms dish concepts ahead of the nutrition agent."""

from dietary_advisor.agents.meal_idea.agent import build_meal_idea_agent
from dietary_advisor.agents.meal_idea.contract import MealConcept

__all__ = [
    "MealConcept",
    "build_meal_idea_agent",
]
