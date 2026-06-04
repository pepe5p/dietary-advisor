"""Test scenarios for the ablation study.

Each `Scenario` couples a frozen evaluation case with a user query and optional
soft criteria scored by the G-Eval judge (stage 3).
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.profiles.eval_profile import case_complexity


@dataclass(frozen=True)
class SoftCriterion:
    """One semantic preference derived from the session query."""

    id: str
    description: str


@dataclass(frozen=True)
class Scenario:
    case_id: str
    query: str
    description: str = ""
    soft_criteria: tuple[SoftCriterion, ...] = ()


SCENARIOS: list[Scenario] = [
    # ---- Level 1: healthy adults ----
    Scenario("L1_01", "Plan one balanced day of meals to maintain my weight."),
    Scenario("L1_02", "Suggest a healthy day of meals around 2000 kcal."),
    Scenario(
        "L1_03",
        "I want to lose weight slowly; design a single day's meals with quick, simple recipes.",
        soft_criteria=(
            SoftCriterion(
                "recipe_simplicity",
                "Recipes should be quick and simple (few steps, common techniques).",
            ),
        ),
    ),
    Scenario("L1_04", "Design a meal plan that helps me gain lean mass."),
    Scenario("L1_05", "Design a high-protein day for an active adult."),
    # ---- Level 2: dietary restrictions ----
    Scenario("L2_01", "Plan a vegetarian day, avoiding all nuts."),
    Scenario("L2_02", "Plan a fully plant-based day with adequate protein."),
    Scenario("L2_03", "Plan a pescatarian, dairy-free day."),
    Scenario("L2_04", "Plan a strictly gluten-free day."),
    Scenario("L2_05", "Plan a vegan day with no eggs or soy products."),
    # ---- Level 3: clinical ----
    Scenario("L3_01", "I have type 2 diabetes and high blood pressure - plan a safe day of meals."),
    Scenario("L3_02", "I have stage 3 chronic kidney disease and hypertension - plan one day of meals."),
    Scenario("L3_03", "I have high LDL cholesterol and obesity - design a Mediterranean day."),
    Scenario("L3_04", "I have type 2 diabetes and celiac disease - plan one gluten-free day."),
    Scenario("L3_05", "I have hypertension and lactose intolerance - plan a DASH-style day."),
]


def filter_scenarios(levels: list[int]) -> list[Scenario]:
    return [s for s in SCENARIOS if case_complexity(s.case_id) in levels]
