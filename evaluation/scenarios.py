"""Test scenarios for the ablation study.

Each `Scenario` couples a profile id with one or more user queries that
exercise the system at that complexity level.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    profile_id: str
    level: int
    query: str
    description: str = ""


# A small but representative grid: 5 profiles x 1 query per level. The runner
# multiplies this by V0..V4 and the configured `--repeats`.
SCENARIOS: list[Scenario] = [
    # ---- Level 1: healthy adults, baseline meal planning ----
    Scenario("L1_01", 1, "Plan one balanced day of meals to maintain my weight."),
    Scenario("L1_02", 1, "Suggest a healthy day of meals around 2000 kcal."),
    Scenario("L1_03", 1, "I want to lose weight slowly; design a single day's meals."),
    Scenario("L1_04", 1, "Design a meal plan that helps me gain lean mass."),
    Scenario("L1_05", 1, "Design a high-protein day for an active adult."),
    # ---- Level 2: dietary restrictions / preferences ----
    Scenario("L2_01", 2, "Plan a vegetarian day, avoiding all nuts."),
    Scenario("L2_02", 2, "Plan a fully plant-based day with adequate protein."),
    Scenario("L2_03", 2, "Plan a pescatarian, dairy-free day."),
    Scenario("L2_04", 2, "Plan a strictly gluten-free day."),
    Scenario("L2_05", 2, "Plan a vegan day with no eggs or soy products."),
    # ---- Level 3: clinical comorbidities ----
    Scenario("L3_01", 3, "I have type 2 diabetes and high blood pressure - plan a safe day of meals."),
    Scenario("L3_02", 3, "I have stage 3 chronic kidney disease and hypertension - plan one day of meals."),
    Scenario("L3_03", 3, "I have high LDL cholesterol and obesity - design a Mediterranean day."),
    Scenario("L3_04", 3, "I have type 2 diabetes and celiac disease - plan one gluten-free day."),
    Scenario("L3_05", 3, "I have hypertension and lactose intolerance - plan a DASH-style day."),
]


def filter_scenarios(levels: list[int]) -> list[Scenario]:
    return [s for s in SCENARIOS if s.level in levels]
