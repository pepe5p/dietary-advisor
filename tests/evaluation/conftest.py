"""Evaluation test fixtures over the session-scoped synthetic food DBs.

Hydration / quantitative tests open the tiny OFF DuckDB built by
``tests.conftest._test_food_dbs`` (never the real ``.data/`` artifacts) and use
a known seeded barcode from ``tests.food_db_fixtures``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dietary_advisor.agents.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.food_db import FoodDb, OffFoodDb
from dietary_advisor.totaller.nutrition import MacroTargets
from evaluation.case_runner.store import RunRecord
from evaluation.records import ScoredRun
from evaluation.scoring.store import ScoreRecord
from evaluation.validation.qualitative import CriterionScore, QualitativeResult
from tests.conftest import LONG_INSTRUCTIONS, LONG_RATIONALE
from tests.food_db_fixtures import ANY_OFF_CODE

# Placeholder barcode for plans that are never hydrated against the DB
# (e.g. the soft-preference judge).
PLACEHOLDER_CODE = "off:0000000000000"


@pytest.fixture()
def off_db() -> Iterator[OffFoodDb]:
    db = OffFoodDb()
    yield db
    db.close()


@pytest.fixture()
def food_db(off_db: OffFoodDb) -> FoodDb:
    return FoodDb(off_db=off_db)


@pytest.fixture()
def any_code() -> str:
    return ANY_OFF_CODE


def agent_plan_single(
    code: str,
    *,
    grams: float = 200.0,
    user_id: str = "test",
    kind: str = "lunch",
    name: str = "Dish",
    food_name: str = "Test food",
    instructions: str = LONG_INSTRUCTIONS,
) -> AgentMealPlan:
    return AgentMealPlan(
        user_id=user_id,
        meals=[
            AgentMeal(
                kind=kind,
                recipe=AgentRecipe(
                    name=name,
                    portions=[PortionRef(code=code, name=food_name, grams=grams)],
                    instructions=instructions,
                ),
            ),
        ],
        rationale=LONG_RATIONALE,
    )


def agent_plan_rice_lunch(grams: float = 200.0, user_id: str = "test", code: str = PLACEHOLDER_CODE) -> AgentMealPlan:
    return agent_plan_single(
        code,
        grams=grams,
        user_id=user_id,
        name="Rice bowl",
        food_name="Rice",
        instructions=LONG_INSTRUCTIONS,
    )


def make_scored_run(
    *,
    llm_model: str = "openrouter:openai/gpt-5.6-luna",
    variant: str = "baseline",
    scenario_id: str = "regular",
    mae_pct: float = 10.0,
    mse_pct: float | None = None,
    per_nutrient_pct: dict[str, float] | None = None,
    soft: float = 0.8,
    safety: float = 1.0,
    safety_violations: list[str] | None = None,
    iterations: int = 1,
    elapsed_s: float = 2.0,
    judge_model: str = "test-judge",
    query: str = "",
) -> ScoredRun:
    return ScoredRun(
        run=RunRecord(
            llm_model=llm_model,
            variant=variant,
            totaller_enabled="totaller" in variant,
            rag_enabled=False,
            reflection_enabled="reflective" in variant,
            scenario_id=scenario_id,
            query=query,
            agent_plan=agent_plan_rice_lunch(user_id=scenario_id),
            targets=MacroTargets(energy_kcal=2000, protein_g=100, carbs_g=200, fat_g=70),
            mae_pct=mae_pct,
            mse_pct=mae_pct**2 if mse_pct is None else mse_pct,
            per_nutrient_pct=per_nutrient_pct if per_nutrient_pct is not None else {"energy_kcal": mae_pct},
            iterations=iterations,
            telemetry={},
            elapsed_s=elapsed_s,
        ),
        score=ScoreRecord(
            llm_model=llm_model,
            variant=variant,
            scenario_id=scenario_id,
            judge_model=judge_model,
            qualitative=QualitativeResult(
                scores=[
                    CriterionScore(
                        criterion_id="recipe-makes-sense",
                        score=soft,
                        reasoning="ok",
                    ),
                ],
                aggregate=soft,
                safety_adherence=safety,
                safety_violations=safety_violations or [],
            ),
            iterations=iterations,
            elapsed_s=elapsed_s,
        ),
    )
