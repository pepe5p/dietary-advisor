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
