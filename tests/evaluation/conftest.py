"""Evaluation test fixtures: reuse the real OFF product DB (no mocks).

Tests that resolve portions to nutrients open the real ``.data/off/off_pl.duckdb``
built by ``setup`` and use real product barcodes discovered at runtime. When
the DB is not present (e.g. in CI, which does not build the 7.6 GB-derived
artifact) the DB-dependent fixtures ``pytest.skip`` the test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from dietary_advisor.agents.agent_output import AgentMeal, AgentMealPlan, AgentRecipe, PortionRef
from dietary_advisor.config import get_settings
from dietary_advisor.food_db import FoodDb, OffFoodDb
from dietary_advisor.food_db.off_food_db import to_code
from tests.conftest import LONG_INSTRUCTIONS, LONG_RATIONALE

# Placeholder barcode for plans that are never hydrated against the DB
# (e.g. the soft-preference judge).
PLACEHOLDER_CODE = "off:0000000000000"


def _off_db_path() -> Path:
    return get_settings().off_db


def _query_code(sql: str) -> str | None:
    path = _off_db_path()
    if not path.exists():
        return None
    con = duckdb.connect(str(path), read_only=True)
    try:
        row = con.execute(sql).fetchone()
        return to_code(row[0]) if row else None
    finally:
        con.close()


@pytest.fixture()
def off_db() -> Iterator[OffFoodDb]:
    path = _off_db_path()
    db = OffFoodDb(path)
    yield db
    db.close()


@pytest.fixture()
def food_db(off_db: OffFoodDb) -> FoodDb:
    """The `FoodDb` facade over the real OFF reader; the lookup surface eval code takes."""
    return FoodDb(off_db=off_db)


@pytest.fixture()
def any_code() -> str:
    """A real barcode of a product with a positive calorie value."""
    code = _query_code(
        "SELECT code FROM products WHERE product_name IS NOT NULL AND energy_kcal_in_100g > 0 LIMIT 1",
    )
    if code is None:
        pytest.skip("OFF product DB unavailable or empty")
    return code


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
