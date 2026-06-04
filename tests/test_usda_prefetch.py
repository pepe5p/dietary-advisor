"""USDA shortlist prefetch."""

from __future__ import annotations

from unittest.mock import MagicMock

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.schemas.nutrition import FoodItem, NutrientName
from dietary_advisor.schemas.profile import DietPattern, Sex, UserProfile
from dietary_advisor.tools.usda_prefetch import prefetch_usda_shortlist, summarize_shortlist


def test_prefetch_populates_shortlist() -> None:
    usda = MagicMock()
    usda.search.return_value = [
        FoodItem(fdc_id=1, name="Chicken", nutrients_per_100g={NutrientName.ENERGY_KCAL: 165.0}),
    ]
    profile = UserProfile(
        user_id="u",
        age=30,
        sex=Sex.MALE,
        height_cm=180,
        weight_kg=78,
        diet_pattern=DietPattern.OMNIVORE,
    )
    deps = AgentDeps(profile=profile, targets=MagicMock(), usda=usda)
    added = prefetch_usda_shortlist(deps, page_size=1)
    assert added
    assert deps.shortlist
    assert usda.search.call_count == 5


def test_summarize_shortlist_truncates() -> None:
    items = [FoodItem(name=f"Food {i}") for i in range(20)]
    text = summarize_shortlist(items, limit=3)
    assert "Food 0" in text
    assert "17 more" in text
