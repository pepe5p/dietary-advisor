"""Tests for the SQLite profile store + ProfileService constraint derivation."""

from __future__ import annotations

import json
from pathlib import Path

from dietary_advisor.profile_manager.service import ProfileService
from dietary_advisor.profile_manager.store import ProfileStore
from dietary_advisor.schemas.nutrition import NutrientName
from dietary_advisor.schemas.profile import (
    Allergen,
    Condition,
    DietPattern,
    Sex,
    UserProfile,
)


def test_store_round_trip() -> None:
    store = ProfileStore()
    p = UserProfile(
        user_id="rt",
        age=30,
        sex=Sex.MALE,
        height_cm=180,
        weight_kg=80,
        allergens=[Allergen.MILK],
        diet_pattern=DietPattern.VEGETARIAN,
    )
    store.upsert(p)
    loaded = store.get("rt")
    assert loaded is not None
    assert loaded.user_id == "rt"
    assert loaded.allergens == [Allergen.MILK]
    assert loaded.diet_pattern == DietPattern.VEGETARIAN


def test_store_delete() -> None:
    store = ProfileStore()
    p = UserProfile(user_id="del", age=20, sex=Sex.FEMALE, height_cm=160, weight_kg=55)
    store.upsert(p)
    assert store.delete("del")
    assert store.get("del") is None
    assert not store.delete("del")


def test_store_import_dir(tmp_path: Path) -> None:
    d = tmp_path / "profiles"
    d.mkdir()
    p = UserProfile(user_id="imp", age=25, sex=Sex.MALE, height_cm=175, weight_kg=70)
    (d / "imp.json").write_text(json.dumps(p.model_dump(mode="json")), encoding="utf-8")

    store = ProfileStore()
    loaded = store.import_dir(d)
    assert len(loaded) == 1
    assert store.get("imp") is not None


def test_service_derives_allergen_constraints() -> None:
    store = ProfileStore()
    service = ProfileService(store=store)
    p = UserProfile(
        user_id="x",
        age=27,
        sex=Sex.FEMALE,
        height_cm=170,
        weight_kg=60,
        allergens=[Allergen.PEANUTS, Allergen.MILK],
        diet_pattern=DietPattern.VEGAN,
        disliked_foods=["mushroom"],
    )
    constraints = service.derive_hard_constraints(p)
    kinds = {(c.kind, c.target) for c in constraints}
    assert ("allergen_exclusion", "peanuts") in kinds
    assert ("allergen_exclusion", "milk") in kinds
    assert ("diet_pattern", "vegan") in kinds
    assert ("ingredient_exclusion", "mushroom") in kinds


def test_service_derives_clinical_rules_for_hypertension() -> None:
    store = ProfileStore()
    service = ProfileService(store=store)
    p = UserProfile(
        user_id="hyp",
        age=58,
        sex=Sex.MALE,
        height_cm=174,
        weight_kg=95,
        conditions=[Condition.HYPERTENSION],
    )
    constraints = service.derive_hard_constraints(p)
    sodium_rules = [c for c in constraints if c.kind == "max_nutrient" and c.target == NutrientName.SODIUM_MG.value]
    assert sodium_rules
    assert sodium_rules[0].value == 2000.0
