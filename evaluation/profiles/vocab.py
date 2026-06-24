"""Exemplary categorical vocabulary for authoring and validating eval cases.

`UserProfile.allergens`/`conditions`/`diet_pattern` are free-form strings in
production (validating them is out of scope for this project). These enums
exist only so the evaluation harness has a closed, typo-proof vocabulary when
authoring `evaluation/profiles/cases.py` and deriving ground-truth constraints
in `evaluation/profiles/derive.py`.
"""

from __future__ import annotations

from enum import Enum


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class Allergen(str, Enum):
    """The 14 EU-regulated allergens (Annex II of EU Regulation 1169/2011)."""

    GLUTEN = "gluten"
    CRUSTACEANS = "crustaceans"
    EGGS = "eggs"
    FISH = "fish"
    PEANUTS = "peanuts"
    SOYBEANS = "soybeans"
    MILK = "milk"
    TREE_NUTS = "tree nuts"
    CELERY = "celery"
    MUSTARD = "mustard"
    SESAME = "sesame"
    SULPHITES = "sulphites"
    LUPIN = "lupin"
    MOLLUSCS = "molluscs"


class Condition(str, Enum):
    """Clinical conditions relevant to dietetic guidance (Level 3 of the study)."""

    NONE = "none"
    TYPE_2_DIABETES = "type 2 diabetes"
    TYPE_1_DIABETES = "type 1 diabetes"
    HYPERTENSION = "hypertension"
    DYSLIPIDEMIA = "dyslipidemia"
    CKD_STAGE_3 = "ckd stage 3"
    CELIAC = "celiac"
    LACTOSE_INTOLERANCE = "lactose intolerance"
    IBS = "ibs"
    OBESITY = "obesity"


class DietPattern(str, Enum):
    """Diet pattern preferences / restrictions (Level 2 of the study)."""

    OMNIVORE = "omnivore"
    PESCATARIAN = "pescatarian"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    KETO = "keto"
    MEDITERRANEAN = "mediterranean"
    DASH = "dash"
    LOW_FODMAP = "low_fodmap"
