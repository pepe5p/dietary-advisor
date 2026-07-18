"""Frozen evaluation cases (L1-L3).

Each case owns a self-contained `UserProfile` literal (the same shape a human
would pass via `dietary-advisor --profile`), independent of the CLI's
built-in profiles in `dietary_advisor.profiles`. This decouples the eval
ground truth from the CLI's example-profile store: editing one has no effect
on the other. The `hard_constraints` are the frozen ground-truth evaluation
data for each case.
"""

from __future__ import annotations

from dietary_advisor.schemas.nutrition import MacroTargets, NutrientName
from dietary_advisor.schemas.profile import ActivityLevel, UserProfile
from evaluation.constraints import ConstraintSource, HardConstraint
from evaluation.profiles.eval_profile import EvalProfile

_L1_01_PROFILE = UserProfile(
    user_id="L1_01",
    age=30,
    sex="male",
    height_cm=180.0,
    weight_kg=78.0,
    activity_level=ActivityLevel.MODERATE,
    preferred_foods=["chicken", "rice"],
    targets=MacroTargets(energy_kcal=2728.0, protein_g=124.8, carbs_g=386.7, fat_g=75.8, fiber_g=25.0),
)
_L1_02_PROFILE = UserProfile(
    user_id="L1_02",
    age=28,
    sex="female",
    height_cm=165.0,
    weight_kg=60.0,
    activity_level=ActivityLevel.MODERATE,
    targets=MacroTargets(energy_kcal=2061.9, protein_g=96.0, carbs_g=290.6, fat_g=57.3, fiber_g=25.0),
)
_L1_03_PROFILE = UserProfile(
    user_id="L1_03",
    age=45,
    sex="male",
    height_cm=178.0,
    weight_kg=90.0,
    activity_level=ActivityLevel.LIGHT,
    disliked_foods=["liver"],
    targets=MacroTargets(energy_kcal=1914.7, protein_g=144.0, carbs_g=215.0, fat_g=53.2, fiber_g=25.0),
)
_L1_04_PROFILE = UserProfile(
    user_id="L1_04",
    age=22,
    sex="female",
    height_cm=170.0,
    weight_kg=58.0,
    activity_level=ActivityLevel.ACTIVE,
    preferred_foods=["fish", "vegetables"],
    targets=MacroTargets(energy_kcal=2640.8, protein_g=92.8, carbs_g=402.4, fat_g=73.4, fiber_g=25.0),
)
_L1_05_PROFILE = UserProfile(
    user_id="L1_05",
    age=35,
    sex="male",
    height_cm=185.0,
    weight_kg=82.0,
    activity_level=ActivityLevel.VERY_ACTIVE,
    preferred_foods=["oats", "salmon", "eggs"],
    targets=MacroTargets(energy_kcal=3431.9, protein_g=131.2, carbs_g=512.3, fat_g=95.3, fiber_g=25.0),
)
_L2_01_PROFILE = UserProfile(
    user_id="L2_01",
    age=27,
    sex="female",
    height_cm=168.0,
    weight_kg=62.0,
    activity_level=ActivityLevel.MODERATE,
    allergens=["peanuts", "tree nuts"],
    diet_pattern="vegetarian",
    disliked_foods=["mushrooms"],
    preferred_foods=["lentils", "tofu"],
    targets=MacroTargets(energy_kcal=2129.7, protein_g=99.2, carbs_g=300.1, fat_g=59.2, fiber_g=25.0),
)
_L2_02_PROFILE = UserProfile(
    user_id="L2_02",
    age=33,
    sex="male",
    height_cm=178.0,
    weight_kg=72.0,
    activity_level=ActivityLevel.MODERATE,
    diet_pattern="vegan",
    preferred_foods=["tofu", "tempeh", "lentils"],
    targets=MacroTargets(energy_kcal=2592.4, protein_g=115.2, carbs_g=370.9, fat_g=72.0, fiber_g=25.0),
)
_L2_03_PROFILE = UserProfile(
    user_id="L2_03",
    age=41,
    sex="female",
    height_cm=162.0,
    weight_kg=64.0,
    activity_level=ActivityLevel.LIGHT,
    allergens=["milk"],
    conditions=["lactose intolerance"],
    diet_pattern="pescatarian",
    preferred_foods=["salmon", "rice"],
    targets=MacroTargets(energy_kcal=1768.9, protein_g=102.4, carbs_g=229.3, fat_g=49.1, fiber_g=25.0),
)
_L2_04_PROFILE = UserProfile(
    user_id="L2_04",
    age=35,
    sex="female",
    height_cm=170.0,
    weight_kg=65.0,
    activity_level=ActivityLevel.MODERATE,
    allergens=["gluten"],
    conditions=["celiac"],
    preferred_foods=["rice", "potato"],
    targets=MacroTargets(energy_kcal=2133.6, protein_g=104.0, carbs_g=296.0, fat_g=59.3, fiber_g=25.0),
)
_L2_05_PROFILE = UserProfile(
    user_id="L2_05",
    age=26,
    sex="female",
    height_cm=167.0,
    weight_kg=55.0,
    activity_level=ActivityLevel.MODERATE,
    allergens=["eggs", "soybeans"],
    diet_pattern="vegan",
    preferred_foods=["lentils", "quinoa"],
    targets=MacroTargets(energy_kcal=2019.3, protein_g=88.0, carbs_g=290.6, fat_g=56.1, fiber_g=25.0),
)
_L3_01_PROFILE = UserProfile(
    user_id="L3_01",
    age=58,
    sex="male",
    height_cm=174.0,
    weight_kg=95.0,
    activity_level=ActivityLevel.LIGHT,
    conditions=["type 2 diabetes", "hypertension"],
    disliked_foods=["beef"],
    preferred_foods=["fish", "vegetables", "oats"],
    targets=MacroTargets(energy_kcal=1859.7, protein_g=152.0, carbs_g=196.7, fat_g=51.7, fiber_g=25.0),
    notes="Recently diagnosed; HbA1c 7.4%, BP 145/92.",
)
_L3_02_PROFILE = UserProfile(
    user_id="L3_02",
    age=67,
    sex="female",
    height_cm=160.0,
    weight_kg=70.0,
    activity_level=ActivityLevel.SEDENTARY,
    conditions=["ckd stage 3", "hypertension"],
    preferred_foods=["white rice", "apple"],
    targets=MacroTargets(energy_kcal=1444.8, protein_g=112.0, carbs_g=158.9, fat_g=40.1, fiber_g=25.0),
    notes="eGFR 42; potassium and sodium restriction needed.",
)
_L3_03_PROFILE = UserProfile(
    user_id="L3_03",
    age=52,
    sex="male",
    height_cm=175.0,
    weight_kg=105.0,
    activity_level=ActivityLevel.LIGHT,
    conditions=["dyslipidemia", "obesity"],
    diet_pattern="mediterranean",
    disliked_foods=["organ meats"],
    preferred_foods=["olive oil", "fish", "vegetables"],
    targets=MacroTargets(energy_kcal=2047.0, protein_g=168.0, carbs_g=215.8, fat_g=56.9, fiber_g=25.0),
    notes="LDL 4.2 mmol/L; statin-naive.",
)
_L3_04_PROFILE = UserProfile(
    user_id="L3_04",
    age=47,
    sex="female",
    height_cm=165.0,
    weight_kg=78.0,
    activity_level=ActivityLevel.MODERATE,
    allergens=["gluten"],
    conditions=["type 2 diabetes", "celiac"],
    preferred_foods=["quinoa", "berries"],
    targets=MacroTargets(energy_kcal=1645.2, protein_g=124.8, carbs_g=183.7, fat_g=45.7, fiber_g=25.0),
)
_L3_05_PROFILE = UserProfile(
    user_id="L3_05",
    age=60,
    sex="male",
    height_cm=172.0,
    weight_kg=88.0,
    activity_level=ActivityLevel.LIGHT,
    allergens=["milk"],
    conditions=["hypertension", "lactose intolerance"],
    diet_pattern="dash",
    preferred_foods=["banana", "oats"],
    targets=MacroTargets(energy_kcal=1842.5, protein_g=140.8, carbs_g=204.7, fat_g=51.2, fiber_g=25.0),
)

EVAL_CASES: dict[str, EvalProfile] = {
    "L1_01": EvalProfile(
        case_id="L1_01",
        profile=_L1_01_PROFILE,
        hard_constraints=(),
    ),
    "L1_02": EvalProfile(
        case_id="L1_02",
        profile=_L1_02_PROFILE,
        hard_constraints=(),
    ),
    "L1_03": EvalProfile(
        case_id="L1_03",
        profile=_L1_03_PROFILE,
        hard_constraints=(
            HardConstraint(kind="ingredient_exclusion", target="liver", source=ConstraintSource.PROFILE),
        ),
    ),
    "L1_04": EvalProfile(
        case_id="L1_04",
        profile=_L1_04_PROFILE,
        hard_constraints=(),
    ),
    "L1_05": EvalProfile(
        case_id="L1_05",
        profile=_L1_05_PROFILE,
        hard_constraints=(),
    ),
    "L2_01": EvalProfile(
        case_id="L2_01",
        profile=_L2_01_PROFILE,
        hard_constraints=(
            HardConstraint.allergen("peanuts", source=ConstraintSource.PROFILE),
            HardConstraint.allergen("tree nuts", source=ConstraintSource.PROFILE),
            HardConstraint.diet("vegetarian", source=ConstraintSource.PROFILE),
            HardConstraint(kind="ingredient_exclusion", target="mushrooms", source=ConstraintSource.PROFILE),
        ),
    ),
    "L2_02": EvalProfile(
        case_id="L2_02",
        profile=_L2_02_PROFILE,
        hard_constraints=(HardConstraint.diet("vegan", source=ConstraintSource.PROFILE),),
    ),
    "L2_03": EvalProfile(
        case_id="L2_03",
        profile=_L2_03_PROFILE,
        hard_constraints=(
            HardConstraint.allergen("milk", source=ConstraintSource.PROFILE),
            HardConstraint.diet("pescatarian", source=ConstraintSource.PROFILE),
            HardConstraint.allergen("milk", source=ConstraintSource.SAFETY),
        ),
    ),
    "L2_04": EvalProfile(
        case_id="L2_04",
        profile=_L2_04_PROFILE,
        hard_constraints=(
            HardConstraint.allergen("gluten", source=ConstraintSource.PROFILE),
            HardConstraint.allergen("gluten", source=ConstraintSource.SAFETY),
        ),
    ),
    "L2_05": EvalProfile(
        case_id="L2_05",
        profile=_L2_05_PROFILE,
        hard_constraints=(
            HardConstraint.allergen("eggs", source=ConstraintSource.PROFILE),
            HardConstraint.allergen("soybeans", source=ConstraintSource.PROFILE),
            HardConstraint.diet("vegan", source=ConstraintSource.PROFILE),
        ),
    ),
    "L3_01": EvalProfile(
        case_id="L3_01",
        profile=_L3_01_PROFILE,
        hard_constraints=(
            HardConstraint(kind="ingredient_exclusion", target="beef", source=ConstraintSource.PROFILE),
            HardConstraint.min_nutrient(
                NutrientName.FIBER_G,
                value=30.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale="ADA Standards of Care: target >=30 g fiber/day to improve glycaemic control.",
            ),
            HardConstraint.max_nutrient(
                NutrientName.SUGAR_G,
                value=50.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale=(
                    "WHO conditional recommendation: <10 % of energy from free sugars (~50 g on a 2000 kcal diet)."
                ),
            ),
            HardConstraint.max_nutrient(
                NutrientName.SODIUM_MG,
                value=2000.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale="WHO recommends <2 g/day sodium for adults with hypertension.",
            ),
        ),
    ),
    "L3_02": EvalProfile(
        case_id="L3_02",
        profile=_L3_02_PROFILE,
        hard_constraints=(
            HardConstraint.max_nutrient(
                NutrientName.SODIUM_MG,
                value=2000.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale="KDOQI: <2 g/day sodium for CKD stage 3+.",
            ),
            HardConstraint.max_nutrient(
                NutrientName.POTASSIUM_MG,
                value=2400.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale="KDOQI: <2.4 g/day potassium for CKD stage 3+.",
            ),
            HardConstraint.max_nutrient(
                NutrientName.SODIUM_MG,
                value=2000.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale="WHO recommends <2 g/day sodium for adults with hypertension.",
            ),
        ),
    ),
    "L3_03": EvalProfile(
        case_id="L3_03",
        profile=_L3_03_PROFILE,
        hard_constraints=(
            HardConstraint.diet("mediterranean", source=ConstraintSource.PROFILE),
            HardConstraint(kind="ingredient_exclusion", target="organ meats", source=ConstraintSource.PROFILE),
            HardConstraint.max_nutrient(
                NutrientName.SATURATED_FAT_G,
                value=20.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale="ESC/EAS lipid guideline: limit saturated fat to <10% of energy.",
            ),
            HardConstraint.max_nutrient(
                NutrientName.CHOLESTEROL_MG,
                value=300.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale="Common dyslipidaemia threshold: <300 mg dietary cholesterol/day.",
            ),
        ),
    ),
    "L3_04": EvalProfile(
        case_id="L3_04",
        profile=_L3_04_PROFILE,
        hard_constraints=(
            HardConstraint.allergen("gluten", source=ConstraintSource.PROFILE),
            HardConstraint.min_nutrient(
                NutrientName.FIBER_G,
                value=30.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale="ADA Standards of Care: target >=30 g fiber/day to improve glycaemic control.",
            ),
            HardConstraint.max_nutrient(
                NutrientName.SUGAR_G,
                value=50.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale=(
                    "WHO conditional recommendation: <10 % of energy from free sugars (~50 g on a 2000 kcal diet)."
                ),
            ),
            HardConstraint.allergen("gluten", source=ConstraintSource.SAFETY),
        ),
    ),
    "L3_05": EvalProfile(
        case_id="L3_05",
        profile=_L3_05_PROFILE,
        hard_constraints=(
            HardConstraint.allergen("milk", source=ConstraintSource.PROFILE),
            HardConstraint.diet("dash", source=ConstraintSource.PROFILE),
            HardConstraint.max_nutrient(
                NutrientName.SODIUM_MG,
                value=2000.0,
                source=ConstraintSource.CLINICAL_GUIDELINE,
                rationale="WHO recommends <2 g/day sodium for adults with hypertension.",
            ),
            HardConstraint.allergen("milk", source=ConstraintSource.SAFETY),
        ),
    ),
}


def all_cases() -> list[EvalProfile]:
    return list(EVAL_CASES.values())


def get_case(case_id: str) -> EvalProfile:
    return EVAL_CASES[case_id]
