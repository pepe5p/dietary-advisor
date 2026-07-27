from dietary_advisor.profile import ActivityLevel, UserProfile
from dietary_advisor.totaller.nutrition import MacroTargets

PROFILES: dict[str, UserProfile] = {
    "regular": UserProfile(
        user_id="regular",
        age=30,
        sex="male",
        height_cm=180.0,
        weight_kg=78.0,
        activity_level=ActivityLevel.LIGHT,
        targets=MacroTargets(energy_kcal=2728.0, protein_g=124.8, carbs_g=386.7, fat_g=75.8, fiber_g=25.0),
    ),
    "preferences": UserProfile(
        user_id="preferences",
        age=30,
        sex="male",
        height_cm=180.0,
        weight_kg=78.0,
        activity_level=ActivityLevel.LIGHT,
        preferred_foods=["chicken", "rice"],
        disliked_foods=["liver", "sausage", "broccoli", "greek yogurt"],
        targets=MacroTargets(energy_kcal=2728.0, protein_g=124.8, carbs_g=386.7, fat_g=75.8, fiber_g=25.0),
        notes="I prefer 3 meals a day + 1 sweet snack. Snack should be some ready shop bought product.",
    ),
    "cut": UserProfile(
        user_id="cut",
        age=23,
        sex="male",
        height_cm=170.0,
        weight_kg=69.0,
        activity_level=ActivityLevel.MODERATE,
        preferred_foods=["natural skyr", "wholegrain pinsa", "all type of berries", "apricots"],
        disliked_foods=["raw tomato", "kefir"],
        targets=MacroTargets(energy_kcal=2000.0, protein_g=150.0, carbs_g=200.0, fat_g=65.0, fiber_g=30.0),
        notes=(
            "I am currently on a cut, so meals should have high volume and low calories density. "
            "I prefer 3 meals a day + 1 snack."
        ),
    ),
    "lactose-intolerant-athlete": UserProfile(
        user_id="lactose-intolerant-athlete",
        age=33,
        sex="male",
        height_cm=178.0,
        weight_kg=72.0,
        conditions=["lactose intolerance"],
        activity_level=ActivityLevel.VERY_ACTIVE,
        targets=MacroTargets(energy_kcal=4000.0, protein_g=180.0, carbs_g=550.0, fat_g=120.0, fiber_g=35.0),
        notes=(
            "I am a professional triathlon athlete. "
            "Meals should be creative and interesting to make eating 4k calories easier."
        ),
    ),
    "vegetarian-allergic": UserProfile(
        user_id="vegetarian-allergic",
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
    ),
    "diabetes-hypertension": UserProfile(
        user_id="diabetes-hypertension",
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
    ),
    "dyslipidemia-obesity": UserProfile(
        user_id="dyslipidemia-obesity",
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
    ),
}


def get_profile(user_id: str) -> UserProfile:
    try:
        return PROFILES[user_id]
    except KeyError:
        raise KeyError(f"Unknown profile id: {user_id!r}. Known ids: {sorted(PROFILES)}") from None
