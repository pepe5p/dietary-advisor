"""Test scenarios for the ablation study.

Each `Scenario` couples a `dietary_advisor.profiles` id with a user query and
optional soft criteria scored by the G-Eval judge. An empty `query` means no
session request - the pipeline omits the user-request prompt block entirely.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SoftCriterion:
    """One semantic preference derived from the session query."""

    id: str
    description: str


ALWAYS_SCORED_SOFT_CRITERIA: tuple[SoftCriterion, ...] = (
    SoftCriterion(
        "recipe-makes-sense",
        (
            "Recipes should be understandable and coherent. "
            "No physically impossible or incoherent steps e.g. 'pour a 1l of something "
            "into a bowl' or 'put oven tray to microwave oven'. "
            "Quantities and instructions should match the ingredients listed."
        ),
    ),
)
PREFERENCES_SOFT_CRITERION = SoftCriterion(
    "preferences-satisfaction",
    (
        "User mentioned in their profile that they prefer some specific foods, "
        "recipes don't have to include them all, but at least one of them should be present."
        "Also, disliked foods should not be present in the recipes."
    ),
)


@dataclass(frozen=True)
class Scenario:
    case_id: str
    profile_id: str
    query: str
    description: str
    soft_criteria: tuple[SoftCriterion, ...] = ()


SCENARIOS: dict[str, Scenario] = {
    "regular": Scenario(
        case_id="regular",
        profile_id="regular",
        query="",
        description="The easiest case, no special session requirements beyond coherent recipes and macro targets.",
    ),
    "quick-and-easy": Scenario(
        case_id="quick-and-easy",
        profile_id="regular",
        query="I don't have much time to prepare meals, so I need quick and easy recipes. The simpler the better.",
        description="Regular user needs to prepare meals quickly and easily.",
        soft_criteria=(
            SoftCriterion(
                "simple-recipe",
                "Recipes should be quick and simple (few steps, common techniques, no obscure ingredients).",
            ),
        ),
    ),
    "preferences": Scenario(
        case_id="preferences",
        profile_id="preferences",
        query="",
        description="User has some preferences for the recipes, so they should be included.",
        soft_criteria=(
            PREFERENCES_SOFT_CRITERION,
            SoftCriterion(
                "shop-bought-snack",
                "User wants one sweet snack that can be bought in a shop like Oreo or Twix.",
            ),
        ),
    ),
    "cut": Scenario(
        case_id="cut",
        profile_id="cut",
        query="I am on a cut so long, add some another sweet snack today that won't ruin my progress.",
        description="User on a reduction diet needs to add a sweet snack.",
        soft_criteria=(
            PREFERENCES_SOFT_CRITERION,
            SoftCriterion(
                "reduction-diet",
                (
                    "Recipes should be filling and satisfying but not too high in calories. "
                    "It is also important not to exceed fats and calories."
                ),
            ),
            SoftCriterion(
                "meals-composition",
                (
                    "User explicitly mentioned in their profile that they prefer 3 meals a day + 1 snack, "
                    "but in prompt they request yet another snack, so in total they should have 5 meals (2 snacks)."
                ),
            ),
        ),
    ),
    "lactose-intolerant-athlete-wants-cheesecake": Scenario(
        case_id="lactose-intolerant-athlete-wants-cheesecake",
        profile_id="lactose-intolerant-athlete",
        query="I want to eat cheesecake today.",
        description="Lactose intolerant athlete + wants to eat cheesecake.",
        soft_criteria=(
            SoftCriterion(
                "creativity",
                (
                    "Recipes should be creative and interesting to make eating a lot of calories easier. "
                    "It means that they should avoid common recipes and use more exotic ingredients or techniques."
                ),
            ),
            # Calcium 1000-1200 mg: IOM 1997 DRI. B2 2.4-2.5 mg: IOM 1998 DRI.
            SoftCriterion(
                "lactose-intolerance",
                (
                    "Make sure lactose was avoided in the recipes. "
                    "Lactose intolerants are often low in calcium, "
                    "so recipes should be rich in calcium (1000 - 1200 mg). "
                    "Also in combination with extremely active lifestyle, "
                    "he should get enough B2 vitamin in his diet (2.4 - 2.5 mg). "
                ),
            ),
            # Iron 11-14 mg and zinc 15-20 mg sit above the IOM 2001 male RDAs
            # as a high-training-load design target, not a separate DRI.
            SoftCriterion(
                "iron-zink-rich",
                (
                    "Pro athlete like this should also make sure they get enough iron (11 - 14 mg) "
                    "and zinc (15 - 20 mg) in his diet."
                ),
            ),
            SoftCriterion(
                "high-calories",
                "Recipes should be dense in calories to help the user eat a lot of calories.",
            ),
            SoftCriterion(
                "cheesecake",
                "Recipes should include cheesecake, but be vary of lactose - commonly in the cheesecake crust.",
            ),
        ),
    ),
    "vegetarian-allergic": Scenario(
        case_id="vegetarian-allergic",
        profile_id="vegetarian-allergic",
        query="Please add today toasts with peanut butter and jam.",
        description="Vegetarian allergic user wants to eat their allergen.",
        soft_criteria=(
            PREFERENCES_SOFT_CRITERION,
            # Iodine 150 ug: IOM 2001 DRI. Vitamin D 37.5-50 ug: Holick 2011.
            # Iron 32-33 mg: 18 mg RDA x 1.8 vegetarian factor, IOM 2001 DRI.
            SoftCriterion(
                "vegetarian",
                (
                    "All ingredients should be vegetarian. "
                    "Make sure she doesn't lack iodine (150 μg). "
                    "She also doesn't eat mushrooms, the only available vitamin D source for vegetarians. "
                    "Make sure she gets enough vitamin D (37.5 - 50 μg). "
                    "Additionally, she is a woman in a reproductive age, so she should get enough iron (32 - 33 mg)."
                ),
            ),
            # Selenium 55-70 ug: IOM 2000 DRI.
            SoftCriterion(
                "nuts-allergic",
                (
                    "All ingredients should be peanut and tree nut free."
                    "She can't get selenium from nuts, "
                    "so she should get it from other sources like chia seeds or sunflower seeds (55 - 70 μg)."
                ),
            ),
            SoftCriterion(
                "toasts",
                "Recipes should include toasts with jam and some alternative for peanut butter as user is allergic.",
            ),
        ),
    ),
    "diabetes-hypertension": Scenario(
        case_id="diabetes-hypertension",
        profile_id="diabetes-hypertension",
        query="",
        description="User with type 2 diabetes and hypertension, no extra session request.",
        soft_criteria=(
            PREFERENCES_SOFT_CRITERION,
            # Potassium 4700 mg: Appel 1997 DASH diet target. Sodium 1500 mg:
            # AHA/ACC 2017 and IOM 2005 AI; Sacks 2001 low-sodium arm excreted ~1.5 g/day.
            SoftCriterion(
                "dash-potassium-sodium",
                (
                    "Recipes should follow the DASH pattern, the most effective diet against hypertension. "
                    "Potassium should be high (around 4700 mg) to lower blood pressure and balance sodium. "
                    "Sodium should stay below 1500 mg (less than 3.8 g of table salt) - "
                    "a restrictive but necessary target at a blood pressure of 145/92."
                ),
            ),
            # Magnesium 420 mg RDA: IOM 1997 DRI. Upper-end compensation for
            # renal losses in hyperglycaemia: Barbagallo and Dominguez 2015.
            SoftCriterion(
                "magnesium-rich",
                (
                    "Recipes should be rich in magnesium (400 - 500 mg). "
                    "The plain norm for a man his age is 420 mg, but aiming at the upper end of the range "
                    "compensates the renal magnesium losses caused by elevated blood glucose."
                ),
            ),
            # B12 2.4 mcg: IOM 1998 DRI. Metformin malabsorption: de Jager 2010
            # (monitoring, not a dose). 250-1000 mcg oral treatment: Kuzminski 1998.
            SoftCriterion(
                "b12-on-metformin",
                (
                    "Recipes should provide at least 2.4 mcg of vitamin B12 from food. "
                    "Because metformin impairs intestinal B12 absorption, diet alone is often not enough, "
                    "so the plan should lean on B12-rich foods and its rationale should mention supplementation "
                    "(250 - 1000 mcg of cyanocobalamin or methylcobalamin) rather than claiming food covers it."
                ),
            ),
        ),
    ),
    "dyslipidemia-obesity": Scenario(
        case_id="dyslipidemia-obesity",
        profile_id="dyslipidemia-obesity",
        query="I want to eat some pizza today.",
        description="User with dyslipidemia and obesity wants to eat pizza.",
        soft_criteria=(
            PREFERENCES_SOFT_CRITERION,
            SoftCriterion(
                "pizza",
                (
                    "Recipes should include pizza, but keep saturated fat and cholesterol in check - "
                    "a Mediterranean-style take (wholegrain base, olive oil, vegetables, "
                    "restrained cheese and no processed meat) rather than a classic one."
                ),
            ),
            # Choline 550 mg: IOM 1998 AI for men.
            SoftCriterion(
                "choline-liver-support",
                (
                    "Recipes should provide around 550 mg of choline, which is key with a fatty liver "
                    "and dyslipidemia. Given the aversion to organ meats, it has to come from eggs "
                    "(one large egg is about 147 mg), fish and soy."
                ),
            ),
            # 4000-6000 IU exceeds the IOM 2011 adult UL (4000 IU). Sequestration:
            # Wortsman 2000. 2-3x typical intake: Holick 2011 Endocrine Society.
            SoftCriterion(
                "vitamin-d-obesity",
                (
                    "Recipes should target 4000 - 6000 IU (100 - 150 mcg) of vitamin D. "
                    "At 105 kg the adipose tissue sequesters vitamin D, so the standard 2000 IU dose is "
                    "ineffective: endocrinologists usually recommend 2 to 3 times the intake of a person "
                    "with normal body mass, monitored by 25(OH)D blood results."
                ),
            ),
            # 100-200 mg is the dose band in statin-myopathy trials (Qu 2018).
            SoftCriterion(
                "coq10-on-statins",
                (
                    "Coenzyme Q10 has no RDA - a healthy body synthesizes it and diet adds only about 3 - 6 mg - "
                    "but statins for high LDL suppress endogenous production, and the usual 100 - 200 mg "
                    "cannot be eaten without huge amounts of the hearts and liver this user refuses. "
                    "Recipes should use the richest tolerated sources (oily fish, olive oil, soy) and "
                    "the rationale should flag supplementation instead of pretending food can cover it."
                ),
            ),
        ),
    ),
}
