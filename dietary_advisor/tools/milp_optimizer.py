"""MILP-based diet optimizer (the modern incarnation of the Stigler/Dantzig diet).

Given a candidate set of `FoodItem`s, target macros and a list of `HardConstraint`s,
this module returns the gram-amounts that minimise the L1 deviation from the
caloric/macro target while satisfying every hard constraint.

This is the "Totaller in optimisation mode": the agent first calls
:func:`shortlist_foods` (LLM-driven) and then offloads the numeric layout to
:func:`optimize_portions` (deterministic). Hybrid pattern straight out of
Adilmetova et al. (Kwerenda Literatury sec. "Systemy Hybrydowe MILP + LLM").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pulp

from dietary_advisor.schemas.constraints import HardConstraint
from dietary_advisor.schemas.meal_plan import Meal, MealKind, MealPlan, Portion, Recipe
from dietary_advisor.schemas.nutrition import FoodItem, MacroTargets, NutrientName

log = logging.getLogger(__name__)

# Bounds on per-food gram amount in a single optimisation run.
_MIN_GRAMS = 0.0
_MAX_GRAMS = 600.0
# L1 deviation weights per macronutrient (kcal weighted highest).
_DEFAULT_WEIGHTS: dict[NutrientName, float] = {
    NutrientName.ENERGY_KCAL: 1.0,
    NutrientName.PROTEIN_G: 5.0,
    NutrientName.CARBS_G: 1.0,
    NutrientName.FAT_G: 1.0,
    NutrientName.FIBER_G: 0.5,
}


@dataclass
class OptimisationResult:
    status: str
    objective: float
    grams: dict[str, float]  # food.name -> grams
    totals: dict[NutrientName, float]


def _expand_constraints(
    constraints: list[HardConstraint],
) -> tuple[list[HardConstraint], list[HardConstraint], list[HardConstraint]]:
    """Split constraints by kind for easier handling."""
    excl = [c for c in constraints if c.kind in {"allergen_exclusion", "ingredient_exclusion", "diet_pattern"}]
    max_n = [c for c in constraints if c.kind == "max_nutrient"]
    min_n = [c for c in constraints if c.kind == "min_nutrient"]
    return excl, max_n, min_n


def _filter_eligible_foods(
    foods: list[FoodItem],
    constraints: list[HardConstraint],
) -> list[FoodItem]:
    """Drop foods that are pre-emptively excluded by hard constraints."""
    excl, _, _ = _expand_constraints(constraints)
    eligible: list[FoodItem] = []
    for f in foods:
        ok = True
        for c in excl:
            if c.kind == "allergen_exclusion" and f.contains_allergen(c.target):
                ok = False
                break
            if c.kind == "ingredient_exclusion" and c.target in f.name.lower():
                ok = False
                break
            if c.kind == "diet_pattern":
                # We require the food to be tagged with the diet pattern.
                if c.target not in {t.lower() for t in f.tags}:
                    ok = False
                    break
        if ok:
            eligible.append(f)
    return eligible


def optimize_portions(
    foods: list[FoodItem],
    targets: MacroTargets,
    constraints: list[HardConstraint] | None = None,
    weights: dict[NutrientName, float] | None = None,
) -> OptimisationResult:
    """Solve the diet LP for the given candidate set.

    Returns an :class:`OptimisationResult`. If no feasible plan exists under
    the hard constraints, `status` will reflect PuLP's infeasibility code.
    """
    constraints = constraints or []
    weights = {**_DEFAULT_WEIGHTS, **(weights or {})}
    eligible = _filter_eligible_foods(foods, constraints)
    if not eligible:
        return OptimisationResult(status="NoEligibleFoods", objective=float("inf"), grams={}, totals={})

    # Decision variables: continuous grams per food (we use an LP, not MILP, for
    # speed; integrality is rarely meaningful for "grams of broccoli").
    prob = pulp.LpProblem("diet", pulp.LpMinimize)
    grams = {f.name: pulp.LpVariable(f"g_{i}", lowBound=_MIN_GRAMS, upBound=_MAX_GRAMS)
             for i, f in enumerate(eligible)}

    # Auxiliary variables for absolute deviation from target on each tracked macro.
    targets_d = targets.as_dict()
    dev_pos: dict[NutrientName, pulp.LpVariable] = {}
    dev_neg: dict[NutrientName, pulp.LpVariable] = {}
    for nutrient in targets_d:
        dev_pos[nutrient] = pulp.LpVariable(f"dp_{nutrient.value}", lowBound=0)
        dev_neg[nutrient] = pulp.LpVariable(f"dn_{nutrient.value}", lowBound=0)

    def total_for(nutrient: NutrientName) -> pulp.LpAffineExpression:
        return pulp.lpSum(
            (f.nutrients_per_100g.get(nutrient, 0.0) / 100.0) * grams[f.name] for f in eligible
        )

    # Tie totals to deviations: total - target = dp - dn.
    for nutrient, target_val in targets_d.items():
        prob += total_for(nutrient) - target_val == dev_pos[nutrient] - dev_neg[nutrient]

    # Hard nutrient bounds.
    _, max_n, min_n = _expand_constraints(constraints)
    for c in max_n:
        nutrient = NutrientName(c.target)
        if c.value is None:
            continue
        prob += total_for(nutrient) <= c.value, f"max_{c.target}"
    for c in min_n:
        nutrient = NutrientName(c.target)
        if c.value is None:
            continue
        prob += total_for(nutrient) >= c.value, f"min_{c.target}"

    # Objective: weighted L1 deviation across tracked macros.
    prob += pulp.lpSum(
        weights.get(n, 1.0) * (dev_pos[n] + dev_neg[n]) / max(targets_d[n], 1.0)
        for n in targets_d
    )

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return OptimisationResult(status=status, objective=float("inf"), grams={}, totals={})

    grams_out = {name: round(float(var.value() or 0.0), 1) for name, var in grams.items()}
    totals_out: dict[NutrientName, float] = {}
    for nutrient in targets_d:
        totals_out[nutrient] = round(
            sum(
                (f.nutrients_per_100g.get(nutrient, 0.0) / 100.0) * grams_out[f.name]
                for f in eligible
            ),
            2,
        )
    return OptimisationResult(
        status=status,
        objective=round(float(pulp.value(prob.objective) or 0.0), 4),
        grams=grams_out,
        totals=totals_out,
    )


def build_meal_plan_from_optimization(
    user_id: str,
    foods: list[FoodItem],
    result: OptimisationResult,
    *,
    rationale: str = "",
) -> MealPlan:
    """Pack an optimisation result into a single-meal `MealPlan`.

    The shape is deliberately simple - the LLM is then expected to *reorganise*
    these portions into breakfast/lunch/dinner. The MILP solves the numeric
    layout; the LLM solves the cultural / palatability layout.
    """
    portions: list[Portion] = []
    for food in foods:
        g = result.grams.get(food.name, 0.0)
        if g <= 0.5:
            continue
        portions.append(Portion(food=food, grams=g))
    if not portions:
        # Construct a minimum-viable empty meal so callers don't crash.
        raise ValueError("Optimization produced no positive portions; cannot build a MealPlan.")

    recipe = Recipe(name="Optimized daily meal", portions=portions)
    meal = Meal(kind=MealKind.LUNCH, recipe=recipe)
    return MealPlan(user_id=user_id, meals=[meal], rationale=rationale)
