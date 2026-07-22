"""Map DuckDB per-100g columns to Totaller `NutrientName` keys.

Values in both food DBs are already stored in the canonical units from
`setup.units.TARGET_UNIT`; this map only renames columns into the Totaller's
enum (no scaling). Columns without a Totaller counterpart (`energy_kj_in_100g`)
are omitted.
"""

from __future__ import annotations

from dietary_advisor.totaller.nutrition import NutrientName

COLUMN_TO_NUTRIENT: dict[str, NutrientName] = {
    "energy_kcal_in_100g": NutrientName.ENERGY_KCAL,
    "proteins_g_in_100g": NutrientName.PROTEIN_G,
    "carbohydrates_g_in_100g": NutrientName.CARBS_G,
    "fat_g_in_100g": NutrientName.FAT_G,
    "saturated_fat_g_in_100g": NutrientName.SATURATED_FAT_G,
    "fiber_g_in_100g": NutrientName.FIBER_G,
    "sugars_g_in_100g": NutrientName.SUGAR_G,
    "salt_g_in_100g": NutrientName.SALT_G,
    "sodium_mg_in_100g": NutrientName.SODIUM_MG,
    "potassium_mg_in_100g": NutrientName.POTASSIUM_MG,
    "calcium_mg_in_100g": NutrientName.CALCIUM_MG,
    "iron_mg_in_100g": NutrientName.IRON_MG,
    "vitamin_c_mg_in_100g": NutrientName.VITAMIN_C_MG,
    "vitamin_d_ug_in_100g": NutrientName.VITAMIN_D_UG,
    "cholesterol_mg_in_100g": NutrientName.CHOLESTEROL_MG,
}


def nutrients_from_row(row: object) -> dict[NutrientName, float]:
    """Pull non-negative canonical nutrient amounts off a 1:1 read model or mapping."""
    get = row.get if isinstance(row, dict) else lambda k: getattr(row, k, None)
    out: dict[NutrientName, float] = {}
    for col, nutrient in COLUMN_TO_NUTRIENT.items():
        value = get(col)
        if value is None:
            continue
        amount = float(value)
        if amount >= 0:
            out[nutrient] = amount
    return out
