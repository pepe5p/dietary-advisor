"""Unit-conversion helpers used by both food-DB builds."""

from __future__ import annotations

import duckdb

from setup.units import register_unit_ratio_macro, TARGET_UNIT, unit_ratio


def test_mass_ratios() -> None:
    assert unit_ratio("g", "mg") == 1000.0
    assert unit_ratio("g", "ug") == 1_000_000.0
    assert unit_ratio("mg", "g") == 0.001
    assert unit_ratio("MG", "mg") == 1.0
    assert unit_ratio(None, "mg") == 1.0
    assert unit_ratio("kcal", "kcal") == 1.0


def test_duckdb_macro_matches_python() -> None:
    con = duckdb.connect()
    try:
        register_unit_ratio_macro(con)
        for src, dst, expected in (
            ("g", "mg", 1000.0),
            ("G", "mg", 1000.0),
            ("g", "ug", 1_000_000.0),
            ("MG", "mg", 1.0),
            (None, "mg", 1.0),
        ):
            got = con.execute("SELECT unit_ratio(?, ?)", [src, dst]).fetchone()[0]  # type: ignore[index]
            assert got == expected == unit_ratio(src, dst)
    finally:
        con.close()


def test_target_units_cover_known_prefixes() -> None:
    assert TARGET_UNIT["sodium"] == "mg"
    assert TARGET_UNIT["vitamin_d"] == "ug"
    assert TARGET_UNIT["proteins"] == "g"
    assert TARGET_UNIT["energy_kcal"] == "kcal"
