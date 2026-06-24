"""Ablation variant configuration helpers.

Builds the leave-one-out grid of `VariantConfig`s used by the evaluation
harness: the full system plus one config per disabled module.
"""

from __future__ import annotations

from dietary_advisor.pipeline import VariantConfig


def leave_one_out_variants() -> list[VariantConfig]:
    return [
        VariantConfig(),
        VariantConfig(food_enabled=False),
        VariantConfig(totaller_enabled=False),
        VariantConfig(rag_enabled=False),
        VariantConfig(reflection_enabled=False),
    ]
