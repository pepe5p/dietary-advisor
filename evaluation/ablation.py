"""Ablation variant configuration helpers.

Wraps `dietary_advisor.pipeline.VARIANTS` so the evaluation harness can deal
in lists/strings without importing the orchestrator internals.
"""

from __future__ import annotations

from dietary_advisor.pipeline import VariantConfig, VARIANTS


def variants_from_ids(ids: list[str]) -> list[VariantConfig]:
    """Resolve variant ids (e.g. ['V0', 'V2', 'V4']) to `VariantConfig`s."""
    out: list[VariantConfig] = []
    for vid in ids:
        if vid not in VARIANTS:
            raise ValueError(f"Unknown variant {vid!r}. Known: {sorted(VARIANTS)}")
        out.append(VARIANTS[vid])
    return out


def all_variants() -> list[VariantConfig]:
    return list(VARIANTS.values())
