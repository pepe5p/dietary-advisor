"""Deterministic USDA shortlist prefetch to reduce LLM tool round-trips."""

from __future__ import annotations

import logging

from dietary_advisor.agents.deps import AgentDeps
from dietary_advisor.schemas.nutrition import FoodItem
from dietary_advisor.schemas.profile import DietPattern

log = logging.getLogger(__name__)

# Staple searches per diet pattern (fixed order for reproducibility).
_PREFETCH_QUERIES: dict[DietPattern, tuple[str, ...]] = {
    DietPattern.OMNIVORE: (
        "chicken breast",
        "brown rice",
        "broccoli",
        "olive oil",
        "greek yogurt",
    ),
    DietPattern.PESCATARIAN: (
        "salmon",
        "brown rice",
        "broccoli",
        "olive oil",
        "greek yogurt",
    ),
    DietPattern.VEGETARIAN: (
        "eggs",
        "brown rice",
        "broccoli",
        "cheddar cheese",
        "lentils cooked",
    ),
    DietPattern.VEGAN: (
        "tofu firm",
        "brown rice",
        "black beans",
        "broccoli",
        "olive oil",
    ),
    DietPattern.KETO: (
        "chicken breast",
        "eggs",
        "avocado",
        "spinach",
        "cheddar cheese",
    ),
    DietPattern.MEDITERRANEAN: (
        "salmon",
        "chickpeas",
        "olive oil",
        "spinach",
        "brown rice",
    ),
    DietPattern.DASH: (
        "chicken breast",
        "brown rice",
        "broccoli",
        "banana",
        "low fat milk",
    ),
    DietPattern.LOW_FODMAP: (
        "chicken breast",
        "brown rice",
        "carrots",
        "spinach",
        "lactose free milk",
    ),
}


def prefetch_usda_shortlist(deps: AgentDeps, *, page_size: int = 2) -> list[str]:
    """Populate ``deps.shortlist`` with staple USDA hits before the nutrition agent runs.

    Returns the names added so the prompt can tell the model what is already available.
    """
    if deps.usda is None:
        return []
    queries = _PREFETCH_QUERIES.get(deps.profile.diet_pattern, _PREFETCH_QUERIES[DietPattern.OMNIVORE])
    seen: set[tuple[int | None, str]] = {(f.fdc_id, f.name.lower()) for f in deps.shortlist}
    added: list[str] = []
    for query in queries:
        try:
            items = deps.usda.search(query, page_size=page_size)
        except Exception as exc:  # noqa: BLE001
            log.warning("USDA prefetch failed for %r: %s", query, exc)
            continue
        for item in items:
            key = (item.fdc_id, item.name.lower())
            if key in seen:
                continue
            seen.add(key)
            deps.shortlist.append(item)
            added.append(item.name)
    return added


def summarize_shortlist(shortlist: list[FoodItem], *, limit: int = 12) -> str:
    """Compact shortlist summary for the nutrition-agent prompt."""
    if not shortlist:
        return "No foods prefetched yet."
    lines: list[str] = []
    for item in shortlist[:limit]:
        lines.append(f"- {item.name} (fdc_id={item.fdc_id})")
    if len(shortlist) > limit:
        lines.append(f"- ... and {len(shortlist) - limit} more")
    return "\n".join(lines)
