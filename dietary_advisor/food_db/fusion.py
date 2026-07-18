"""Reciprocal rank fusion, shared by the OFF and USDA hybrid-search readers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# Reciprocal-rank-fusion damping constant. 60 is the value from the original
# RRF paper and is the de-facto default; it keeps any single top rank from
# dominating so the two channels genuinely blend.
_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[list[dict[str, Any]]],
    key: Callable[[dict[str, Any]], Any],
    k: int = _RRF_K,
) -> list[dict[str, Any]]:
    """Fuse several ranked result lists into one via reciprocal rank fusion.

    Each row contributes `1 / (k + rank)` to its key's score, so a record
    surfaced by both channels outranks one that scores highly in only one.
    Preserves the first row seen per key (column sets are identical here).
    """
    scores: dict[Any, float] = {}
    rows: dict[Any, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, row in enumerate(ranked, start=1):
            identity = key(row)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
            rows.setdefault(identity, row)
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [rows[i] for i in ordered]
