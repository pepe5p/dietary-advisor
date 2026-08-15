"""Aggregate per-run score records into variant-level summaries."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from evaluation.scoring.store import ScoreRecord


class VariantSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    llm_model: str
    variant: str
    n_runs: int
    mae_pct: float
    mse_pct: float
    soft_aggregate: float
    safety_adherence: float
    iterations: float
    elapsed_s: float
    n_safety_violations: int


def summarize(records: list[ScoreRecord]) -> list[VariantSummary]:
    groups: dict[tuple[str, str], list[ScoreRecord]] = {}
    for record in records:
        key = (record.llm_model, record.variant)
        groups.setdefault(key, []).append(record)

    summaries: list[VariantSummary] = []
    for (llm_model, variant), group in sorted(groups.items()):
        summaries.append(
            VariantSummary(
                llm_model=llm_model,
                variant=variant,
                n_runs=len(group),
                mae_pct=round(_mean([r.mae_pct for r in group]), 2),
                mse_pct=round(_mean([r.mse_pct for r in group]), 2),
                soft_aggregate=round(_mean([r.qualitative.aggregate for r in group]), 4),
                safety_adherence=round(_mean([r.qualitative.safety_adherence for r in group]), 4),
                iterations=round(_mean([float(r.iterations) for r in group]), 2),
                elapsed_s=round(_mean([r.elapsed_s for r in group]), 2),
                n_safety_violations=sum(len(r.qualitative.safety_violations) for r in group),
            ),
        )
    return summaries


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
