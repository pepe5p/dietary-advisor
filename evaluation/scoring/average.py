"""Average judge-repetition score records into one mean verdict."""

from __future__ import annotations

from evaluation.scoring.store import ScoreRecord
from evaluation.validation.qualitative import CriterionScore, QualitativeResult


def _group_by_criterion(records: list[ScoreRecord]) -> dict[str, list[CriterionScore]]:
    """Collect each criterion's per-rep scores, keyed in first-seen order.

    The judge validator guarantees identical criterion ids across reps; this
    grouping is defensive so averaging never raises on a stray stored record.
    """
    grouped: dict[str, list[CriterionScore]] = {}
    for record in records:
        for score in record.qualitative.scores:
            grouped.setdefault(score.criterion_id, []).append(score)
    return grouped


def average_score_records(records: list[ScoreRecord]) -> ScoreRecord:
    if not records:
        raise ValueError("average_score_records requires at least one record")
    if len(records) == 1:
        return records[0]

    first = records[0]
    latest = max(records, key=lambda record: record.scored_at)

    averaged_scores: list[CriterionScore] = []
    for criterion_id, by_rep in _group_by_criterion(records).items():
        mean_score = sum(score.score for score in by_rep) / len(by_rep)
        closest = min(by_rep, key=lambda score: abs(score.score - mean_score))
        averaged_scores.append(
            CriterionScore(
                criterion_id=criterion_id,
                score=round(mean_score, 4),
                reasoning=closest.reasoning,
            ),
        )

    soft_agg = round(sum(record.qualitative.aggregate for record in records) / len(records), 4)
    safety = round(sum(record.qualitative.safety_adherence for record in records) / len(records), 4)

    seen_violations: set[str] = set()
    violations: list[str] = []
    for record in records:
        for violation in record.qualitative.safety_violations:
            if violation not in seen_violations:
                seen_violations.add(violation)
                violations.append(violation)

    return first.model_copy(
        update={
            "qualitative": QualitativeResult(
                scores=averaged_scores,
                aggregate=soft_agg,
                safety_adherence=safety,
                safety_violations=violations,
            ),
            "scored_at": latest.scored_at,
        },
    )
