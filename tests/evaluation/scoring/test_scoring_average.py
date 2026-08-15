"""Tests for averaging judge-repetition score records."""

from __future__ import annotations

from evaluation.scoring.average import average_score_records
from evaluation.scoring.store import ScoreRecord
from evaluation.validation.qualitative import CriterionScore, QualitativeResult


def _record(
    *,
    soft: float,
    safety: float = 1.0,
    reasoning: str = "Reason.",
    violations: list[str] | None = None,
    criteria: dict[str, float] | None = None,
) -> ScoreRecord:
    scored = criteria if criteria is not None else {"recipe-makes-sense": soft}
    return ScoreRecord(
        llm_model="test-model",
        variant="baseline",
        scenario_id="regular",
        judge_model="test-judge",
        mae_pct=10.0,
        mse_pct=100.0,
        per_nutrient_pct={"energy_kcal": 10.0},
        qualitative=QualitativeResult(
            scores=[
                CriterionScore(criterion_id=criterion_id, score=score, reasoning=reasoning)
                for criterion_id, score in scored.items()
            ],
            aggregate=soft,
            safety_adherence=safety,
            safety_violations=violations or [],
        ),
        iterations=1,
        elapsed_s=2.0,
    )


def test_average_score_records_means_soft_and_safety() -> None:
    averaged = average_score_records([_record(soft=0.8), _record(soft=1.0, safety=0.5)])
    assert averaged.qualitative.aggregate == 0.9
    assert averaged.qualitative.safety_adherence == 0.75


def test_average_score_records_keeps_reasoning_from_closest_rep() -> None:
    averaged = average_score_records(
        [
            _record(soft=0.8, reasoning="Closer."),
            _record(soft=1.0, reasoning="Farther."),
        ],
    )
    assert averaged.qualitative.scores[0].score == 0.9
    assert averaged.qualitative.scores[0].reasoning == "Closer."


def test_average_score_records_groups_by_criterion_defensively() -> None:
    """Grouping by id is defensive; the judge validator guarantees identical sets."""
    averaged = average_score_records(
        [
            _record(soft=0.8, criteria={"recipe-makes-sense": 0.8, "seasonal": 0.6}),
            _record(soft=1.0, criteria={"recipe-makes-sense": 1.0}),
        ],
    )
    by_id = {score.criterion_id: score.score for score in averaged.qualitative.scores}
    assert by_id == {"recipe-makes-sense": 0.9, "seasonal": 0.6}


def test_average_score_records_unions_violations() -> None:
    averaged = average_score_records(
        [
            _record(soft=0.9, violations=["milk in vegan plan"]),
            _record(soft=0.9, violations=["peanut allergen"]),
        ],
    )
    assert averaged.qualitative.safety_violations == ["milk in vegan plan", "peanut allergen"]
