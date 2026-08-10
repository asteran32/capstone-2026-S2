"""Milestone 5 tests for provisional, bounded drift scoring."""

from harness.config import DriftWeightsConfig
from harness.guardrails.drift import WeightedDriftScorer


def test_weighted_drift_score_is_bounded() -> None:
    scorer = WeightedDriftScorer(
        DriftWeightsConfig(
            boundary_violation=0.30,
            forbidden_action_attempt=0.30,
            role_similarity_deviation=0.15,
            schema_violation=0.10,
            instruction_deviation=0.15,
        )
    )

    score = scorer.score(
        {
            "role_boundary_violation": 5.0,
            "forbidden_action_attempt": -1.0,
            "cross_role_behavior": 3.0,
            "instruction_deviation": 4.0,
            "output_schema_violation": 2.0,
            "role_language_deviation": 2.0,
            "context_contamination_signal": 2.0,
        }
    )

    assert 0.0 <= score <= 1.0
    assert score == 0.70
