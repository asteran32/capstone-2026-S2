"""Provisional, replaceable drift-scoring strategies for Layer 2."""

from __future__ import annotations

from typing import Mapping, Protocol

from harness.config import DriftWeightsConfig


class DriftScorer(Protocol):
    """Scores raw consistency indicators without depending on agent implementations."""

    def score(self, indicators: Mapping[str, float]) -> float:
        """Return a normalized drift score in the inclusive range [0.0, 1.0]."""


class WeightedDriftScorer:
    """Initial deterministic scorer using provisional engineering weights."""

    def __init__(self, weights: DriftWeightsConfig) -> None:
        self._weights = weights

    def score(self, indicators: Mapping[str, float]) -> float:
        boundary = max(
            _indicator(indicators, "role_boundary_violation"),
            _indicator(indicators, "cross_role_behavior"),
        )
        similarity = max(
            _indicator(indicators, "cross_role_behavior"),
            _indicator(indicators, "role_language_deviation"),
        )
        instruction = max(
            _indicator(indicators, "instruction_deviation"),
            _indicator(indicators, "context_contamination_signal"),
        )
        score = (
            self._weights.boundary_violation * boundary
            + self._weights.forbidden_action_attempt
            * _indicator(indicators, "forbidden_action_attempt")
            + self._weights.role_similarity_deviation * similarity
            + self._weights.schema_violation
            * _indicator(indicators, "output_schema_violation")
            + self._weights.instruction_deviation * instruction
        )
        return min(1.0, max(0.0, score))


def _indicator(indicators: Mapping[str, float], name: str) -> float:
    return min(1.0, max(0.0, float(indicators.get(name, 0.0))))
