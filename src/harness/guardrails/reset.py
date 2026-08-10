"""Selective Layer 3 reset policy and decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from harness.config import GuardrailsConfig


@dataclass(frozen=True)
class ResetDecision:
    """Whether a reset should occur and the evidence that triggered it."""

    required: bool
    reason: str | None = None


class ResetPolicy:
    """Applies bounded, configuration-driven Layer 3 reset conditions."""

    def __init__(self, config: GuardrailsConfig) -> None:
        self._enabled = config.layer3.enabled
        self._reset_threshold = config.thresholds.reset
        self._max_repairs = config.limits.max_repair_attempts_per_turn
        self._max_resets = config.limits.max_resets_per_turn

    @property
    def max_repairs(self) -> int:
        return self._max_repairs

    @property
    def max_resets(self) -> int:
        return self._max_resets

    def decide(self, state: Mapping[str, Any]) -> ResetDecision:
        """Determine whether Layer 3 reset is enabled and warranted."""

        if not self._enabled:
            return ResetDecision(required=False)
        indicators = state.get("drift_indicators", {})
        if float(state.get("drift_score", 0.0)) >= self._reset_threshold:
            return ResetDecision(required=True, reason="reset_threshold_exceeded")
        if float(indicators.get("forbidden_action_attempt", 0.0)) > 0.0:
            return ResetDecision(required=True, reason="forbidden_action_attempt")
        if self._repeated_boundary_violation(state):
            return ResetDecision(required=True, reason="repeated_boundary_violation")
        if int(state.get("repair_count", 0)) >= self._max_repairs:
            return ResetDecision(required=True, reason="repair_limit_exhausted")
        return ResetDecision(required=False)

    @staticmethod
    def _repeated_boundary_violation(state: Mapping[str, Any]) -> bool:
        observations = state.get("drift_history", [])
        count = sum(
            float(observation.get("indicators", {}).get("role_boundary_violation", 0.0))
            > 0.0
            for observation in observations
        )
        return count >= 2
