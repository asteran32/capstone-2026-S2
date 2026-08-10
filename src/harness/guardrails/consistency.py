"""Deterministic Layer 2 role-consistency evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from harness.config import GuardrailsConfig
from harness.guardrails.contracts import AgentId, RoleContractLoader
from harness.guardrails.drift import DriftScorer, WeightedDriftScorer
from harness.models.schemas import (
    CodeReviewOutput,
    ConsistencyResult,
    ExecutionRequest,
    ProblemDesignOutput,
)


EVALUATOR_VERSION = "0.1-provisional"
_OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "ProblemDesignOutput": ProblemDesignOutput,
    "CodeReviewOutput": CodeReviewOutput,
    "ExecutionRequest": ExecutionRequest,
}
_INDICATOR_NAMES = (
    "role_boundary_violation",
    "forbidden_action_attempt",
    "cross_role_behavior",
    "instruction_deviation",
    "output_schema_violation",
    "role_language_deviation",
    "context_contamination_signal",
)


class ConsistencyMonitor:
    """Evaluates one candidate against its role contract using deterministic heuristics."""

    def __init__(
        self,
        config: GuardrailsConfig,
        *,
        contract_loader: RoleContractLoader | None = None,
        scorer: DriftScorer | None = None,
    ) -> None:
        self._enabled = config.layer2.enabled
        self._repair_threshold = config.thresholds.repair
        self._contract_loader = contract_loader or RoleContractLoader()
        self._scorer = scorer or WeightedDriftScorer(config.drift_score.weights)

    def evaluate(
        self, agent_id: AgentId, candidate_output: Mapping[str, Any] | None
    ) -> ConsistencyResult:
        """Return a structured, non-mutating consistency assessment."""

        if not self._enabled:
            return _result(
                valid=True,
                indicators=_empty_indicators(),
                violations=[],
                schema_errors=[],
                score=0.0,
                action="pass",
            )

        contract = self._contract_loader.load(agent_id)
        candidate = {} if candidate_output is None else dict(candidate_output)
        indicators = _empty_indicators()
        schema_errors = _schema_errors(contract.output.schema_name, candidate)
        if schema_errors:
            indicators["output_schema_violation"] = 1.0

        rendered = json.dumps(candidate, sort_keys=True).lower()
        forbidden = _matching_actions(rendered, contract.forbidden_actions)
        if forbidden:
            indicators["forbidden_action_attempt"] = 1.0

        cross_role_actions = _cross_role_actions(agent_id, rendered, self._contract_loader)
        if cross_role_actions:
            indicators["role_boundary_violation"] = 1.0
            indicators["cross_role_behavior"] = 1.0
            indicators["role_language_deviation"] = 1.0

        if "ignore role contract" in rendered or "ignore instructions" in rendered:
            indicators["instruction_deviation"] = 1.0
        if any(marker in rendered for marker in ("internal reasoning", "candidate_output")):
            indicators["context_contamination_signal"] = 1.0

        violations = [f"forbidden_action:{action}" for action in forbidden]
        violations.extend(f"cross_role_action:{action}" for action in cross_role_actions)
        score = self._scorer.score(indicators)
        action = "repair" if score >= self._repair_threshold else "pass"
        return _result(
            valid=not violations and not schema_errors,
            indicators=indicators,
            violations=violations,
            schema_errors=schema_errors,
            score=score,
            action=action,
        )


def _result(
    *,
    valid: bool,
    indicators: dict[str, float],
    violations: list[str],
    schema_errors: list[str],
    score: float,
    action: str,
) -> ConsistencyResult:
    return ConsistencyResult(
        valid=valid,
        role_adherence_score=1.0 - score,
        violations=violations,
        drift_signals=indicators,
        recommended_action=action,
        evaluator_version=EVALUATOR_VERSION,
        schema_errors=schema_errors,
    )


def _empty_indicators() -> dict[str, float]:
    return {name: 0.0 for name in _INDICATOR_NAMES}


def _schema_errors(schema_name: str, candidate: dict[str, Any]) -> list[str]:
    schema = _OUTPUT_SCHEMAS.get(schema_name)
    if schema is None:
        return [f"unsupported_output_schema:{schema_name}"]
    try:
        schema.model_validate(candidate)
    except ValidationError as error:
        return [str(entry["type"]) for entry in error.errors()]
    return []


def _matching_actions(rendered: str, actions: list[str]) -> list[str]:
    return [action for action in actions if action.lower() in rendered]


def _cross_role_actions(
    agent_id: AgentId, rendered: str, contract_loader: RoleContractLoader
) -> list[str]:
    actions: set[str] = set()
    for other_agent in ("problem_designer", "code_reviewer", "test_runner"):
        if other_agent != agent_id:
            actions.update(contract_loader.load(other_agent).allowed_actions)
    return sorted(action for action in actions if action.lower() in rendered)
