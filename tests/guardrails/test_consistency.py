"""Milestone 5 tests for deterministic Layer 2 consistency checks."""

from __future__ import annotations

from harness.config import load_guardrails_config
from harness.guardrails.consistency import EVALUATOR_VERSION, ConsistencyMonitor


def _review_output() -> dict[str, object]:
    return {
        "correctness_analysis": "The loop skips the final element.",
        "detected_issues": [{"kind": "off-by-one"}],
        "hints": ["Check the range endpoint."],
        "pedagogical_feedback": "Trace the final iteration.",
        "confidence": 0.9,
    }


def test_valid_candidate_passes_with_all_raw_indicators() -> None:
    result = ConsistencyMonitor(load_guardrails_config()).evaluate(
        "code_reviewer", _review_output()
    )

    assert result.valid is True
    assert result.recommended_action == "pass"
    assert result.evaluator_version == EVALUATOR_VERSION
    assert set(result.drift_signals) == {
        "role_boundary_violation",
        "forbidden_action_attempt",
        "cross_role_behavior",
        "instruction_deviation",
        "output_schema_violation",
        "role_language_deviation",
        "context_contamination_signal",
    }


def test_forbidden_action_injection_is_detected() -> None:
    output = _review_output()
    output["correctness_analysis"] = "I will execute_code the learner program."

    result = ConsistencyMonitor(load_guardrails_config()).evaluate("code_reviewer", output)

    assert result.valid is False
    assert result.drift_signals["forbidden_action_attempt"] == 1.0
    assert "forbidden_action:execute_code" in result.violations
    assert result.recommended_action == "repair"


def test_schema_failure_is_separate_from_role_drift() -> None:
    result = ConsistencyMonitor(load_guardrails_config()).evaluate(
        "code_reviewer", {"confidence": "not-a-number"}
    )

    assert result.valid is False
    assert result.schema_errors
    assert result.drift_signals["output_schema_violation"] == 1.0
    assert not result.violations


def test_disabled_layer_two_does_not_intervene() -> None:
    config = load_guardrails_config()
    config.layer2.enabled = False
    output = _review_output()
    output["correctness_analysis"] = "I will execute_code the learner program."

    result = ConsistencyMonitor(config).evaluate("code_reviewer", output)

    assert result.valid is True
    assert result.recommended_action == "pass"
    assert all(value == 0.0 for value in result.drift_signals.values())
