"""Milestone 7 safety-validator separation tests."""

from harness.config import load_safety_config
from harness.safety.validator import SafetyValidator


def test_safety_validator_passes_non_execution_candidates_without_drift_labels() -> None:
    result = SafetyValidator(load_safety_config()).validate(
        "code_reviewer", {"correctness_analysis": "Use a smaller loop bound."}
    )

    assert result.allowed is True
    assert result.violations == []
    assert result.execution_result is None


def test_safety_validator_blocks_unsafe_execution_without_role_evaluation() -> None:
    result = SafetyValidator(load_safety_config()).validate(
        "test_runner",
        {
            "language": "python",
            "source_code": "import socket",
            "test_cases": [],
            "timeout_seconds": 1,
        },
    )

    assert result.allowed is False
    assert result.execution_result is not None
    assert result.execution_result.status == "blocked"
    assert "blocked_import:socket" in result.violations
