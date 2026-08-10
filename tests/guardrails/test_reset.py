"""Milestone 6 tests for configuration-driven Layer 3 reset decisions."""

from harness.config import load_guardrails_config
from harness.guardrails.reset import ResetPolicy


def test_reset_policy_detects_forbidden_action_when_layer_three_enabled() -> None:
    decision = ResetPolicy(load_guardrails_config()).decide(
        {"drift_indicators": {"forbidden_action_attempt": 1.0}}
    )

    assert decision.required is True
    assert decision.reason == "forbidden_action_attempt"


def test_reset_policy_is_disabled_with_layer_three() -> None:
    config = load_guardrails_config()
    config.layer3.enabled = False

    assert ResetPolicy(config).decide({"drift_score": 1.0}).required is False
