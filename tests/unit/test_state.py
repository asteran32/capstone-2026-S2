"""Tests for the M1 shared state contract."""

from typing import get_type_hints

from harness.graph.state import HarnessState


def test_harness_state_contract() -> None:
    hints = get_type_hints(HarnessState)

    assert HarnessState.__total__ is False
    assert {
        "session_id",
        "thread_id",
        "trace_id",
        "active_agent",
        "candidate_output",
        "candidate_output_id",
        "drift_indicators",
        "guardrail_action",
        "experiment_condition",
    }.issubset(hints)
