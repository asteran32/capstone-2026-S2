"""Bounded learner-intent classification and deterministic agent routing."""

from __future__ import annotations

from typing import Literal

from harness.graph.state import AgentName, HarnessState


Intent = Literal[
    "new_problem",
    "modify_problem",
    "review_code",
    "request_hint",
    "run_tests",
    "interpret_tests",
    "unknown",
]


def classify_intent(state: HarnessState) -> Intent:
    """Classify a learner request into the fixed M4 intent vocabulary."""

    message = state.get("user_message", "").lower()
    if "test result" in message and any(
        verb in message for verb in ("interpret", "explain", "understand")
    ):
        return "interpret_tests"
    if any(phrase in message for phrase in ("run test", "run my test", "test my code")):
        return "run_tests"
    if "problem" in message and any(
        verb in message for verb in ("modify", "revise", "change")
    ):
        return "modify_problem"
    if any(phrase in message for phrase in ("new problem", "create problem", "give me a problem")):
        return "new_problem"
    if "hint" in message:
        return "request_hint"
    if state.get("learner_code") is not None or any(
        phrase in message for phrase in ("review", "my code", "bug", "error")
    ):
        return "review_code"
    return "unknown"


def route_for_intent(intent: str) -> AgentName:
    """Map a bounded intent to one of exactly three role agents."""

    if intent in {"new_problem", "modify_problem", "unknown"}:
        return "problem_designer"
    if intent in {"review_code", "request_hint"}:
        return "code_reviewer"
    if intent in {"run_tests", "interpret_tests"}:
        return "test_runner"
    raise ValueError(f"Unsupported routing intent: {intent}")
