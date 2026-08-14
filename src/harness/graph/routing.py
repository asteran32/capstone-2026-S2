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
    if any(result in message for result in ("test result", "테스트 결과")) and any(
        verb in message
        for verb in ("interpret", "explain", "understand", "해석", "설명", "이해")
    ):
        return "interpret_tests"
    if any(
        phrase in message
        for phrase in (
            "run test",
            "run my test",
            "test my code",
            "테스트해",
            "테스트 해",
            "테스트 실행",
            "테스트 돌려",
        )
    ):
        return "run_tests"
    if any(noun in message for noun in ("problem", "문제")) and any(
        verb in message for verb in ("modify", "revise", "change", "수정", "변경", "바꿔")
    ):
        return "modify_problem"
    if any(
        phrase in message
        for phrase in (
            "new problem",
            "create problem",
            "give me a problem",
            "문제 만들어",
            "문제를 만들어",
            "문제 생성",
            "문제를 내",
        )
    ):
        return "new_problem"
    if any(phrase in message for phrase in ("hint", "힌트")):
        return "request_hint"
    if any(
        phrase in message
        for phrase in (
            "review",
            "my code",
            "bug",
            "error",
            "코드 검토",
            "코드를 검토",
            "버그",
            "오류",
        )
    ):
        return "review_code"
    if state.get("learner_code") is not None:
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
