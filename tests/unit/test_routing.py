"""Milestone 4 tests for bounded intent classification and routing."""

import pytest

from harness.graph.routing import classify_intent, route_for_intent


@pytest.mark.parametrize(
    ("state", "intent"),
    [
        ({"user_message": "Give me a new problem about lists."}, "new_problem"),
        ({"user_message": "Revise this problem."}, "modify_problem"),
        ({"user_message": "Can you give me a hint?"}, "request_hint"),
        ({"user_message": "Run tests for this code."}, "run_tests"),
        ({"user_message": "Explain these test results."}, "interpret_tests"),
        ({"user_message": "", "learner_code": "print(1)"}, "review_code"),
        ({"user_message": "Hello"}, "unknown"),
        ({"user_message": "초급 Python 문제를 만들어 줘."}, "new_problem"),
        ({"user_message": "현재 문제를 수정해 줘."}, "modify_problem"),
        ({"user_message": "힌트를 줘."}, "request_hint"),
        (
            {"user_message": "이 코드를 테스트해 줘.", "learner_code": "print(1)"},
            "run_tests",
        ),
        ({"user_message": "테스트 결과를 설명해 줘."}, "interpret_tests"),
        (
            {"user_message": "이 코드를 검토해 줘.", "learner_code": "print(1)"},
            "review_code",
        ),
    ],
)
def test_intent_classification_is_bounded(state: dict[str, object], intent: str) -> None:
    assert classify_intent(state) == intent


@pytest.mark.parametrize(
    ("intent", "agent"),
    [
        ("new_problem", "problem_designer"),
        ("modify_problem", "problem_designer"),
        ("review_code", "code_reviewer"),
        ("request_hint", "code_reviewer"),
        ("run_tests", "test_runner"),
        ("interpret_tests", "test_runner"),
        ("unknown", "problem_designer"),
    ],
)
def test_all_intents_route_to_exactly_one_role_agent(intent: str, agent: str) -> None:
    assert route_for_intent(intent) == agent


def test_unknown_intent_is_deterministically_routed() -> None:
    assert route_for_intent(classify_intent({"user_message": "Hello"})) == "problem_designer"
