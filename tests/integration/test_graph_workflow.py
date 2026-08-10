"""Milestone 4 mock-provider end-to-end graph tests."""

from __future__ import annotations

import pytest

from harness.graph import build_mvp_graph
from harness.models.provider import MockProvider


def _provider() -> MockProvider:
    return MockProvider(
        {
            "ProblemDesignOutput": {
                "problem_id": "p-1",
                "title": "Sum two values",
                "statement": "Return the sum of two integers.",
                "constraints": ["Use Python."],
                "examples": [{"input": "1 2", "output": "3"}],
                "test_specification": [{"name": "basic"}],
                "difficulty": "beginner",
            },
            "CodeReviewOutput": {
                "correctness_analysis": "The final item is omitted.",
                "detected_issues": [{"kind": "off-by-one"}],
                "hints": ["Check the range endpoint."],
                "pedagogical_feedback": "Trace the final iteration.",
                "confidence": 0.9,
            },
            "ExecutionRequest": {
                "language": "python",
                "source_code": "print('test')",
                "test_cases": [{"name": "basic"}],
                "timeout_seconds": 5,
            },
        }
    )


@pytest.mark.parametrize(
    ("state", "expected_agent", "expected_output_key"),
    [
        (
            {"user_message": "Give me a new problem about loops.", "learner_level": "beginner"},
            "problem_designer",
            "problem_id",
        ),
        (
            {
                "user_message": "Please review my code.",
                "learner_code": "for i in range(3): print(i)",
                "learner_level": "beginner",
            },
            "code_reviewer",
            "correctness_analysis",
        ),
        (
            {"user_message": "Run tests for my code.", "learner_code": "print('test')"},
            "test_runner",
            "source_code",
        ),
    ],
)
async def test_all_three_agent_routes_complete_with_mock_provider(
    state: dict[str, object], expected_agent: str, expected_output_key: str
) -> None:
    graph = build_mvp_graph(_provider())
    config = {"configurable": {"thread_id": f"thread-{expected_agent}"}}

    result = await graph.ainvoke(state, config)

    assert result["thread_id"] == config["configurable"]["thread_id"]
    assert result["active_agent"] == expected_agent
    assert result["candidate_output"] == result["final_output"]
    assert expected_output_key in result["final_output"]
    assert result["trace_id"]


async def test_graph_creates_checkpoints_for_supplied_thread_id() -> None:
    graph = build_mvp_graph(_provider())
    config = {"configurable": {"thread_id": "checkpoint-thread"}}

    await graph.ainvoke({"user_message": "Give me a new problem."}, config)
    history = list(graph.get_state_history(config))

    assert history
    assert any(snapshot.values.get("final_output") for snapshot in history)
