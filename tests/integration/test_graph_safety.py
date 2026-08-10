"""Milestone 7 graph integration tests for independent safety validation."""

from harness.graph import build_mvp_graph
from harness.models.provider import MockProvider


def _provider(source_code: str) -> MockProvider:
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
                "source_code": source_code,
                "test_cases": [{"name": "basic"}],
                "timeout_seconds": 1,
            },
        }
    )


async def test_safety_blocks_unsafe_test_request_without_marking_role_drift() -> None:
    graph = build_mvp_graph(_provider("import socket"))

    result = await graph.ainvoke(
        {"user_message": "Run tests for my code."},
        {"configurable": {"thread_id": "m7-unsafe"}},
    )

    assert result["consistency_result"]["recommended_action"] == "pass"
    assert result["drift_indicators"]["forbidden_action_attempt"] == 0.0
    assert result["safety_result"]["allowed"] is False
    assert "blocked_import:socket" in result["safety_result"]["violations"]
    assert result["final_output"] is None


async def test_safety_executes_approved_test_request() -> None:
    graph = build_mvp_graph(_provider("print('approved')"))

    result = await graph.ainvoke(
        {"user_message": "Run tests for my code."},
        {"configurable": {"thread_id": "m7-approved"}},
    )

    assert result["safety_result"]["allowed"] is True
    assert result["safety_result"]["execution_result"]["status"] == "success"
    assert result["latest_test_results"][0]["passed"] is True
    assert result["candidate_output"] == result["final_output"]
