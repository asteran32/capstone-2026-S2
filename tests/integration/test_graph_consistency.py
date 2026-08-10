"""Milestone 5 graph tests for Layer 2 monitoring and raw drift retention."""

from __future__ import annotations

from harness.config import load_guardrails_config
from harness.graph import build_mvp_graph
from harness.models.provider import MockProvider


def _provider(review_text: str = "The loop skips the final item.") -> MockProvider:
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
                "correctness_analysis": review_text,
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


async def test_layer_two_monitors_each_agent_route_and_retains_raw_observation() -> None:
    graph = build_mvp_graph(_provider())

    for thread_id, state in (
        ("m5-problem", {"user_message": "Give me a new problem."}),
        ("m5-review", {"user_message": "Review my code.", "learner_code": "print(1)"}),
        ("m5-test", {"user_message": "Run tests.", "learner_code": "print(1)"}),
    ):
        result = await graph.ainvoke(state, {"configurable": {"thread_id": thread_id}})
        assert result["consistency_result"] is not None
        assert result["drift_history"]
        assert result["candidate_output"] == result["final_output"]


async def test_forbidden_injection_preserves_candidate_and_marks_repair() -> None:
    graph = build_mvp_graph(_provider("I will execute_code the learner program."))

    result = await graph.ainvoke(
        {"user_message": "Review my code.", "learner_code": "print(1)"},
        {"configurable": {"thread_id": "m5-injection"}},
    )

    assert result["guardrail_action"] == "repair"
    assert result["candidate_output"] == result["final_output"]
    assert result["drift_history"][-1]["indicators"]["forbidden_action_attempt"] == 1.0


async def test_disabling_layer_two_keeps_graph_topology_and_passes_candidate() -> None:
    enabled_graph = build_mvp_graph(_provider())
    disabled_config = load_guardrails_config()
    disabled_config.layer2.enabled = False
    disabled_graph = build_mvp_graph(_provider("I will execute_code the learner program."), guardrails=disabled_config)

    assert set(enabled_graph.get_graph().nodes) == set(disabled_graph.get_graph().nodes)
    result = await disabled_graph.ainvoke(
        {"user_message": "Review my code.", "learner_code": "print(1)"},
        {"configurable": {"thread_id": "m5-disabled"}},
    )
    assert result["guardrail_action"] == "pass"
    assert result["candidate_output"] == result["final_output"]
