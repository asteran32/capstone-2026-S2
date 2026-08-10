"""Milestone 2 tests for role-specific context projection."""

from __future__ import annotations

import pytest

from harness.memory.context import AgentContextMemory
from harness.memory.projector import ContextProjector


@pytest.fixture
def shared_state() -> dict[str, object]:
    return {
        "learner_level": "beginner",
        "user_message": "Please help with my loop.",
        "learner_code": "for index in range(3): print(index)",
        "current_problem": {"id": "p-1", "title": "Loop practice"},
        "latest_test_results": [{"name": "basic", "passed": False}],
        "candidate_output": {"private": "other agent output"},
        "final_output": {"private": "delivered output"},
        "consistency_result": {"private": "guardrail result"},
        "drift_indicators": {"private": 1.0},
    }


def test_context_projector_isolates_roles(shared_state: dict[str, object]) -> None:
    projector = ContextProjector()

    designer = projector.project(shared_state, "problem_designer")
    reviewer = projector.project(shared_state, "code_reviewer")
    runner = projector.project(shared_state, "test_runner")

    assert "learner_code" not in designer.task
    assert "latest_test_results" not in designer.task
    assert reviewer.task["learner_code"] == shared_state["learner_code"]
    assert runner.task["latest_test_results"] == shared_state["latest_test_results"]
    for projection in (designer, reviewer, runner):
        assert "candidate_output" not in projection.task
        assert "final_output" not in projection.task
        assert "consistency_result" not in projection.task
        assert "drift_indicators" not in projection.task


def test_context_projector_exposes_only_same_role_history(
    shared_state: dict[str, object],
) -> None:
    projection = ContextProjector().project(
        shared_state,
        "code_reviewer",
        agent_context=AgentContextMemory(
            agent_id="code_reviewer", history=[{"feedback": "Check indices."}]
        ),
    )

    assert projection.agent_context == [{"feedback": "Check indices."}]


def test_context_projector_rejects_cross_role_history(
    shared_state: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="owning agent"):
        ContextProjector().project(
            shared_state,
            "test_runner",
            agent_context=AgentContextMemory(
                agent_id="code_reviewer", history=[{"feedback": "private"}]
            ),
        )
