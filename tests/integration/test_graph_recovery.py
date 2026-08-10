"""Milestone 6 graph tests for bounded repair and selective reset recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.config import load_guardrails_config
from harness.graph import build_mvp_graph
from harness.models.provider import GenerationResult


def _invalid_review() -> dict[str, object]:
    return {
        "correctness_analysis": "I will execute_code the learner program.",
        "detected_issues": [],
        "hints": [],
        "pedagogical_feedback": "",
        "confidence": 0.9,
    }


def _valid_review() -> dict[str, object]:
    return {
        "correctness_analysis": "The final loop item is omitted.",
        "detected_issues": [{"kind": "off-by-one"}],
        "hints": ["Check the range endpoint."],
        "pedagogical_feedback": "Trace the final iteration.",
        "confidence": 0.9,
    }


@dataclass
class _SequenceProvider:
    outputs: list[dict[str, object]]
    calls: int = field(default=0, init=False)

    async def generate(self, *_: Any, **__: Any) -> GenerationResult:
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return GenerationResult(output=output)


async def test_repaired_output_is_revalidated_before_delivery() -> None:
    config = load_guardrails_config()
    config.layer3.enabled = False
    provider = _SequenceProvider([_invalid_review(), _valid_review()])
    graph = build_mvp_graph(provider, guardrails=config)

    result = await graph.ainvoke(
        {"user_message": "Review my code.", "learner_code": "print(1)"},
        {"configurable": {"thread_id": "m6-repair"}},
    )

    assert result["repair_count"] == 1
    assert result["guardrail_action"] == "pass"
    assert result["candidate_output"] == _valid_review()
    assert result["intervention_history"][0]["original_candidate"] == _invalid_review()
    assert [entry["phase"] for entry in result["drift_history"]] == [
        "initial",
        "post_repair",
    ]


async def test_reset_clears_active_context_and_preserves_task_state() -> None:
    provider = _SequenceProvider([_invalid_review(), _valid_review()])
    graph = build_mvp_graph(provider, guardrails=load_guardrails_config())
    state = {
        "user_message": "Review my code.",
        "learner_code": "print(1)",
        "current_problem": {"id": "p-1"},
        "learner_level": "beginner",
        "latest_test_results": [{"name": "basic", "passed": False}],
        "agent_contexts": {
            "code_reviewer": [{"private": "contaminated"}],
            "problem_designer": [{"keep": "separate role"}],
        },
    }

    result = await graph.ainvoke(state, {"configurable": {"thread_id": "m6-reset"}})

    assert result["reset_count"] == 1
    assert result["repair_count"] == 0
    assert result["current_problem"] == state["current_problem"]
    assert result["learner_code"] == state["learner_code"]
    assert result["latest_test_results"] == state["latest_test_results"]
    assert result["agent_contexts"]["problem_designer"] == [{"keep": "separate role"}]
    assert result["agent_contexts"]["code_reviewer"] == [
        {"candidate_output": _valid_review()}
    ]
    assert result["intervention_history"][0]["reason"] == "forbidden_action_attempt"
    assert result["drift_history"][-1]["phase"] == "post_reset"


async def test_intervention_limits_terminate_without_an_infinite_loop() -> None:
    config = load_guardrails_config()
    config.layer3.enabled = False
    config.limits.max_repair_attempts_per_turn = 1
    provider = _SequenceProvider([_invalid_review()])
    graph = build_mvp_graph(provider, guardrails=config)

    result = await graph.ainvoke(
        {"user_message": "Review my code.", "learner_code": "print(1)"},
        {"configurable": {"thread_id": "m6-bounded"}},
    )

    assert result["repair_count"] == 1
    assert result["reset_count"] == 0
    assert result["guardrail_action"] == "block"
    assert result["error_type"] == "InterventionLimitError"


async def test_layer_three_toggle_preserves_static_graph_topology() -> None:
    enabled_graph = build_mvp_graph(_SequenceProvider([_valid_review()]))
    disabled = load_guardrails_config()
    disabled.layer3.enabled = False
    disabled_graph = build_mvp_graph(_SequenceProvider([_valid_review()]), guardrails=disabled)

    assert set(enabled_graph.get_graph().nodes) == set(disabled_graph.get_graph().nodes)
