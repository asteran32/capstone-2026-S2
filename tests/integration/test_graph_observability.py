"""Milestone 8 graph-level trace and durable checkpoint tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.config import ModelConfig, PersistenceConfig, load_guardrails_config
from harness.graph import build_mvp_graph, build_mvp_graph_async
from harness.models.provider import GenerationResult, MockProvider
from harness.observability.events import TraceEventType
from harness.observability.logger import JSONLTraceLogger
from harness.observability.metrics import MetricsCollector


def _valid_problem_provider() -> MockProvider:
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
                "correctness_analysis": "The final loop item is omitted.",
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


async def test_trace_events_capture_versions_and_reset_causal_linkage(tmp_path: Path) -> None:
    logger = JSONLTraceLogger(tmp_path / "traces")
    collector = MetricsCollector()
    graph = build_mvp_graph(
        _SequenceProvider([_invalid_review(), _valid_review()]),
        guardrails=load_guardrails_config(),
        trace_logger=logger,
        metrics_collector=collector,
        model_config=ModelConfig(
            provider="mock", model_name="deterministic-test", temperature=0.0, seed=7
        ),
    )

    result = await graph.ainvoke(
        {"user_message": "Review my code.", "learner_code": "print(1)"},
        {"configurable": {"thread_id": "m8-observability"}},
    )

    events = collector.events
    assert result["trace_id"]
    assert all(event.trace_id == result["trace_id"] for event in events)
    assert all(event.session_id == "m8-observability" for event in events)
    assert all(event.turn_id == 1 for event in events)
    assert TraceEventType.SESSION_STARTED in [event.event for event in events]
    assert TraceEventType.OUTPUT_DELIVERED in [event.event for event in events]
    initial_candidate = next(
        event
        for event in events
        if event.event is TraceEventType.AGENT_OUTPUT_RECEIVED
        and event.metadata["output_source"] == "agent"
    )
    reset_event = next(event for event in events if event.event is TraceEventType.RESET_TRIGGERED)
    assert reset_event.caused_by_output_id == initial_candidate.candidate_output_id
    assert initial_candidate.contract_version == "1.0"
    assert initial_candidate.prompt_version == "base-agent-policy-v1"
    assert initial_candidate.model_configuration["model_name"] == "deterministic-test"
    assert logger.path.exists()


async def test_sqlite_checkpoint_recovers_a_thread_in_a_new_graph(tmp_path: Path) -> None:
    persistence = PersistenceConfig(
        backend="sqlite", sqlite_path=str(tmp_path / "checkpoints" / "harness.sqlite")
    )
    config = {"configurable": {"thread_id": "m8-durable-thread"}}
    first_graph = await build_mvp_graph_async(
        _valid_problem_provider(),
        persistence=persistence,
        trace_logger=JSONLTraceLogger(tmp_path / "first-traces"),
    )

    first_result = await first_graph.ainvoke(
        {"user_message": "Give me a new problem."}, config
    )
    resumed_graph = await build_mvp_graph_async(
        _valid_problem_provider(),
        persistence=persistence,
        trace_logger=JSONLTraceLogger(tmp_path / "resumed-traces"),
    )
    resumed_result = await resumed_graph.ainvoke(
        {"user_message": "Give me another new problem."}, config
    )

    assert first_result["session_id"] == "m8-durable-thread"
    assert resumed_result["session_id"] == first_result["session_id"]
    assert resumed_result["turn_id"] == first_result["turn_id"] + 1
    assert resumed_result["thread_id"] == "m8-durable-thread"
