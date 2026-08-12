"""Milestone 8 tests for safe JSONL traces and raw metric derivation."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from harness.observability.events import TraceEvent, TraceEventType
from harness.observability.logger import JSONLTraceLogger, redact_sensitive_data
from harness.observability.metrics import MetricsCollector


def _event(
    event_type: TraceEventType,
    *,
    agent: str = "code_reviewer",
    drift_score: float = 0.0,
    indicators: dict[str, float] | None = None,
) -> TraceEvent:
    return TraceEvent(
        event_id=f"event-{event_type}",
        timestamp=datetime.now(timezone.utc),
        event=event_type,
        session_id="session-1",
        thread_id="thread-1",
        trace_id="trace-1",
        turn_id=1,
        agent=agent,
        drift_score=drift_score,
        drift_indicators={} if indicators is None else indicators,
    )


def test_jsonl_trace_serializes_machine_readable_event_without_credentials(tmp_path) -> None:
    logger = JSONLTraceLogger(tmp_path)
    event = _event(TraceEventType.AGENT_OUTPUT_RECEIVED).model_copy(
        update={
            "candidate_output": {
                "api_key": "sk-secret-value",
                "message": "OPENAI_API_KEY=sk-other-secret",
            }
        }
    )

    logger.log(event)

    serialized = json.loads(logger.path.read_text(encoding="utf-8"))
    assert serialized["event"] == "AGENT_OUTPUT_RECEIVED"
    assert serialized["session_id"] == "session-1"
    assert serialized["candidate_output"]["api_key"] == "[REDACTED]"
    assert "sk-secret" not in logger.path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in serialized["candidate_output"]["message"]


def test_redaction_preserves_non_sensitive_observations() -> None:
    assert redact_sensitive_data({"drift_score": 0.3}) == {"drift_score": 0.3}


def test_metrics_retain_raw_events_and_derive_behavioural_state() -> None:
    collector = MetricsCollector()
    collector.record(
        _event(
            TraceEventType.CONSISTENCY_CHECKED,
            drift_score=0.6,
            indicators={"role_boundary_violation": 1.0, "forbidden_action_attempt": 0.0},
        )
    )
    collector.record(_event(TraceEventType.REPAIR_TRIGGERED))
    collector.record(
        _event(
            TraceEventType.CONSISTENCY_CHECKED,
            drift_score=0.7,
            indicators={"role_boundary_violation": 0.0, "forbidden_action_attempt": 1.0},
        )
    )
    collector.record(_event(TraceEventType.RESET_TRIGGERED))

    behaviour = collector.behavioural_state("code_reviewer")

    assert len(collector.events) == 4
    assert behaviour.total_turns == 2
    assert behaviour.drift_score_current == 0.7
    assert behaviour.drift_score_mean == pytest.approx(0.65)
    assert behaviour.boundary_violation_count == 1
    assert behaviour.forbidden_action_count == 1
    assert behaviour.repair_count == 1
    assert behaviour.reset_count == 1
    assert behaviour.consecutive_violation_count == 2
