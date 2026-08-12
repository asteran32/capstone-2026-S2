"""JSONL trace persistence and graph-facing event recording."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from harness.observability.events import PROMPT_VERSION, TraceEvent, TraceEventType
from harness.observability.metrics import MetricsCollector


class TraceLoggingError(RuntimeError):
    """Raised when a trace cannot be serialized or written safely."""


_SENSITIVE_KEY = re.compile(
    r"^(?:api[_-]?key|authorization|password|secret|token|credential|"
    r"access[_-]?token|refresh[_-]?token)$",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(?:openai[_-]?api[_-]?key|api[_-]?key|authorization|password|secret|token)"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


def redact_sensitive_data(value: Any, *, key: str | None = None) -> Any:
    """Return a recursively redacted copy suitable for long-lived experiment logs."""

    if key is not None and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_sensitive_data(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, str):
        value = _SENSITIVE_ASSIGNMENT.sub("[REDACTED]", value)
        return _OPENAI_KEY.sub("[REDACTED]", value)
    return value


class JSONLTraceLogger:
    """Append safe, structured trace events to a single JSON Lines file."""

    def __init__(self, trace_dir: str | Path = "data/traces") -> None:
        self._trace_dir = Path(trace_dir)
        self._path = self._trace_dir / "trace.jsonl"
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """The JSONL target, created lazily on the first recorded event."""

        return self._path

    def log(self, event: TraceEvent) -> None:
        """Serialize one event atomically enough for single-process harness runs."""

        try:
            payload = redact_sensitive_data(event.model_dump(mode="json"))
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            with self._lock:
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as trace_file:
                    trace_file.write(f"{serialized}\n")
        except (OSError, TypeError, ValueError) as error:
            raise TraceLoggingError(f"Unable to write trace event to {self._path}") from error


@dataclass(frozen=True)
class TraceMetadata:
    """Stable reproducibility metadata included with every graph event."""

    model_configuration: Mapping[str, Any]
    contract_versions: Mapping[str, str]
    prompt_version: str = PROMPT_VERSION


class TraceRecorder:
    """Build trace events, write them, and retain their raw observations for metrics."""

    def __init__(
        self,
        logger: JSONLTraceLogger,
        metrics: MetricsCollector,
        metadata: TraceMetadata,
    ) -> None:
        self._logger = logger
        self._metrics = metrics
        self._metadata = metadata

    @property
    def metrics(self) -> MetricsCollector:
        """Expose the injected collector without storing metrics in agent state."""

        return self._metrics

    def emit(
        self,
        event_type: TraceEventType,
        state: Mapping[str, Any],
        *,
        agent: str | None = None,
        evaluator_version: str | None = None,
        candidate_output: Mapping[str, Any] | None = None,
        final_output: Mapping[str, Any] | None = None,
        candidate_output_id: str | None = None,
        caused_by_output_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TraceEvent:
        """Create and persist an event linked to a fully initialized graph turn."""

        event_agent = agent or state.get("active_agent")
        event = TraceEvent(
            event_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            event=event_type,
            session_id=str(state["session_id"]),
            thread_id=str(state["thread_id"]),
            trace_id=str(state["trace_id"]),
            turn_id=int(state["turn_id"]),
            experiment_id=state.get("experiment_id"),
            experiment_condition=state.get("experiment_condition"),
            agent=event_agent,
            prompt_version=self._metadata.prompt_version if event_agent is not None else None,
            contract_version=(
                self._metadata.contract_versions.get(event_agent)
                if event_agent is not None
                else None
            ),
            evaluator_version=evaluator_version,
            model_configuration=dict(self._metadata.model_configuration),
            drift_score=state.get("drift_score"),
            drift_indicators=dict(state.get("drift_indicators", {})),
            guardrail_action=state.get("guardrail_action"),
            candidate_output_id=candidate_output_id or state.get("candidate_output_id"),
            caused_by_output_id=caused_by_output_id,
            candidate_output=(
                None if candidate_output is None else dict(candidate_output)
            ),
            final_output=None if final_output is None else dict(final_output),
            metadata={} if metadata is None else dict(metadata),
        )
        self._logger.log(event)
        self._metrics.record(event)
        return event
