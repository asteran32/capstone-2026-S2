"""Raw trace retention and derived per-agent behavioural metrics."""

from __future__ import annotations

from collections import defaultdict

from harness.models.schemas import AgentBehaviourState
from harness.observability.events import TraceEvent, TraceEventType


class MetricsCollector:
    """Collect raw trace observations without discarding their source events."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Return immutable access to all raw events retained for later analysis."""

        return tuple(self._events)

    def record(self, event: TraceEvent) -> None:
        """Retain an unaggregated observation before any metric is derived."""

        self._events.append(event)

    def behavioural_state(self, agent_id: str) -> AgentBehaviourState:
        """Derive the required behaviour vector from retained trace observations."""

        consistency_events = [
            event
            for event in self._events
            if event.event is TraceEventType.CONSISTENCY_CHECKED and event.agent == agent_id
        ]
        drift_scores = [event.drift_score or 0.0 for event in consistency_events]
        boundary_violations = sum(
            event.drift_indicators.get("role_boundary_violation", 0.0) > 0.0
            for event in consistency_events
        )
        forbidden_actions = sum(
            event.drift_indicators.get("forbidden_action_attempt", 0.0) > 0.0
            for event in consistency_events
        )
        interventions = defaultdict(int)
        for event in self._events:
            if event.agent != agent_id:
                continue
            if event.event is TraceEventType.REPAIR_TRIGGERED:
                interventions["repair"] += 1
            elif event.event is TraceEventType.RESET_TRIGGERED:
                interventions["reset"] += 1

        consecutive_violations = 0
        for event in reversed(consistency_events):
            if (
                event.drift_indicators.get("role_boundary_violation", 0.0) > 0.0
                or event.drift_indicators.get("forbidden_action_attempt", 0.0) > 0.0
            ):
                consecutive_violations += 1
            else:
                break

        return AgentBehaviourState(
            agent_id=agent_id,
            total_turns=len(consistency_events),
            drift_score_current=drift_scores[-1] if drift_scores else 0.0,
            drift_score_mean=(sum(drift_scores) / len(drift_scores)) if drift_scores else 0.0,
            boundary_violation_count=boundary_violations,
            forbidden_action_count=forbidden_actions,
            repair_count=interventions["repair"],
            reset_count=interventions["reset"],
            consecutive_violation_count=consecutive_violations,
        )
