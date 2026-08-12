"""Structured, machine-readable events emitted by the harness."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


PROMPT_VERSION = "base-agent-policy-v1"


class TraceEventType(StrEnum):
    """The minimum lifecycle events required for experiment trace analysis."""

    SESSION_STARTED = "SESSION_STARTED"
    USER_MESSAGE_RECEIVED = "USER_MESSAGE_RECEIVED"
    INTENT_CLASSIFIED = "INTENT_CLASSIFIED"
    AGENT_SELECTED = "AGENT_SELECTED"
    AGENT_INVOKED = "AGENT_INVOKED"
    AGENT_OUTPUT_RECEIVED = "AGENT_OUTPUT_RECEIVED"
    CONSISTENCY_CHECKED = "CONSISTENCY_CHECKED"
    DRIFT_SCORE_UPDATED = "DRIFT_SCORE_UPDATED"
    REPAIR_TRIGGERED = "REPAIR_TRIGGERED"
    RESET_TRIGGERED = "RESET_TRIGGERED"
    SAFETY_CHECKED = "SAFETY_CHECKED"
    SANDBOX_EXECUTION_REQUESTED = "SANDBOX_EXECUTION_REQUESTED"
    SANDBOX_EXECUTION_COMPLETED = "SANDBOX_EXECUTION_COMPLETED"
    OUTPUT_DELIVERED = "OUTPUT_DELIVERED"
    SESSION_ENDED = "SESSION_ENDED"
    ERROR = "ERROR"


class TraceEvent(BaseModel):
    """One immutable observation from a single graph turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    timestamp: datetime
    event: TraceEventType
    session_id: str
    thread_id: str
    trace_id: str
    turn_id: int = Field(ge=1)
    experiment_id: str | None = None
    experiment_condition: str | None = None
    agent: str | None = None
    prompt_version: str | None = None
    contract_version: str | None = None
    evaluator_version: str | None = None
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    drift_score: float | None = Field(default=None, ge=0.0, le=1.0)
    drift_indicators: dict[str, float] = Field(default_factory=dict)
    guardrail_action: str | None = None
    candidate_output_id: str | None = None
    caused_by_output_id: str | None = None
    candidate_output: dict[str, Any] | None = None
    final_output: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
