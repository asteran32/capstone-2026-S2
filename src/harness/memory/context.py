"""Typed memory boundaries used by Milestone 2 context projection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from harness.guardrails.contracts import AgentId


class TaskMemory(BaseModel):
    """Essential task facts, deliberately independent of agent history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learner_level: str | None = None
    user_message: str | None = None
    learner_code: str | None = None
    current_problem: dict[str, Any] | None = None
    latest_test_results: list[dict[str, Any]] = Field(default_factory=list)


class AgentContextMemory(BaseModel):
    """Conversation artifacts belonging to exactly one role agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: AgentId
    history: list[dict[str, Any]] = Field(default_factory=list)


class ProjectedContext(BaseModel):
    """The only task and history data exposed to a selected agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: AgentId
    task: dict[str, Any]
    agent_context: list[dict[str, Any]] = Field(default_factory=list)
