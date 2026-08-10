"""Typed shared state passed between future LangGraph nodes."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict


AgentName = Literal["problem_designer", "code_reviewer", "test_runner"]
GuardrailAction = Literal["pass", "repair", "reset", "block"]


class HarnessState(TypedDict, total=False):
    """JSON-serializable state contract for a single harness session."""

    session_id: str
    thread_id: str
    trace_id: str
    turn_id: int

    learner_id: str | None
    learner_level: str

    user_message: str
    learner_code: str | None

    intent: str
    active_agent: AgentName | None

    current_problem: dict[str, Any] | None
    latest_test_results: list[dict[str, Any]]
    agent_contexts: dict[AgentName, list[dict[str, Any]]]

    candidate_output: dict[str, Any] | None
    final_output: dict[str, Any] | None

    role_contract_result: dict[str, Any] | None
    consistency_result: dict[str, Any] | None
    safety_result: dict[str, Any] | None

    drift_score: float
    drift_indicators: dict[str, float]
    drift_history: Annotated[list[dict[str, Any]], add]

    retry_count: int
    repair_count: int
    reset_count: int
    guardrail_action: GuardrailAction | None
    monitor_phase: Literal["initial", "post_repair", "post_reset"]
    intervention_history: Annotated[list[dict[str, Any]], add]

    experiment_id: str | None
    experiment_condition: str
    random_seed: int | None

    error_type: str | None
    error_message: str | None
