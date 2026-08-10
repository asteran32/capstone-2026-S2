"""Pydantic schemas shared by future agents and harness components."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    """Base model that rejects fields outside the documented contract."""

    model_config = ConfigDict(extra="forbid")


class ProblemDesignInput(SchemaModel):
    learner_level: str
    topic: str
    difficulty: str
    constraints: list[str] = Field(default_factory=list)
    relevant_context: list[dict[str, Any]] = Field(default_factory=list)


class ProblemDesignOutput(SchemaModel):
    problem_id: str
    title: str
    statement: str
    constraints: list[str]
    examples: list[dict[str, Any]]
    reference_solution: str | None = None
    test_specification: list[dict[str, Any]]
    difficulty: str


class CodeReviewInput(SchemaModel):
    problem: dict[str, Any]
    learner_code: str
    learner_level: str
    test_results: list[dict[str, Any]] = Field(default_factory=list)
    relevant_context: list[dict[str, Any]] = Field(default_factory=list)


class CodeReviewOutput(SchemaModel):
    correctness_analysis: str
    detected_issues: list[dict[str, Any]]
    hints: list[str]
    pedagogical_feedback: str
    confidence: float


class ExecutionRequest(SchemaModel):
    language: Literal["python"]
    source_code: str
    test_cases: list[dict[str, Any]]
    timeout_seconds: int = Field(gt=0)


class TestCaseResult(SchemaModel):
    name: str
    passed: bool
    expected: str | None
    actual: str | None
    stderr: str | None


class ExecutionResult(SchemaModel):
    status: Literal["success", "failure", "timeout", "blocked"]
    test_results: list[TestCaseResult]
    duration_ms: int = Field(ge=0)
    exit_code: int | None
    stdout: str
    stderr: str


class ConsistencyResult(SchemaModel):
    valid: bool
    role_adherence_score: float = Field(ge=0.0, le=1.0)
    violations: list[str]
    drift_signals: dict[str, float]
    recommended_action: Literal["pass", "repair", "reset", "block"]
    evaluator_version: str
    schema_errors: list[str] = Field(default_factory=list)


class DriftObservation(SchemaModel):
    agent_id: str
    turn_id: int
    drift_score: float = Field(ge=0.0, le=1.0)
    indicators: dict[str, float]
    violations: list[str]
    action: Literal["pass", "repair", "reset", "block"]
    evaluator_version: str


class ExperimentObservation(SchemaModel):
    experiment_id: str
    session_id: str
    turn_id: int
    condition: str
    agent: str
    role_adherence_score: float | None
    drift_score: float | None
    boundary_violation: bool
    forbidden_action_attempt: bool
    repair_count: int
    reset_count: int
    task_success: bool | None
    latency_ms: int
    token_input: int | None
    token_output: int | None


class AgentBehaviourState(SchemaModel):
    agent_id: str
    total_turns: int
    drift_score_current: float
    drift_score_mean: float
    boundary_violation_count: int
    forbidden_action_count: int
    repair_count: int
    reset_count: int
    consecutive_violation_count: int
