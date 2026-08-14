"""Tests for the M1 Pydantic schema contracts."""

import pytest
from pydantic import ValidationError

from harness.models.schemas import (
    CodeReviewOutput,
    ExecutionRequest,
    ExperimentObservation,
    ProblemDesignOutput,
    SafetyResult,
)


def test_output_schemas_validate() -> None:
    problem = ProblemDesignOutput(
        problem_id="p-1",
        title="Sum two values",
        statement="Return the sum of two integers.",
        constraints=["Use Python."],
        examples=[{"input": "1 2", "output": "3"}],
        test_specification=[{"name": "basic"}],
        difficulty="beginner",
    )
    review = CodeReviewOutput(
        correctness_analysis="The loop omits the final item.",
        detected_issues=[{"kind": "off-by-one"}],
        hints=["Check the range endpoint."],
        pedagogical_feedback="Try tracing the final iteration.",
        confidence=0.9,
    )
    observation = ExperimentObservation(
        experiment_id="exp-1",
        session_id="session-1",
        trace_id="trace-1",
        turn_id=1,
        condition="FULL",
        agent="code_reviewer",
        role_adherence_score=0.9,
        drift_score=0.1,
        boundary_violation=False,
        forbidden_action_attempt=False,
        repair_count=0,
        reset_count=0,
        task_success=True,
        latency_ms=12,
        token_input=10,
        token_output=20,
        drift_indicators={"role_boundary_violation": 0.0},
        model_configuration={"provider": "mock", "model_name": "test-model"},
        random_seed=42,
        configuration_hash="hash-1",
        configuration_source="config/experiment.yaml",
    )

    assert problem.reference_solution is None
    assert review.confidence == 0.9
    assert observation.condition == "FULL"
    assert SafetyResult(allowed=True, evaluator_version="safety-v1").allowed is True


def test_execution_request_rejects_unsupported_language() -> None:
    with pytest.raises(ValidationError):
        ExecutionRequest(
            language="javascript",
            source_code="console.log('hello')",
            test_cases=[],
            timeout_seconds=5,
        )


@pytest.mark.parametrize(
    "schema_type", [ProblemDesignOutput, CodeReviewOutput, ExecutionRequest]
)
def test_agent_output_schemas_use_closed_nested_objects(schema_type: type) -> None:
    schema = schema_type.model_json_schema()

    def assert_closed(value: object) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                assert value.get("additionalProperties") is False
            for child in value.values():
                assert_closed(child)
        elif isinstance(value, list):
            for child in value:
                assert_closed(child)

    assert_closed(schema)
