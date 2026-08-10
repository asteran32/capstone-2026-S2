"""Milestone 3 tests for the three fixed role-agent modules."""

from __future__ import annotations

import pytest

from harness.agents import CodeReviewAgent, ProblemDesignAgent, TestRunnerAgent
from harness.agents.base import StructuredOutputError
from harness.models.provider import MockProvider
from harness.models.schemas import (
    CodeReviewInput,
    CodeReviewOutput,
    ExecutionRequest,
    ProblemDesignInput,
    ProblemDesignOutput,
)


def _provider() -> MockProvider:
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
                "correctness_analysis": "The final item is omitted.",
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


@pytest.mark.parametrize(
    ("agent", "task_context", "expected_type"),
    [
        (
            ProblemDesignAgent(_provider()),
            ProblemDesignInput(learner_level="beginner", topic="loops", difficulty="easy"),
            ProblemDesignOutput,
        ),
        (
            CodeReviewAgent(_provider()),
            CodeReviewInput(
                problem={"id": "p-1"},
                learner_code="print(1)",
                learner_level="beginner",
            ),
            CodeReviewOutput,
        ),
        (
            TestRunnerAgent(_provider()),
            {"source_code": "print('test')", "available_test_cases": [{"name": "basic"}]},
            ExecutionRequest,
        ),
    ],
)
async def test_each_fixed_role_agent_runs_against_mock_provider(
    agent: object, task_context: object, expected_type: type[object]
) -> None:
    output = await agent.invoke(task_context, {"role_scoped": True})  # type: ignore[union-attr]

    assert isinstance(output, expected_type)


async def test_agents_inject_their_own_contract_without_inter_agent_calls() -> None:
    provider = _provider()
    agent = CodeReviewAgent(provider)

    await agent.invoke(
        CodeReviewInput(
            problem={"id": "p-1"}, learner_code="print(1)", learner_level="beginner"
        ),
        {"learner_code": "print(1)"},
    )

    prompt = provider.calls[0]["messages"]
    assert '"agent_id": "code_reviewer"' in prompt[0]["content"]
    assert len(provider.calls) == 1


async def test_invalid_provider_output_is_rejected_by_agent_schema() -> None:
    provider = MockProvider({"CodeReviewOutput": {"confidence": "not-a-number"}})
    agent = CodeReviewAgent(provider)

    with pytest.raises(StructuredOutputError, match="invalid CodeReviewOutput"):
        await agent.invoke(
            CodeReviewInput(
                problem={"id": "p-1"}, learner_code="print(1)", learner_level="beginner"
            ),
            {},
        )


async def test_test_runner_returns_request_without_executing_it() -> None:
    provider = _provider()
    agent = TestRunnerAgent(provider)

    request = await agent.invoke({"source_code": "print('test')"}, {})

    assert isinstance(request, ExecutionRequest)
    assert len(provider.calls) == 1
    assert request.language == "python"
