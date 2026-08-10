"""Milestone 6 tests for bounded, role-scoped candidate repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from harness.guardrails.repair import RepairError, RepairManager
from harness.models.provider import GenerationResult


@dataclass
class _FixedProvider:
    output: Any

    async def generate(self, *_: Any, **__: Any) -> GenerationResult:
        return GenerationResult(output=self.output)


def _review_output() -> dict[str, object]:
    return {
        "correctness_analysis": "The final item is omitted.",
        "detected_issues": [{"kind": "off-by-one"}],
        "hints": ["Check the range endpoint."],
        "pedagogical_feedback": "Trace the final iteration.",
        "confidence": 0.9,
    }


async def test_repair_preserves_task_context_and_returns_valid_role_output() -> None:
    repaired = await RepairManager(_FixedProvider(_review_output())).repair(
        "code_reviewer",
        original_candidate={"correctness_analysis": "I will execute_code."},
        task_context={"learner_code": "print(1)"},
        projected_context={"learner_code": "print(1)"},
        violations=["forbidden_action:execute_code"],
        schema_errors=[],
    )

    assert repaired.correctness_analysis == _review_output()["correctness_analysis"]


async def test_repair_rejects_invalid_provider_output() -> None:
    with pytest.raises(RepairError, match="invalid CodeReviewOutput"):
        await RepairManager(_FixedProvider({"confidence": "bad"})).repair(
            "code_reviewer",
            original_candidate={},
            task_context={},
            projected_context={},
            violations=[],
            schema_errors=[],
        )
