"""Bounded candidate repair using the existing provider and role contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from harness.guardrails.contracts import AgentId, RoleContractLoader, build_agent_prompt
from harness.models.provider import LLMProvider
from harness.models.schemas import CodeReviewOutput, ExecutionRequest, ProblemDesignOutput


class RepairError(Exception):
    """Raised when a repaired response does not satisfy the active role schema."""


_OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "ProblemDesignOutput": ProblemDesignOutput,
    "CodeReviewOutput": CodeReviewOutput,
    "ExecutionRequest": ExecutionRequest,
}


class RepairManager:
    """Requests a schema-valid correction without introducing a fourth role agent."""

    def __init__(
        self, provider: LLMProvider, *, contract_loader: RoleContractLoader | None = None
    ) -> None:
        self._provider = provider
        self._contract_loader = contract_loader or RoleContractLoader()

    async def repair(
        self,
        agent_id: AgentId,
        *,
        original_candidate: Mapping[str, Any] | None,
        task_context: Mapping[str, Any],
        projected_context: Mapping[str, Any],
        violations: list[str],
        schema_errors: list[str],
    ) -> BaseModel:
        """Request a corrected output while preserving the original task context."""

        contract = self._contract_loader.load(agent_id)
        output_schema = _OUTPUT_SCHEMAS.get(contract.output.schema_name)
        if output_schema is None:
            raise RepairError(f"Unsupported output schema: {contract.output.schema_name}")

        repair_context = dict(task_context)
        repair_context["repair_request"] = {
            "violations": violations,
            "schema_errors": schema_errors,
            "original_candidate": original_candidate,
            "instruction": (
                "Rewrite the candidate while preserving valid task content and "
                "remaining strictly within the role contract."
            ),
        }
        messages = build_agent_prompt(
            contract,
            task_context=repair_context,
            projected_context=projected_context,
            output_schema=output_schema,
        )
        result = await self._provider.generate(messages, output_schema)
        output = result.output
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError as error:
                raise RepairError("Repair provider returned non-JSON output") from error
        try:
            return output_schema.model_validate(output)
        except ValidationError as error:
            raise RepairError(
                f"Repair provider returned invalid {output_schema.__name__}"
            ) from error
