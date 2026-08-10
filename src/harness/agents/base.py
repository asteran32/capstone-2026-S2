"""Shared provider, contract, prompt, and output-validation behaviour for agents."""

from __future__ import annotations

import json
from typing import Any, Generic, Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from harness.guardrails.contracts import AgentId, RoleContractLoader, build_agent_prompt
from harness.models.provider import GenerationResult, LLMProvider


OutputModel = TypeVar("OutputModel", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when provider output does not satisfy an agent's output schema."""


class BaseAgent(Generic[OutputModel]):
    """Common implementation for a single role-specialized LLM agent."""

    agent_id: AgentId
    output_schema: type[OutputModel]

    def __init__(
        self,
        provider: LLMProvider,
        *,
        contract_loader: RoleContractLoader | None = None,
    ) -> None:
        self._provider = provider
        self._contract_loader = contract_loader or RoleContractLoader()

    async def invoke(
        self,
        task_context: BaseModel | Mapping[str, Any],
        projected_context: BaseModel | Mapping[str, Any],
    ) -> OutputModel:
        """Generate and validate a role-scoped candidate output."""

        contract = self._contract_loader.load(self.agent_id)
        prompt = build_agent_prompt(
            role_contract=contract,
            task_context=_to_mapping(task_context),
            projected_context=_to_mapping(projected_context),
            output_schema=self.output_schema,
        )
        result = await self._provider.generate(prompt, self.output_schema)
        return self._validate_output(result)

    def _validate_output(self, result: GenerationResult) -> OutputModel:
        output = result.output
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError as error:
                raise StructuredOutputError(
                    f"{self.agent_id} returned non-JSON structured output"
                ) from error
        try:
            return self.output_schema.model_validate(output)
        except ValidationError as error:
            raise StructuredOutputError(
                f"{self.agent_id} returned invalid {self.output_schema.__name__}"
            ) from error


def _to_mapping(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)
