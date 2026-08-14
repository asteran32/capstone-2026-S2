"""Layer 1 role-contract loading and deterministic prompt assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


AgentId = Literal["problem_designer", "code_reviewer", "test_runner"]


class RoleContractError(Exception):
    """Raised when a role contract cannot be loaded or validated."""


class ContractRole(BaseModel):
    """Human-readable role identity and purpose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class ContractOutput(BaseModel):
    """The structured output schema expected from a role."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_by_alias=True)

    schema_name: str = Field(min_length=1, alias="schema", serialization_alias="schema")


class BoundaryPolicy(BaseModel):
    """The required response behaviour for out-of-scope requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reject_out_of_scope_request: bool
    acknowledge_boundary: bool


class RoleContract(BaseModel):
    """Validated machine-readable contract for one fixed harness role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: AgentId
    contract_version: str = Field(min_length=1)
    role: ContractRole
    allowed_actions: list[str] = Field(min_length=1)
    forbidden_actions: list[str] = Field(min_length=1)
    output: ContractOutput
    boundary_policy: BoundaryPolicy

    @model_validator(mode="after")
    def action_lists_are_unambiguous(self) -> "RoleContract":
        allowed = set(self.allowed_actions)
        forbidden = set(self.forbidden_actions)
        if len(allowed) != len(self.allowed_actions):
            raise ValueError("allowed_actions must not contain duplicates")
        if len(forbidden) != len(self.forbidden_actions):
            raise ValueError("forbidden_actions must not contain duplicates")
        overlap = allowed & forbidden
        if overlap:
            raise ValueError(
                "actions cannot be both allowed and forbidden: "
                + ", ".join(sorted(overlap))
            )
        return self


class RoleContractLoader:
    """Loads versioned role contracts from the repository configuration."""

    def __init__(self, contracts_dir: str | Path = "config/roles") -> None:
        self._contracts_dir = Path(contracts_dir)

    def load(self, agent_id: AgentId) -> RoleContract:
        """Load and validate the contract for ``agent_id``."""

        path = self._contracts_dir / f"{agent_id}.yaml"
        try:
            with path.open(encoding="utf-8") as contract_file:
                data = yaml.safe_load(contract_file)
        except FileNotFoundError as error:
            raise RoleContractError(f"Role contract not found: {path}") from error
        except yaml.YAMLError as error:
            raise RoleContractError(f"Invalid YAML in role contract {path}: {error}") from error

        if not isinstance(data, dict):
            raise RoleContractError(f"Role contract in {path} must be a YAML mapping")

        try:
            contract = RoleContract.model_validate(data)
        except ValidationError as error:
            raise RoleContractError(f"Invalid role contract in {path}: {error}") from error

        if contract.agent_id != agent_id:
            raise RoleContractError(
                f"Role contract {path} declares agent_id={contract.agent_id!r}; "
                f"expected {agent_id!r}"
            )
        return contract

    def contract_version(self, agent_id: AgentId) -> str:
        """Return an active contract version for trace metadata."""

        return self.load(agent_id).contract_version


def build_agent_prompt(
    role_contract: RoleContract | None,
    task_context: Mapping[str, Any],
    projected_context: Mapping[str, Any],
    output_schema: str | type[Any],
) -> list[dict[str, str]]:
    """Build a provider-neutral instruction sequence without invoking an LLM."""

    schema_name = output_schema if isinstance(output_schema, str) else output_schema.__name__
    task_text = json.dumps(dict(task_context), sort_keys=True, default=str)
    context_text = json.dumps(dict(projected_context), sort_keys=True, default=str)

    if role_contract is None:
        system_content = (
            "BASE_AGENT_POLICY\n"
            "Complete only the current task and return the required structured output."
        )
    else:
        contract_text = json.dumps(
            role_contract.model_dump(mode="json", by_alias=True), sort_keys=True
        )
        system_content = (
            "BASE_AGENT_POLICY\n"
            "Act only within the supplied role contract. Do not claim actions "
            "outside its allowed actions.\n\n"
            f"ROLE_CONTRACT\n{contract_text}"
        )

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": (
                f"CURRENT_TASK\n{task_text}\n\n"
                f"PROJECTED_CONTEXT\n{context_text}\n\n"
                f"OUTPUT_SCHEMA_INSTRUCTION\nReturn output conforming to {schema_name}."
            ),
        },
    ]
