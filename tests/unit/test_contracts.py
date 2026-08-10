"""Milestone 2 tests for Layer 1 role contracts and prompt assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.guardrails.contracts import (
    AgentId,
    RoleContractError,
    RoleContractLoader,
    build_agent_prompt,
)


@pytest.mark.parametrize(
    ("agent_id", "forbidden_action"),
    [
        ("problem_designer", "review_learner_code"),
        ("code_reviewer", "execute_code"),
        ("test_runner", "arbitrary_shell_execution"),
    ],
)
def test_role_contracts_load_with_explicit_boundaries(
    agent_id: AgentId, forbidden_action: str
) -> None:
    contract = RoleContractLoader().load(agent_id)

    assert contract.contract_version == "1.0"
    assert contract.allowed_actions
    assert forbidden_action in contract.forbidden_actions
    assert not set(contract.allowed_actions) & set(contract.forbidden_actions)


def test_contract_version_is_available_programmatically() -> None:
    assert RoleContractLoader().contract_version("code_reviewer") == "1.0"


def test_contract_loader_rejects_mismatched_agent_id(tmp_path: Path) -> None:
    (tmp_path / "problem_designer.yaml").write_text(
        """agent_id: code_reviewer
contract_version: '1.0'
role: {name: Incorrect, purpose: Incorrect role}
allowed_actions: [inspect_code]
forbidden_actions: [execute_code]
output: {schema: CodeReviewOutput}
boundary_policy: {reject_out_of_scope_request: true, acknowledge_boundary: true}
"""
    )

    with pytest.raises(RoleContractError, match="expected 'problem_designer'"):
        RoleContractLoader(tmp_path).load("problem_designer")


def test_contract_loader_rejects_overlapping_action_lists(tmp_path: Path) -> None:
    (tmp_path / "problem_designer.yaml").write_text(
        """agent_id: problem_designer
contract_version: '1.0'
role: {name: Problem Designer, purpose: Create problems}
allowed_actions: [create_problem]
forbidden_actions: [create_problem]
output: {schema: ProblemDesignOutput}
boundary_policy: {reject_out_of_scope_request: true, acknowledge_boundary: true}
"""
    )

    with pytest.raises(RoleContractError, match="both allowed and forbidden"):
        RoleContractLoader(tmp_path).load("problem_designer")


def test_prompt_assembly_contains_all_required_parts() -> None:
    contract = RoleContractLoader().load("code_reviewer")

    prompt = build_agent_prompt(
        contract,
        task_context={"request": "Review this solution."},
        projected_context={"learner_code": "print(1)"},
        output_schema="CodeReviewOutput",
    )
    rendered = "\n".join(message["content"] for message in prompt)

    assert [message["role"] for message in prompt] == ["system", "user"]
    for section in (
        "BASE_AGENT_POLICY",
        "ROLE_CONTRACT",
        "CURRENT_TASK",
        "PROJECTED_CONTEXT",
        "OUTPUT_SCHEMA_INSTRUCTION",
    ):
        assert section in rendered
    assert "CodeReviewOutput" in rendered
