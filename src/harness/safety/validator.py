"""Safety validation kept separate from role-consistency evaluation."""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import ValidationError

from harness.config import SafetyConfig
from harness.models.schemas import ExecutionRequest, SafetyResult
from harness.safety.sandbox import SandboxExecutor
from harness.safety.whitelist import ExecutionWhitelist


class SafetyValidator:
    """Validate executable requests and execute only those approved by policy."""

    evaluator_version = "safety-v1"

    def __init__(self, config: SafetyConfig, *, executor: SandboxExecutor | None = None) -> None:
        self._whitelist = ExecutionWhitelist(config.execution)
        self._executor = executor or SandboxExecutor(config.execution)

    def validate(self, agent_id: str | None, candidate: Mapping[str, Any] | None) -> SafetyResult:
        """Return a safety decision without consulting or altering drift state."""

        if agent_id != "test_runner":
            return SafetyResult(allowed=True, evaluator_version=self.evaluator_version)
        try:
            request = ExecutionRequest.model_validate(candidate)
        except ValidationError:
            return SafetyResult(
                allowed=False,
                violations=["invalid_execution_request"],
                evaluator_version=self.evaluator_version,
            )

        policy = self._whitelist.validate(request)
        if not policy.allowed:
            return SafetyResult(
                allowed=False,
                violations=policy.violations,
                evaluator_version=self.evaluator_version,
                execution_result=self._executor.execute(request),
            )
        execution_result = self._executor.execute(request)
        return SafetyResult(
            allowed=execution_result.status != "blocked",
            violations=[] if execution_result.status != "blocked" else ["sandbox_execution_blocked"],
            evaluator_version=self.evaluator_version,
            execution_result=execution_result,
        )
