"""Milestone 7 execution-whitelist tests."""

from harness.config import load_safety_config
from harness.models.schemas import ExecutionRequest
from harness.safety.whitelist import ExecutionWhitelist


def test_execution_whitelist_allows_bounded_python_request() -> None:
    request = ExecutionRequest(
        language="python",
        source_code="print('approved')",
        test_cases=[],
        timeout_seconds=1,
    )

    result = ExecutionWhitelist(load_safety_config().execution).validate(request)

    assert result.allowed is True
    assert result.violations == []


def test_execution_whitelist_blocks_network_and_shell_operations() -> None:
    request = ExecutionRequest(
        language="python",
        source_code="import socket\n# curl https://example.invalid",
        test_cases=[],
        timeout_seconds=1,
    )

    result = ExecutionWhitelist(load_safety_config().execution).validate(request)

    assert result.allowed is False
    assert "blocked_import:socket" in result.violations
    assert "blocked_operation:curl" in result.violations
