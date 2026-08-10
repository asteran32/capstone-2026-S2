"""Milestone 7 isolated Python sandbox tests."""

from harness.config import load_safety_config
from harness.models.schemas import ExecutionRequest
from harness.safety.sandbox import SandboxExecutor


def test_sandbox_executes_approved_python_request() -> None:
    result = SandboxExecutor(load_safety_config().execution).execute(
        ExecutionRequest(
            language="python",
            source_code="print('3')",
            test_cases=[{"name": "basic", "expected": "3"}],
            timeout_seconds=1,
        )
    )

    assert result.status == "success"
    assert result.test_results[0].passed is True
    assert result.stdout == "3\n"


def test_sandbox_blocks_unapproved_network_request() -> None:
    result = SandboxExecutor(load_safety_config().execution).execute(
        ExecutionRequest(
            language="python",
            source_code="import socket\nsocket.create_connection(('example.invalid', 80))",
            test_cases=[],
            timeout_seconds=1,
        )
    )

    assert result.status == "blocked"
    assert "blocked_import:socket" in result.stderr


def test_sandbox_enforces_request_timeout() -> None:
    result = SandboxExecutor(load_safety_config().execution).execute(
        ExecutionRequest(
            language="python",
            source_code="while True:\n    pass",
            test_cases=[],
            timeout_seconds=1,
        )
    )

    assert result.status == "timeout"
    assert result.exit_code is None
