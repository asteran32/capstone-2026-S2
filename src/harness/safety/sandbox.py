"""Bounded Python-only sandbox executor with no shell command interface."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from harness.config import ExecutionSafetyConfig
from harness.models.schemas import (
    ExecutionRequest,
    ExecutionResult,
    TestCaseResult,
    TestCaseSpecification,
)
from harness.safety.whitelist import ExecutionWhitelist


class SandboxExecutionError(Exception):
    """Raised when the sandbox cannot prepare an approved execution."""


class SandboxExecutor:
    """Execute approved Python requests using a temporary, shell-free process."""

    def __init__(self, config: ExecutionSafetyConfig) -> None:
        self._config = config
        self._whitelist = ExecutionWhitelist(config)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Run an approved request and return a bounded, structured result."""

        policy = self._whitelist.validate(request)
        if not policy.allowed:
            return self._blocked_result("; ".join(policy.violations))

        started = time.monotonic()
        results: list[TestCaseResult] = []
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code: int | None = 0
        with tempfile.TemporaryDirectory(prefix="harness-sandbox-") as directory:
            sandbox_dir = Path(directory)
            source_path = sandbox_dir / "candidate.py"
            runner_path = sandbox_dir / "runner.py"
            source_path.write_text(request.source_code, encoding="utf-8")
            runner_path.write_text(
                _runner_source(network_enabled=self._config.network_enabled), encoding="utf-8"
            )
            test_cases = request.test_cases or [TestCaseSpecification(name="execution")]
            for index, test_case in enumerate(test_cases):
                case_name = test_case.name or f"case-{index + 1}"
                try:
                    completed = subprocess.run(
                        [sys.executable, "-I", str(runner_path), str(source_path)],
                        cwd=sandbox_dir,
                        input=test_case.input or "",
                        text=True,
                        capture_output=True,
                        timeout=request.timeout_seconds,
                        env=_sandbox_environment(sandbox_dir),
                        check=False,
                    )
                except subprocess.TimeoutExpired as error:
                    duration = _duration_ms(started)
                    partial_stdout = _limit_output(_as_text(error.stdout), self._config.max_output_chars)
                    partial_stderr = _limit_output(_as_text(error.stderr), self._config.max_output_chars)
                    results.append(
                        TestCaseResult(
                            name=case_name,
                            passed=False,
                            expected=_expected_output(test_case),
                            actual=partial_stdout or None,
                            stderr=partial_stderr or "execution timed out",
                        )
                    )
                    return ExecutionResult(
                        status="timeout",
                        test_results=results,
                        duration_ms=duration,
                        exit_code=None,
                        stdout=partial_stdout,
                        stderr=partial_stderr or "execution timed out",
                    )

                stdout = _limit_output(completed.stdout, self._config.max_output_chars)
                stderr = _limit_output(completed.stderr, self._config.max_output_chars)
                expected = _expected_output(test_case)
                passed = completed.returncode == 0 and (
                    expected is None or stdout.strip() == expected.strip()
                )
                results.append(
                    TestCaseResult(
                        name=case_name,
                        passed=passed,
                        expected=expected,
                        actual=stdout or None,
                        stderr=stderr or None,
                    )
                )
                stdout_parts.append(stdout)
                stderr_parts.append(stderr)
                exit_code = completed.returncode
                if not passed:
                    break

        return ExecutionResult(
            status="success" if all(result.passed for result in results) else "failure",
            test_results=results,
            duration_ms=_duration_ms(started),
            exit_code=exit_code,
            stdout=_limit_output("".join(stdout_parts), self._config.max_output_chars),
            stderr=_limit_output("".join(stderr_parts), self._config.max_output_chars),
        )

    @staticmethod
    def _blocked_result(reason: str) -> ExecutionResult:
        return ExecutionResult(
            status="blocked",
            test_results=[],
            duration_ms=0,
            exit_code=None,
            stdout="",
            stderr=reason,
        )


def _sandbox_environment(sandbox_dir: Path) -> dict[str, str]:
    """Provide a minimal environment so host secrets are not inherited."""

    return {"HOME": str(sandbox_dir), "LANG": "C", "PATH": ""}


def _runner_source(*, network_enabled: bool) -> str:
    if network_enabled:
        return """from pathlib import Path\nimport sys\nsource = Path(sys.argv[1])\nexec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'))\n"""
    return """from pathlib import Path\nimport socket\nimport sys\n\ndef _network_disabled(*args, **kwargs):\n    raise PermissionError('sandbox network access is disabled')\n\nsocket.socket = _network_disabled\nsocket.create_connection = _network_disabled\nsource = Path(sys.argv[1])\nexec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'))\n"""


def _expected_output(test_case: TestCaseSpecification) -> str | None:
    return test_case.expected


def _limit_output(value: str, maximum: int) -> str:
    return value[:maximum]


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode() if isinstance(value, bytes) else value


def _duration_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
