"""Approved execution-request policy for the Python sandbox."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from harness.config import ExecutionSafetyConfig
from harness.models.schemas import ExecutionRequest


@dataclass(frozen=True)
class WhitelistResult:
    """Result of validating an execution request against the approved policy."""

    allowed: bool
    violations: list[str]


class ExecutionWhitelist:
    """Allows only bounded Python source that needs no shell command from an LLM."""

    _blocked_import_roots = frozenset(
        {
            "_socket",
            "ctypes",
            "ftplib",
            "http",
            "multiprocessing",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "subprocess",
            "urllib",
        }
    )
    _blocked_calls = frozenset({"__import__", "compile", "eval", "exec", "open"})
    _blocked_text = frozenset(
        {
            "apt-get",
            "curl",
            "pip install",
            "rm -rf",
            "sudo",
            "wget",
            "ssh",
        }
    )

    def __init__(self, config: ExecutionSafetyConfig) -> None:
        self._config = config

    def validate(self, request: ExecutionRequest) -> WhitelistResult:
        """Validate language, bounds, and source before an executor is invoked."""

        violations: list[str] = []
        if request.language not in self._config.allowed_languages:
            violations.append(f"language_not_allowed:{request.language}")
        if request.timeout_seconds > self._config.timeout_seconds:
            violations.append("timeout_exceeds_policy")
        violations.extend(self._source_violations(request.source_code))
        return WhitelistResult(allowed=not violations, violations=violations)

    @classmethod
    def _source_violations(cls, source_code: str) -> list[str]:
        lowered = source_code.lower()
        violations = [
            f"blocked_operation:{operation}"
            for operation in sorted(cls._blocked_text)
            if operation in lowered
        ]
        if ".." in source_code:
            violations.append("host_filesystem_traversal")
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {name.name.split(".", 1)[0] for name in node.names}
                violations.extend(
                    f"blocked_import:{root}"
                    for root in sorted(roots & cls._blocked_import_roots)
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                if root in cls._blocked_import_roots:
                    violations.append(f"blocked_import:{root}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in cls._blocked_calls:
                    violations.append(f"blocked_call:{node.func.id}")
        return list(dict.fromkeys(violations))
