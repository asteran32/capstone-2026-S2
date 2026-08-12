"""Provider-neutral model generation interfaces and implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel

from harness.config import ModelConfig


class LLMInvocationError(Exception):
    """Raised when an LLM provider cannot complete a generation request."""


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token counts, when they are available."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Raw generated content plus optional provider usage metadata."""

    output: Any
    token_usage: TokenUsage | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Provider boundary used by role agents."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        response_schema: type[BaseModel] | None = None,
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate content without embedding domain-specific role logic."""


class MockProvider:
    """Deterministic provider returning fixtures keyed by response-schema name."""

    def __init__(self, responses: Mapping[str, Any]) -> None:
        self._responses = dict(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        response_schema: type[BaseModel] | None = None,
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        schema_name = None if response_schema is None else response_schema.__name__
        self.calls.append(
            {
                "messages": messages,
                "response_schema": response_schema,
                "temperature": temperature,
                "seed": seed,
            }
        )
        if schema_name is None:
            raise LLMInvocationError("MockProvider requires a response schema")
        if schema_name not in self._responses:
            raise LLMInvocationError(f"Mock response missing for schema: {schema_name}")
        return GenerationResult(output=self._responses[schema_name])


class OpenAIProvider:
    """OpenAI Responses API adapter configured by ``ModelConfig``."""

    def __init__(self, config: ModelConfig, *, client: Any | None = None) -> None:
        if config.provider != "openai":
            raise ValueError("OpenAIProvider requires ModelConfig.provider='openai'")
        self._config = config
        self._client = self._create_client() if client is None else client

    @staticmethod
    def _create_client() -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as error:
            raise LLMInvocationError(
                "OpenAIProvider requires the optional OpenAI SDK dependency"
            ) from error
        return AsyncOpenAI()

    async def generate(
        self,
        messages: list[dict[str, str]],
        response_schema: type[BaseModel] | None = None,
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        request: dict[str, Any] = {
            "model": self._config.model_name,
            "input": [
                {
                    "role": "developer" if message["role"] == "system" else message["role"],
                    "content": message["content"],
                }
                for message in messages
            ],
            "temperature": self._config.temperature if temperature is None else temperature,
        }
        # The Responses API does not currently accept a seed request field.
        # Keep it in the provider protocol/config for providers that support it
        # and for experiment metadata, but do not send it to OpenAI.
        if response_schema is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_schema.__name__,
                    "schema": response_schema.model_json_schema(),
                    "strict": True,
                }
            }

        try:
            response = await self._client.responses.create(**request)
        except Exception as error:
            raise LLMInvocationError("OpenAI generation failed") from error

        output: Any = getattr(response, "output_text", None)
        if response_schema is not None and isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                pass
        return GenerationResult(output=output, token_usage=_extract_token_usage(response))


def _extract_token_usage(response: Any) -> TokenUsage | None:
    """Normalize usage fields exposed by SDK response objects or test doubles."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    def get_value(name: str) -> int | None:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return value if isinstance(value, int) else None

    return TokenUsage(
        input_tokens=get_value("input_tokens") or get_value("prompt_tokens"),
        output_tokens=get_value("output_tokens") or get_value("completion_tokens"),
        total_tokens=get_value("total_tokens"),
    )
