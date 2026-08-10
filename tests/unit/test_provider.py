"""Milestone 3 tests for provider abstraction and OpenAI adaptation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness.config import ModelConfig
from harness.models.provider import (
    LLMInvocationError,
    LLMProvider,
    MockProvider,
    OpenAIProvider,
)
from harness.models.schemas import CodeReviewOutput


def _review_output() -> dict[str, object]:
    return {
        "correctness_analysis": "The final element is skipped.",
        "detected_issues": [{"kind": "off-by-one"}],
        "hints": ["Check the range boundary."],
        "pedagogical_feedback": "Trace the last loop iteration.",
        "confidence": 0.9,
    }


async def test_mock_provider_is_deterministic_and_matches_protocol() -> None:
    provider = MockProvider({"CodeReviewOutput": _review_output()})

    first = await provider.generate([], CodeReviewOutput, temperature=0.2, seed=4)
    second = await provider.generate([], CodeReviewOutput, temperature=0.2, seed=4)

    assert isinstance(provider, LLMProvider)
    assert first.output == second.output == _review_output()
    assert len(provider.calls) == 2


async def test_mock_provider_fails_clearly_for_missing_fixture() -> None:
    with pytest.raises(LLMInvocationError, match="Mock response missing"):
        await MockProvider({}).generate([], CodeReviewOutput)


class _FakeResponses:
    def __init__(self, response: object | Exception) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    async def create(self, **request: object) -> object:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _openai_config() -> ModelConfig:
    return ModelConfig(
        provider="openai", model_name="test-model", temperature=0.3, seed=17
    )


async def test_openai_provider_uses_configured_structured_request_and_usage() -> None:
    responses = _FakeResponses(
        SimpleNamespace(
            output_text='{"confidence": 0.9}',
            usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
        )
    )
    provider = OpenAIProvider(_openai_config(), client=SimpleNamespace(responses=responses))

    result = await provider.generate(
        [{"role": "system", "content": "policy"}], CodeReviewOutput
    )

    request = responses.requests[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0.3
    assert request["seed"] == 17
    assert request["input"] == [{"role": "developer", "content": "policy"}]
    assert request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "CodeReviewOutput",
            "schema": CodeReviewOutput.model_json_schema(),
            "strict": True,
        }
    }
    assert result.output == {"confidence": 0.9}
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 18


async def test_openai_provider_normalizes_provider_errors() -> None:
    responses = _FakeResponses(RuntimeError("network unavailable"))
    provider = OpenAIProvider(_openai_config(), client=SimpleNamespace(responses=responses))

    with pytest.raises(LLMInvocationError, match="OpenAI generation failed"):
        await provider.generate([], CodeReviewOutput)
