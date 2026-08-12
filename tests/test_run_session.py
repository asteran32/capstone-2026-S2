"""CLI provider-selection tests that never make paid API requests."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import ConfigurationError
from harness.models.provider import MockProvider
from scripts import run_session


def _write_config(config_dir: Path, *, provider: str = "openai") -> None:
    source_dir = Path("config")
    for name in ("experiment.yaml", "guardrails.yaml", "safety.yaml"):
        (config_dir / name).write_text(
            (source_dir / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (config_dir / "models.yaml").write_text(
        f"provider: {provider}\n"
        "model_name: ${MODEL_NAME}\n"
        "temperature: 0.0\n"
        "seed: 42\n",
        encoding="utf-8",
    )


def test_create_provider_uses_mock_by_default_without_configuration() -> None:
    provider = run_session._create_provider("mock", environment={})

    assert isinstance(provider, MockProvider)


def test_create_openai_provider_requires_api_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY is required"):
        run_session._create_provider(
            "openai",
            config_dir=tmp_path,
            environment={"MODEL_NAME": "test-model"},
        )


def test_create_openai_provider_requires_model_name(tmp_path: Path) -> None:
    _write_config(tmp_path)

    with pytest.raises(ConfigurationError, match="MODEL_NAME is missing"):
        run_session._create_provider(
            "openai",
            config_dir=tmp_path,
            environment={"OPENAI_API_KEY": "test-key"},
        )


def test_create_openai_provider_uses_loaded_model_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path)
    captured: dict[str, object] = {}

    class FakeOpenAIProvider:
        def __init__(self, config: object) -> None:
            captured["config"] = config

    monkeypatch.setattr(run_session, "OpenAIProvider", FakeOpenAIProvider)

    provider = run_session._create_provider(
        "openai",
        config_dir=tmp_path,
        environment={
            "OPENAI_API_KEY": "test-key",
            "MODEL_NAME": "test-model",
        },
    )

    assert isinstance(provider, FakeOpenAIProvider)
    assert captured["config"].model_name == "test-model"  # type: ignore[union-attr]


def test_create_openai_provider_rejects_configured_provider_mismatch(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path, provider="another-provider")

    with pytest.raises(ConfigurationError, match="must set provider: openai"):
        run_session._create_provider(
            "openai",
            config_dir=tmp_path,
            environment={
                "OPENAI_API_KEY": "test-key",
                "MODEL_NAME": "test-model",
            },
        )
