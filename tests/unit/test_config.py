"""Tests for M1 typed configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import (
    ConfigurationError,
    PersistenceConfig,
    load_application_config,
    load_role_configuration,
)


def _write_valid_config(config_dir: Path) -> None:
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        "provider: mock\nmodel_name: test-model\ntemperature: 0.0\nseed: 7\n"
    )
    (config_dir / "experiment.yaml").write_text(
        """experiment:
  id: test-experiment
  condition: FULL
  repetitions: 1
  seed: 7
guardrails:
  layer1: {enabled: true}
  layer2: {enabled: true}
  layer3: {enabled: true}
logging:
  save_raw_outputs: true
  save_state_snapshots: false
"""
    )
    (config_dir / "guardrails.yaml").write_text(
        """layer1: {enabled: true}
layer2: {enabled: true}
layer3: {enabled: true}
drift_score:
  weights:
    boundary_violation: 0.30
    forbidden_action_attempt: 0.30
    role_similarity_deviation: 0.15
    schema_violation: 0.10
    instruction_deviation: 0.15
thresholds: {repair: 0.30, reset: 0.60}
limits: {max_repair_attempts_per_turn: 2, max_resets_per_turn: 1}
"""
    )
    (config_dir / "safety.yaml").write_text(
        """execution:
  allowed_languages: [python]
  timeout_seconds: 5
  network_enabled: false
  allow_arbitrary_shell: false
  max_output_chars: 20000
"""
    )


def test_valid_config_loads(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _write_valid_config(config_dir)

    config = load_application_config(config_dir)

    assert config.models.provider == "mock"
    assert config.models.model_name == "test-model"
    assert config.experiment.experiment.condition == "FULL"
    assert config.guardrails.thresholds.reset == 0.60
    assert config.safety.execution.allowed_languages == ["python"]
    assert config.experiment.persistence.backend == "memory"


def test_checked_in_config_loads_with_model_environment() -> None:
    config = load_application_config(environment={"MODEL_NAME": "test-model"})

    assert config.models.model_name == "test-model"
    assert config.guardrails.drift_score.weights.boundary_violation == 0.30


def test_invalid_config_fails(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _write_valid_config(config_dir)
    (config_dir / "models.yaml").write_text(
        "provider: mock\nmodel_name: test-model\ntemperature: 3.0\n"
    )

    with pytest.raises(ConfigurationError, match="models.yaml"):
        load_application_config(config_dir)


def test_invalid_role_configuration_fails_clearly(tmp_path: Path) -> None:
    role_path = tmp_path / "role.yaml"
    role_path.write_text("- not\n- a mapping\n")

    with pytest.raises(ConfigurationError, match="must be a YAML mapping"):
        load_role_configuration(role_path)


def test_sqlite_persistence_requires_a_durable_path() -> None:
    with pytest.raises(ValueError, match="sqlite_path is required"):
        PersistenceConfig(backend="sqlite")
    with pytest.raises(ValueError, match="cannot be :memory:"):
        PersistenceConfig(backend="sqlite", sqlite_path=":memory:")
