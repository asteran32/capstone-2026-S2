"""Typed application configuration and YAML loading utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationError,
    model_validator,
)


class ConfigurationError(Exception):
    """Raised when application configuration cannot be loaded or validated."""


class ModelConfig(BaseModel):
    """Configuration shared by future LLM provider implementations."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    seed: int | None = None


class LayerConfig(BaseModel):
    """Toggle for an independently configurable guardrail layer."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool


class DriftWeightsConfig(BaseModel):
    """Provisional drift-indicator weight configuration."""

    model_config = ConfigDict(extra="forbid")

    boundary_violation: float = Field(ge=0.0, le=1.0)
    forbidden_action_attempt: float = Field(ge=0.0, le=1.0)
    role_similarity_deviation: float = Field(ge=0.0, le=1.0)
    schema_violation: float = Field(ge=0.0, le=1.0)
    instruction_deviation: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "DriftWeightsConfig":
        total = sum(self.model_dump().values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError("drift-score weights must sum to 1.0")
        return self


class DriftScoreConfig(BaseModel):
    """Provisional drift-scoring configuration."""

    model_config = ConfigDict(extra="forbid")

    weights: DriftWeightsConfig


class DriftThresholdsConfig(BaseModel):
    """Thresholds used by future repair and reset policy implementations."""

    model_config = ConfigDict(extra="forbid")

    repair: float = Field(ge=0.0, le=1.0)
    reset: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def reset_not_lower_than_repair(self) -> "DriftThresholdsConfig":
        if self.reset < self.repair:
            raise ValueError("reset threshold must be greater than or equal to repair")
        return self


class GuardrailLimitsConfig(BaseModel):
    """Bounds for future guardrail intervention loops."""

    model_config = ConfigDict(extra="forbid")

    max_repair_attempts_per_turn: int = Field(ge=0)
    max_resets_per_turn: int = Field(ge=0)


class GuardrailsConfig(BaseModel):
    """Guardrail-layer settings and provisional scoring values."""

    model_config = ConfigDict(extra="forbid")

    layer1: LayerConfig
    layer2: LayerConfig
    layer3: LayerConfig
    drift_score: DriftScoreConfig
    thresholds: DriftThresholdsConfig
    limits: GuardrailLimitsConfig


class ExperimentMetadataConfig(BaseModel):
    """Experiment identity and reproducibility settings."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    condition: str = Field(min_length=1)
    repetitions: int = Field(ge=1)
    seed: int | None = None


class ExperimentGuardrailsConfig(BaseModel):
    """Per-experiment layer activation settings."""

    model_config = ConfigDict(extra="forbid")

    layer1: LayerConfig
    layer2: LayerConfig
    layer3: LayerConfig


class LoggingConfig(BaseModel):
    """Experiment logging settings."""

    model_config = ConfigDict(extra="forbid")

    save_raw_outputs: bool
    save_state_snapshots: bool


class ExperimentConfig(BaseModel):
    """Experiment configuration loaded from ``experiment.yaml``."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentMetadataConfig
    guardrails: ExperimentGuardrailsConfig
    logging: LoggingConfig


class ExecutionSafetyConfig(BaseModel):
    """Static safety settings for the future sandbox implementation."""

    model_config = ConfigDict(extra="forbid")

    allowed_languages: list[str] = Field(min_length=1)
    timeout_seconds: int = Field(gt=0)
    network_enabled: bool
    allow_arbitrary_shell: bool
    max_output_chars: int = Field(gt=0)


class SafetyConfig(BaseModel):
    """Safety configuration loaded from ``safety.yaml``."""

    model_config = ConfigDict(extra="forbid")

    execution: ExecutionSafetyConfig


class RoleConfiguration(RootModel[dict[str, Any]]):
    """Typed YAML mapping; M2 adds the semantic role-contract model."""


class ApplicationConfig(BaseModel):
    """Complete typed application configuration available after M1."""

    model_config = ConfigDict(frozen=True)

    models: ModelConfig
    experiment: ExperimentConfig
    guardrails: GuardrailsConfig
    safety: SafetyConfig


_PLACEHOLDER = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")


def _resolve_environment_values(
    value: Any,
    environment: Mapping[str, str],
    *,
    path: Path,
) -> Any:
    if isinstance(value, str):
        match = _PLACEHOLDER.fullmatch(value)
        if not match:
            return value
        variable = match.group(1)
        resolved = environment.get(variable)
        if not resolved:
            raise ConfigurationError(
                f"{path}: required environment variable {variable} is missing"
            )
        return resolved
    if isinstance(value, list):
        return [
            _resolve_environment_values(item, environment, path=path)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _resolve_environment_values(item, environment, path=path)
            for key, item in value.items()
        }
    return value


def _load_yaml_mapping(
    path: Path,
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file not found: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {path}: {error}") from error

    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Configuration in {path} must be a YAML mapping")

    resolved = _resolve_environment_values(loaded, environment, path=path)
    return dict(resolved)


def _validate_config(
    config_type: type[BaseModel], data: dict[str, Any], path: Path) -> BaseModel:
    try:
        return config_type.model_validate(data)
    except ValidationError as error:
        raise ConfigurationError(f"Invalid configuration in {path}: {error}") from error


def load_application_config(
    config_dir: str | Path = "config",
    *,
    environment: Mapping[str, str] | None = None,
) -> ApplicationConfig:
    """Load the four typed application configuration files from ``config_dir``."""

    directory = Path(config_dir)
    resolved_environment = os.environ if environment is None else environment

    model_path = directory / "models.yaml"
    experiment_path = directory / "experiment.yaml"
    guardrail_path = directory / "guardrails.yaml"
    safety_path = directory / "safety.yaml"

    models = _validate_config(
        ModelConfig,
        _load_yaml_mapping(model_path, environment=resolved_environment),
        model_path,
    )
    experiment = _validate_config(
        ExperimentConfig,
        _load_yaml_mapping(experiment_path, environment=resolved_environment),
        experiment_path,
    )
    guardrails = _validate_config(
        GuardrailsConfig,
        _load_yaml_mapping(guardrail_path, environment=resolved_environment),
        guardrail_path,
    )
    safety = _validate_config(
        SafetyConfig,
        _load_yaml_mapping(safety_path, environment=resolved_environment),
        safety_path,
    )

    return ApplicationConfig(
        models=models,
        experiment=experiment,
        guardrails=guardrails,
        safety=safety,
    )


def load_role_configuration(path: str | Path) -> RoleConfiguration:
    """Load a role YAML mapping before M2 adds semantic contract validation."""

    role_path = Path(path)
    role_data = _load_yaml_mapping(role_path, environment=os.environ)
    return RoleConfiguration.model_validate(role_data)
