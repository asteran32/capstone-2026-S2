"""Shared command-line composition for M9 baseline and ablation entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Sequence

from harness.config import (
    ConfigurationError,
    ModelConfig,
    load_application_config,
    load_experiment_config,
)
from harness.experiments.conditions import ExperimentCondition
from harness.experiments.runner import (
    ExperimentRunner,
    create_experiment_mock_provider,
    experiment_with_overrides,
)
from harness.models.provider import LLMInvocationError, OpenAIProvider


def run_experiment_cli(
    argv: Sequence[str] | None = None,
    *,
    forced_condition: ExperimentCondition | None = None,
) -> int:
    """Parse one M9 command, run it, and print reproducibility metadata."""

    parser = _parser(forced_condition=forced_condition)
    arguments = parser.parse_args(argv)
    try:
        config_path = Path(arguments.config)
        experiment = load_experiment_config(config_path)
        condition = forced_condition or (
            ExperimentCondition(arguments.condition)
            if arguments.condition is not None
            else experiment.experiment.condition
        )
        experiment = experiment_with_overrides(
            experiment,
            condition=condition,
            repetitions=arguments.repetitions,
            seed=arguments.seed,
        )
        provider_name = "mock" if arguments.mock_provider else arguments.provider
        if provider_name == "mock":
            provider = create_experiment_mock_provider()
            model_config = ModelConfig(
                provider="mock",
                model_name="deterministic-mock",
                temperature=0.0,
                seed=experiment.experiment.seed,
            )
        else:
            if not os.environ.get("OPENAI_API_KEY", "").strip():
                raise ConfigurationError(
                    "OPENAI_API_KEY is required when --provider openai is selected"
                )
            application = load_application_config(config_path.parent)
            model_config = application.models
            provider = OpenAIProvider(model_config)

        runner = ExperimentRunner(
            provider,
            experiment,
            model_config=model_config,
            configuration_source=str(config_path),
        )
        result = asyncio.run(
            runner.run(
                {
                    "user_message": arguments.message,
                    "learner_code": arguments.learner_code,
                },
                output_path=arguments.output,
            )
        )
    except (ConfigurationError, LLMInvocationError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "condition": condition.value,
                "configuration_hash": result.configuration_hash,
                "observations": len(result.observations),
                "output": str(result.output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser(*, forced_condition: ExperimentCondition | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-topology role-guardrail experiment."
    )
    parser.add_argument("--config", default="config/experiment.yaml")
    if forced_condition is None:
        parser.add_argument(
            "--condition",
            choices=[condition.value for condition in ExperimentCondition],
        )
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument(
        "--message", default="Give me a beginner Python addition problem."
    )
    parser.add_argument("--learner-code")
    parser.add_argument(
        "--provider", choices=("mock", "openai"), default="mock"
    )
    parser.add_argument(
        "--mock-provider",
        action="store_true",
        help="Force the deterministic mock provider without API credentials.",
    )
    return parser
