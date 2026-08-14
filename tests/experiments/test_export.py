"""M9 observation export and trace-linkage tests."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.config import ModelConfig, load_experiment_config
from harness.experiments.conditions import ExperimentCondition
from harness.experiments.runner import (
    ExperimentRunner,
    create_experiment_mock_provider,
    experiment_with_overrides,
)
from harness.models.provider import GenerationResult, TokenUsage


@dataclass
class _UsageProvider:
    output: dict[str, Any]

    async def generate(self, *_: Any, **__: Any) -> GenerationResult:
        return GenerationResult(
            output=self.output,
            token_usage=TokenUsage(input_tokens=11, output_tokens=7, total_tokens=18),
        )


async def test_export_contains_condition_model_trace_and_raw_indicators(
    tmp_path: Path,
) -> None:
    config = experiment_with_overrides(
        load_experiment_config(), condition=ExperimentCondition.L2
    )
    config.experiment.id = "m9-export"
    config.logging.trace_dir = str(tmp_path / "traces")
    model = ModelConfig(
        provider="mock", model_name="export-mock", temperature=0.0, seed=42
    )
    output = tmp_path / "observations.csv"

    result = await ExperimentRunner(
        create_experiment_mock_provider(), config, model_config=model
    ).run({"user_message": "Give me a new problem."}, output_path=output)

    with output.open(encoding="utf-8", newline="") as exported:
        rows = list(csv.DictReader(exported))
    assert len(rows) == 1
    row = rows[0]
    assert row["condition"] == "L2"
    assert row["trace_id"] == result.observations[0].trace_id
    assert json.loads(row["model_configuration"])["model_name"] == "export-mock"
    assert "role_boundary_violation" in json.loads(row["drift_indicators"])
    assert row["configuration_hash"] == result.configuration_hash
    assert row["configuration_source"] == "config/experiment.yaml"


async def test_export_captures_provider_token_usage(tmp_path: Path) -> None:
    config = load_experiment_config()
    config.experiment.id = "m9-usage"
    config.logging.trace_dir = str(tmp_path / "traces")
    output = tmp_path / "usage.csv"
    provider = _UsageProvider(
        {
            "problem_id": "usage-problem",
            "title": "Add",
            "statement": "Add two values.",
            "constraints": [],
            "examples": [{"input": "1 2", "output": "3"}],
            "reference_solution": None,
            "test_specification": [],
            "difficulty": "beginner",
        }
    )

    result = await ExperimentRunner(
        provider,
        config,
        model_config=ModelConfig(
            provider="mock", model_name="usage-mock", temperature=0.0
        ),
    ).run({"user_message": "Give me a new problem."}, output_path=output)

    assert result.observations[0].token_input == 11
    assert result.observations[0].token_output == 7
