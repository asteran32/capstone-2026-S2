"""M9 baseline, ablation, reproducibility, and topology tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import ExperimentConfig, ModelConfig, load_experiment_config
from harness.experiments.conditions import ExperimentCondition
from harness.experiments.runner import (
    ExperimentRunner,
    create_experiment_mock_provider,
    experiment_with_overrides,
)
from harness.models.provider import MockProvider


class _StepClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        current = self._value
        self._value += 0.01
        return current


def _config(
    tmp_path: Path,
    condition: ExperimentCondition,
    *,
    repetitions: int = 1,
    seed: int = 42,
) -> ExperimentConfig:
    config = experiment_with_overrides(
        load_experiment_config(),
        condition=condition,
        repetitions=repetitions,
        seed=seed,
    )
    config.experiment.id = "m9-test"
    config.logging.trace_dir = str(tmp_path / "traces")
    return config


def _model(seed: int = 42) -> ModelConfig:
    return ModelConfig(
        provider="mock",
        model_name="deterministic-mock",
        temperature=0.0,
        seed=seed,
    )


@pytest.mark.parametrize(
    "condition",
    [
        ExperimentCondition.BASELINE,
        ExperimentCondition.FULL,
        ExperimentCondition.L1_L2,
    ],
)
async def test_required_baseline_full_and_partial_conditions_execute(
    tmp_path: Path, condition: ExperimentCondition
) -> None:
    runner = ExperimentRunner(
        create_experiment_mock_provider(),
        _config(tmp_path, condition),
        model_config=_model(),
        clock=_StepClock(),
    )

    result = await runner.run(
        {"user_message": "Give me a new problem."},
        output_path=tmp_path / f"{condition.value}.csv",
    )

    assert len(result.observations) == 1
    assert result.observations[0].condition == condition.value
    assert result.output_path.exists()


async def test_baseline_measures_drift_without_intervention_or_contract_injection(
    tmp_path: Path,
) -> None:
    provider = MockProvider(
        {
            "CodeReviewOutput": {
                "correctness_analysis": "I will execute_code now.",
                "detected_issues": [],
                "hints": [],
                "pedagogical_feedback": "Boundary test.",
                "confidence": 1.0,
            }
        }
    )
    runner = ExperimentRunner(
        provider,
        _config(tmp_path, ExperimentCondition.BASELINE),
        model_config=_model(),
        clock=_StepClock(),
    )

    result = await runner.run(
        {"user_message": "Review my code.", "learner_code": "print(1)"},
        output_path=tmp_path / "baseline.csv",
    )

    observation = result.observations[0]
    assert observation.forbidden_action_attempt is True
    assert observation.repair_count == 0
    assert observation.reset_count == 0
    assert observation.task_success is True
    assert len(provider.calls) == 1
    assert "ROLE_CONTRACT" not in provider.calls[0]["messages"][0]["content"]


async def test_full_condition_injects_role_contract(tmp_path: Path) -> None:
    provider = create_experiment_mock_provider()
    runner = ExperimentRunner(
        provider,
        _config(tmp_path, ExperimentCondition.FULL),
        model_config=_model(),
        clock=_StepClock(),
    )

    await runner.run(
        {"user_message": "Give me a new problem."},
        output_path=tmp_path / "full.csv",
    )

    assert "ROLE_CONTRACT" in provider.calls[0]["messages"][0]["content"]


async def test_topology_is_equal_across_every_condition(tmp_path: Path) -> None:
    signatures = set()
    for condition in ExperimentCondition:
        runner = ExperimentRunner(
            create_experiment_mock_provider(),
            _config(tmp_path, condition),
            model_config=_model(),
            clock=_StepClock(),
        )
        result = await runner.run(
            {"user_message": "Give me a new problem."},
            output_path=tmp_path / f"topology-{condition.value}.csv",
        )
        signatures.add(result.topology_signature)

    assert len(signatures) == 1


async def test_same_seed_and_mock_provider_are_deterministic(tmp_path: Path) -> None:
    config = _config(
        tmp_path, ExperimentCondition.L1_L2, repetitions=2, seed=17
    )
    first = await ExperimentRunner(
        create_experiment_mock_provider(),
        config,
        model_config=_model(17),
        clock=_StepClock(),
    ).run(
        {"user_message": "Give me a new problem."},
        output_path=tmp_path / "first.csv",
    )
    second = await ExperimentRunner(
        create_experiment_mock_provider(),
        config,
        model_config=_model(17),
        clock=_StepClock(),
    ).run(
        {"user_message": "Give me a new problem."},
        output_path=tmp_path / "second.csv",
    )

    assert [item.model_dump() for item in first.observations] == [
        item.model_dump() for item in second.observations
    ]
    assert first.configuration_hash == second.configuration_hash
