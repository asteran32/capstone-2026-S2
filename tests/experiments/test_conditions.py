"""M9 condition mapping and typed configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import ConfigurationError, load_experiment_config
from harness.experiments.conditions import ExperimentCondition


def test_all_required_conditions_have_exact_layer_mappings() -> None:
    assert {condition.value for condition in ExperimentCondition} == {
        "BASELINE",
        "L1",
        "L2",
        "L3",
        "L1_L2",
        "L1_L3",
        "L2_L3",
        "FULL",
    }
    assert ExperimentCondition.BASELINE.layer_flags == {
        "layer1": False,
        "layer2": False,
        "layer3": False,
    }
    assert ExperimentCondition.L1_L3.layer_flags == {
        "layer1": True,
        "layer2": False,
        "layer3": True,
    }
    assert ExperimentCondition.FULL.active_layers == frozenset({1, 2, 3})


def test_checked_in_experiment_configuration_is_typed() -> None:
    config = load_experiment_config()

    assert config.experiment.condition is ExperimentCondition.FULL
    assert config.experiment.repetitions == 1
    assert config.experiment.seed == 42


def test_condition_and_layer_flag_mismatch_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """experiment:
  id: mismatch
  condition: BASELINE
  repetitions: 1
  seed: 42
guardrails:
  layer1: {enabled: true}
  layer2: {enabled: true}
  layer3: {enabled: true}
logging:
  save_raw_outputs: true
  save_state_snapshots: false
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="BASELINE requires layer flags"):
        load_experiment_config(path)


def test_unknown_condition_fails_typed_validation(tmp_path: Path) -> None:
    text = Path("config/experiment.yaml").read_text(encoding="utf-8")
    path = tmp_path / "experiment.yaml"
    path.write_text(text.replace("condition: FULL", "condition: UNKNOWN"), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="experiment.yaml"):
        load_experiment_config(path)
