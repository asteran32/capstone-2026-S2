"""M9 command-line entry-point tests using only the mock provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.experiments.cli import run_experiment_cli
from harness.experiments.conditions import ExperimentCondition


def _config(tmp_path: Path) -> Path:
    text = Path("config/experiment.yaml").read_text(encoding="utf-8")
    path = tmp_path / "experiment.yaml"
    path.write_text(
        text.replace("trace_dir: data/traces", f"trace_dir: {tmp_path / 'traces'}"),
        encoding="utf-8",
    )
    return path


def test_baseline_cli_forces_baseline_and_exports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "baseline.csv"

    exit_code = run_experiment_cli(
        [
            "--config",
            str(_config(tmp_path)),
            "--repetitions",
            "2",
            "--output",
            str(output),
            "--mock-provider",
        ],
        forced_condition=ExperimentCondition.BASELINE,
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["condition"] == "BASELINE"
    assert summary["observations"] == 2
    assert output.exists()


def test_ablation_cli_accepts_partial_condition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "ablation.csv"

    exit_code = run_experiment_cli(
        [
            "--config",
            str(_config(tmp_path)),
            "--condition",
            "L2_L3",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["condition"] == "L2_L3"
    assert output.exists()
