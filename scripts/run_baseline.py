"""Run the M9 BASELINE condition with no role-guardrail intervention."""

from harness.experiments.cli import run_experiment_cli
from harness.experiments.conditions import ExperimentCondition


if __name__ == "__main__":
    raise SystemExit(
        run_experiment_cli(forced_condition=ExperimentCondition.BASELINE)
    )
