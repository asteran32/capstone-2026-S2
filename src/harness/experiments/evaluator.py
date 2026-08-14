"""Convert M9 graph results into reproducible observations and CSV records."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.models.schemas import ExperimentObservation


def observation_from_state(
    state: Mapping[str, Any],
    *,
    latency_ms: int,
    model_configuration: Mapping[str, Any],
    configuration_hash: str,
    configuration_source: str,
    token_input: int | None = None,
    token_output: int | None = None,
) -> ExperimentObservation:
    """Create one per-turn observation without re-running the LLM."""

    consistency = state.get("consistency_result") or {}
    indicators = dict(state.get("drift_indicators") or {})
    safety = state.get("safety_result") or {}
    return ExperimentObservation(
        experiment_id=str(state["experiment_id"]),
        session_id=str(state["session_id"]),
        trace_id=str(state["trace_id"]),
        turn_id=int(state["turn_id"]),
        condition=str(state["experiment_condition"]),
        agent=str(state["active_agent"]),
        role_adherence_score=consistency.get("role_adherence_score"),
        drift_score=state.get("drift_score"),
        boundary_violation=indicators.get("role_boundary_violation", 0.0) > 0.0,
        forbidden_action_attempt=(
            indicators.get("forbidden_action_attempt", 0.0) > 0.0
        ),
        repair_count=int(state.get("repair_count", 0)),
        reset_count=int(state.get("reset_count", 0)),
        task_success=bool(
            state.get("final_output") is not None and safety.get("allowed", False)
        ),
        latency_ms=latency_ms,
        token_input=token_input,
        token_output=token_output,
        drift_indicators=indicators,
        model_configuration=dict(model_configuration),
        random_seed=state.get("random_seed"),
        configuration_hash=configuration_hash,
        configuration_source=configuration_source,
    )


def export_observations_csv(
    observations: Sequence[ExperimentObservation], path: str | Path
) -> Path:
    """Export observations with JSON-encoded nested fields."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ExperimentObservation.model_fields)
    with target.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for observation in observations:
            row = observation.model_dump(mode="json")
            for field in ("drift_indicators", "model_configuration"):
                row[field] = json.dumps(
                    row[field], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            writer.writerow(row)
    return target
