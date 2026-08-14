"""M9 condition-controlled experiment execution with fixed graph topology."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.config import (
    ExperimentConfig,
    GuardrailsConfig,
    ModelConfig,
    SafetyConfig,
    load_guardrails_config,
    load_safety_config,
)
from harness.experiments.conditions import ExperimentCondition
from harness.experiments.evaluator import (
    export_observations_csv,
    observation_from_state,
)
from harness.graph import build_mvp_graph_async
from harness.models.provider import GenerationResult, LLMProvider, MockProvider
from harness.models.schemas import ExperimentObservation
from harness.observability.logger import JSONLTraceLogger
from harness.observability.metrics import MetricsCollector


TopologySignature = tuple[
    tuple[str, ...],
    tuple[tuple[str, str, bool], ...],
]


@dataclass(frozen=True)
class ExperimentRunResult:
    """Observations and reproducibility artifacts produced by one runner call."""

    observations: tuple[ExperimentObservation, ...]
    output_path: Path
    topology_signature: TopologySignature
    configuration_hash: str


@dataclass(frozen=True)
class _UsageSnapshot:
    input_tokens: int
    output_tokens: int
    observed: bool


class _UsageTrackingProvider:
    """Track provider-reported usage without adding state to role agents."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self._input_tokens = 0
        self._output_tokens = 0
        self._observed = False

    async def generate(self, *args: Any, **kwargs: Any) -> GenerationResult:
        result = await self._provider.generate(*args, **kwargs)
        usage = result.token_usage
        if usage is not None:
            self._observed = True
            self._input_tokens += usage.input_tokens or 0
            self._output_tokens += usage.output_tokens or 0
        return result

    def snapshot(self) -> _UsageSnapshot:
        return _UsageSnapshot(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            observed=self._observed,
        )

    def delta(self, before: _UsageSnapshot) -> tuple[int | None, int | None]:
        if not self._observed or self.snapshot() == before:
            return None, None
        return (
            self._input_tokens - before.input_tokens,
            self._output_tokens - before.output_tokens,
        )


class ExperimentRunner:
    """Execute repeated single-turn sessions under one ablation condition."""

    def __init__(
        self,
        provider: LLMProvider,
        experiment: ExperimentConfig,
        *,
        base_guardrails: GuardrailsConfig | None = None,
        safety: SafetyConfig | None = None,
        model_config: ModelConfig | None = None,
        configuration_source: str = "config/experiment.yaml",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._provider = provider
        self._experiment = experiment
        self._base_guardrails = base_guardrails or load_guardrails_config()
        self._safety = safety or load_safety_config()
        self._model_config = model_config or _model_configuration(provider)
        self._configuration_source = configuration_source
        self._clock = clock

    async def run(
        self,
        input_state: Mapping[str, Any],
        *,
        output_path: str | Path | None = None,
    ) -> ExperimentRunResult:
        """Run configured repetitions and export their per-turn observations."""

        metadata = self._experiment.experiment
        condition = metadata.condition
        guardrails = guardrails_for_condition(self._base_guardrails, condition)
        experiment_id = metadata.id or _default_experiment_id(condition, metadata.seed)
        configuration_hash = experiment_configuration_hash(
            self._experiment, guardrails, self._safety, self._model_config
        )
        collector = MetricsCollector()
        tracking_provider = _UsageTrackingProvider(self._provider)
        trace_dir = (
            Path(self._experiment.logging.trace_dir)
            / experiment_id
            / condition.value.lower()
        )
        graph = await build_mvp_graph_async(
            tracking_provider,
            persistence=self._experiment.persistence,
            guardrails=guardrails,
            safety=self._safety,
            trace_logger=JSONLTraceLogger(trace_dir),
            metrics_collector=collector,
            model_config=self._model_config,
        )
        topology = graph_topology_signature(graph)
        observations: list[ExperimentObservation] = []
        try:
            for repetition in range(metadata.repetitions):
                session_id = f"{experiment_id}-{condition.value.lower()}-{repetition + 1:04d}"
                random_seed = (
                    None if metadata.seed is None else metadata.seed + repetition
                )
                if random_seed is not None:
                    random.seed(random_seed)
                state = {
                    **dict(input_state),
                    "session_id": session_id,
                    "thread_id": session_id,
                    "trace_id": f"{session_id}-trace",
                    "experiment_id": experiment_id,
                    "experiment_condition": condition.value,
                    "random_seed": random_seed,
                }
                started = self._clock()
                usage_before = tracking_provider.snapshot()
                result = await graph.ainvoke(
                    state,
                    {"configurable": {"thread_id": session_id}},
                )
                latency_ms = max(0, int((self._clock() - started) * 1000))
                token_input, token_output = tracking_provider.delta(usage_before)
                observations.append(
                    observation_from_state(
                        result,
                        latency_ms=latency_ms,
                        model_configuration=self._model_config.model_dump(mode="json"),
                        configuration_hash=configuration_hash,
                        configuration_source=self._configuration_source,
                        token_input=token_input,
                        token_output=token_output,
                    )
                )
        finally:
            close = getattr(graph, "aclose", None)
            if close is not None:
                await close()

        target = Path(output_path) if output_path is not None else Path(
            "data/experiments"
        ) / f"{experiment_id}-{condition.value.lower()}.csv"
        export_observations_csv(observations, target)
        return ExperimentRunResult(
            observations=tuple(observations),
            output_path=target,
            topology_signature=topology,
            configuration_hash=configuration_hash,
        )


def guardrails_for_condition(
    base: GuardrailsConfig, condition: ExperimentCondition
) -> GuardrailsConfig:
    """Derive layer activation without editing source or the base configuration."""

    data = base.model_dump(mode="json")
    flags = condition.layer_flags
    for layer_name, enabled in flags.items():
        data[layer_name]["enabled"] = enabled
    data["passive_measurement"] = not flags["layer2"]
    return GuardrailsConfig.model_validate(data)


def experiment_with_overrides(
    experiment: ExperimentConfig,
    *,
    condition: ExperimentCondition | None = None,
    repetitions: int | None = None,
    seed: int | None = None,
) -> ExperimentConfig:
    """Return a revalidated configuration for CLI overrides."""

    data = experiment.model_dump(mode="json")
    resolved_condition = condition or experiment.experiment.condition
    data["experiment"]["condition"] = resolved_condition.value
    if repetitions is not None:
        data["experiment"]["repetitions"] = repetitions
    if seed is not None:
        data["experiment"]["seed"] = seed
    for layer_name, enabled in resolved_condition.layer_flags.items():
        data["guardrails"][layer_name]["enabled"] = enabled
    return ExperimentConfig.model_validate(data)


def experiment_configuration_hash(
    experiment: ExperimentConfig,
    guardrails: GuardrailsConfig,
    safety: SafetyConfig,
    model: ModelConfig,
) -> str:
    """Hash all configuration that can affect an M9 experiment run."""

    payload = {
        "experiment": experiment.model_dump(mode="json"),
        "resolved_guardrails": guardrails.model_dump(mode="json"),
        "safety": safety.model_dump(mode="json"),
        "model": model.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def graph_topology_signature(graph: Any) -> TopologySignature:
    """Return a condition-independent representation of graph nodes and edges."""

    representation = graph.get_graph()
    nodes = tuple(sorted(representation.nodes))
    edges = tuple(
        sorted(
            (edge.source, edge.target, bool(edge.conditional))
            for edge in representation.edges
        )
    )
    return nodes, edges


def create_experiment_mock_provider() -> MockProvider:
    """Return deterministic fixtures covering all three fixed role routes."""

    return MockProvider(
        {
            "ProblemDesignOutput": {
                "problem_id": "experiment-problem",
                "title": "Add two numbers",
                "statement": "Return the sum of two integers.",
                "constraints": ["Use Python."],
                "examples": [{"input": "1 2", "output": "3"}],
                "reference_solution": None,
                "test_specification": [{"name": "basic"}],
                "difficulty": "beginner",
            },
            "CodeReviewOutput": {
                "correctness_analysis": "Deterministic experiment review.",
                "detected_issues": [],
                "hints": ["Trace one example."],
                "pedagogical_feedback": "Check the expected output.",
                "confidence": 1.0,
            },
            "ExecutionRequest": {
                "language": "python",
                "source_code": "print('experiment')",
                "test_cases": [],
                "timeout_seconds": 5,
            },
        }
    )


def _model_configuration(provider: LLMProvider) -> ModelConfig:
    configured = getattr(provider, "_config", None)
    if isinstance(configured, ModelConfig):
        return configured
    return ModelConfig(
        provider=type(provider).__name__,
        model_name="deterministic-mock",
        temperature=0.0,
        seed=0,
    )


def _default_experiment_id(
    condition: ExperimentCondition, seed: int | None
) -> str:
    return f"exp-{condition.value.lower()}-{0 if seed is None else seed}"
