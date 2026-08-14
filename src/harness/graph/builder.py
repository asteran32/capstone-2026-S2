"""Construction of the executable graph and its M9-configurable infrastructure."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from harness.agents import CodeReviewAgent, ProblemDesignAgent, TestRunnerAgent
from harness.config import (
    GuardrailsConfig,
    ModelConfig,
    PersistenceConfig,
    SafetyConfig,
    load_guardrails_config,
    load_safety_config,
)
from harness.graph.nodes import (
    classify_intent_node,
    create_agent_node,
    create_consistency_monitor_node,
    create_intervention_selector,
    create_repair_node,
    create_reset_node,
    finalize_response,
    initialize_session,
    route_agent_node,
    select_agent_edge,
    create_safety_validator_node,
    intervention_exhausted,
)
from harness.graph.state import HarnessState
from harness.graph.persistence import (
    ManagedAsyncGraph,
    PersistenceError,
    create_async_checkpointer,
    create_checkpointer,
)
from harness.guardrails.consistency import ConsistencyMonitor
from harness.guardrails.contracts import RoleContractLoader
from harness.guardrails.repair import RepairManager
from harness.guardrails.reset import ResetPolicy
from harness.models.provider import LLMProvider
from harness.observability.logger import JSONLTraceLogger, TraceMetadata, TraceRecorder
from harness.observability.metrics import MetricsCollector
from harness.safety.validator import SafetyValidator


_LOGGER = logging.getLogger(__name__)


def build_mvp_graph(
    provider: LLMProvider,
    *,
    checkpointer: Any | None = None,
    persistence: PersistenceConfig | None = None,
    guardrails: GuardrailsConfig | None = None,
    safety: SafetyConfig | None = None,
    trace_logger: JSONLTraceLogger | None = None,
    metrics_collector: MetricsCollector | None = None,
    model_config: ModelConfig | None = None,
    trace_dir: str = "data/traces",
) -> Any:
    """Compile the fixed-topology graph with injected M9 layer configuration."""

    if checkpointer is None and persistence is not None and persistence.backend == "sqlite":
        message = "Use await build_mvp_graph_async(..., persistence=...) for SQLite persistence"
        _LOGGER.error(message)
        raise PersistenceError(message)
    guardrail_config = guardrails or load_guardrails_config()
    safety_config = safety or load_safety_config()
    reset_policy = ResetPolicy(guardrail_config)
    contract_loader = RoleContractLoader()
    contract_versions = (
        {
            agent_id: contract_loader.contract_version(agent_id)
            for agent_id in ("problem_designer", "code_reviewer", "test_runner")
        }
        if guardrail_config.layer1.enabled
        else {}
    )
    recorder = TraceRecorder(
        trace_logger or JSONLTraceLogger(trace_dir),
        metrics_collector or MetricsCollector(),
        TraceMetadata(
            model_configuration=_model_configuration(provider, model_config),
            contract_versions=contract_versions,
        ),
    )
    builder = StateGraph(HarnessState)
    builder.add_node("initialize_session", lambda state, config: initialize_session(state, config, recorder=recorder))
    builder.add_node("classify_intent", lambda state: classify_intent_node(state, recorder=recorder))
    builder.add_node("route_agent", lambda state: route_agent_node(state, recorder=recorder))
    builder.add_node(
        "problem_designer",
        create_agent_node(
            ProblemDesignAgent(
                provider, role_contract_enabled=guardrail_config.layer1.enabled
            ),
            "problem_designer",
            recorder=recorder,
        ),
    )
    builder.add_node(
        "code_reviewer",
        create_agent_node(
            CodeReviewAgent(
                provider, role_contract_enabled=guardrail_config.layer1.enabled
            ),
            "code_reviewer",
            recorder=recorder,
        ),
    )
    builder.add_node(
        "test_runner",
        create_agent_node(
            TestRunnerAgent(
                provider, role_contract_enabled=guardrail_config.layer1.enabled
            ),
            "test_runner",
            recorder=recorder,
        ),
    )
    builder.add_node(
        "consistency_monitor",
        create_consistency_monitor_node(ConsistencyMonitor(guardrail_config), recorder=recorder),
    )
    builder.add_node(
        "repair_output",
        create_repair_node(
            RepairManager(
                provider, role_contract_enabled=guardrail_config.layer1.enabled
            ),
            recorder=recorder,
        ),
    )
    builder.add_node("reset_agent", create_reset_node(reset_policy, recorder=recorder))
    builder.add_node(
        "intervention_exhausted", lambda state: intervention_exhausted(state, recorder=recorder)
    )
    builder.add_node(
        "safety_validator",
        create_safety_validator_node(SafetyValidator(safety_config), recorder=recorder),
    )
    builder.add_node("finalize_response", lambda state: finalize_response(state, recorder=recorder))

    builder.add_edge(START, "initialize_session")
    builder.add_edge("initialize_session", "classify_intent")
    builder.add_edge("classify_intent", "route_agent")
    builder.add_conditional_edges(
        "route_agent",
        select_agent_edge,
        {
            "problem_designer": "problem_designer",
            "code_reviewer": "code_reviewer",
            "test_runner": "test_runner",
        },
    )
    for agent_node in ("problem_designer", "code_reviewer", "test_runner"):
        builder.add_edge(agent_node, "consistency_monitor")
    builder.add_conditional_edges(
        "consistency_monitor",
        create_intervention_selector(reset_policy),
        {
            "pass": "safety_validator",
            "repair": "repair_output",
            "reset": "reset_agent",
            "block": "intervention_exhausted",
        },
    )
    builder.add_edge("repair_output", "consistency_monitor")
    builder.add_conditional_edges(
        "reset_agent",
        select_agent_edge,
        {
            "problem_designer": "problem_designer",
            "code_reviewer": "code_reviewer",
            "test_runner": "test_runner",
        },
    )
    builder.add_edge("intervention_exhausted", "safety_validator")
    builder.add_edge("safety_validator", "finalize_response")
    builder.add_edge("finalize_response", END)

    return builder.compile(
        checkpointer=checkpointer if checkpointer is not None else create_checkpointer(persistence)
    )


async def build_mvp_graph_async(
    provider: LLMProvider,
    *,
    checkpointer: Any | None = None,
    persistence: PersistenceConfig | None = None,
    guardrails: GuardrailsConfig | None = None,
    safety: SafetyConfig | None = None,
    trace_logger: JSONLTraceLogger | None = None,
    metrics_collector: MetricsCollector | None = None,
    model_config: ModelConfig | None = None,
    trace_dir: str = "data/traces",
) -> Any:
    """Compose the same graph with an async SQLite saver when configured."""

    resolved_checkpointer = checkpointer
    owns_checkpointer = resolved_checkpointer is None
    if resolved_checkpointer is None:
        resolved_persistence = persistence or PersistenceConfig()
        resolved_checkpointer = await create_async_checkpointer(resolved_persistence)
    graph = build_mvp_graph(
        provider,
        checkpointer=resolved_checkpointer,
        guardrails=guardrails,
        safety=safety,
        trace_logger=trace_logger,
        metrics_collector=metrics_collector,
        model_config=model_config,
        trace_dir=trace_dir,
    )
    if owns_checkpointer and getattr(resolved_checkpointer, "conn", None) is not None:
        return ManagedAsyncGraph(graph, resolved_checkpointer)
    return graph


def _model_configuration(
    provider: LLMProvider, model_config: ModelConfig | None
) -> dict[str, Any]:
    """Record supplied configuration, or safe provider identity when unavailable."""

    if model_config is not None:
        return model_config.model_dump(mode="json")
    provider_config = getattr(provider, "_config", None)
    if isinstance(provider_config, ModelConfig):
        return provider_config.model_dump(mode="json")
    return {"provider": type(provider).__name__}
