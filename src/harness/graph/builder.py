"""M4 construction of the executable LangGraph workflow."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from harness.agents import CodeReviewAgent, ProblemDesignAgent, TestRunnerAgent
from harness.config import GuardrailsConfig, load_guardrails_config
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
    intervention_exhausted,
)
from harness.graph.state import HarnessState
from harness.guardrails.consistency import ConsistencyMonitor
from harness.guardrails.repair import RepairManager
from harness.guardrails.reset import ResetPolicy
from harness.models.provider import LLMProvider


def build_mvp_graph(
    provider: LLMProvider,
    *,
    checkpointer: Any | None = None,
    guardrails: GuardrailsConfig | None = None,
) -> Any:
    """Compile the M6 graph with bounded repair and Layer 3 reset recovery."""

    guardrail_config = guardrails or load_guardrails_config()
    reset_policy = ResetPolicy(guardrail_config)
    builder = StateGraph(HarnessState)
    builder.add_node("initialize_session", initialize_session)
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("route_agent", route_agent_node)
    builder.add_node(
        "problem_designer", create_agent_node(ProblemDesignAgent(provider), "problem_designer")
    )
    builder.add_node(
        "code_reviewer", create_agent_node(CodeReviewAgent(provider), "code_reviewer")
    )
    builder.add_node("test_runner", create_agent_node(TestRunnerAgent(provider), "test_runner"))
    builder.add_node(
        "consistency_monitor", create_consistency_monitor_node(ConsistencyMonitor(guardrail_config))
    )
    builder.add_node("repair_output", create_repair_node(RepairManager(provider)))
    builder.add_node("reset_agent", create_reset_node(reset_policy))
    builder.add_node("intervention_exhausted", intervention_exhausted)
    builder.add_node("finalize_response", finalize_response)

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
            "pass": "finalize_response",
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
    builder.add_edge("intervention_exhausted", "finalize_response")
    builder.add_edge("finalize_response", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
