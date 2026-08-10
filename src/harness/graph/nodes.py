"""M4 graph nodes for session setup, agent invocation, and finalization."""

from typing import Any, Callable, Mapping
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from harness.agents.base import BaseAgent
from harness.graph.state import AgentName, HarnessState
from harness.guardrails.consistency import ConsistencyMonitor
from harness.graph.routing import classify_intent, route_for_intent
from harness.memory.projector import ContextProjector


def initialize_session(
    state: HarnessState, config: RunnableConfig | None = None
) -> HarnessState:
    """Initialize stable session/thread identity and a fresh turn trace."""

    configurable = {} if config is None else config.get("configurable", {})
    thread_id = state.get("thread_id") or configurable.get("thread_id") or str(uuid4())
    return {
        "session_id": state.get("session_id") or thread_id,
        "thread_id": thread_id,
        "trace_id": str(uuid4()),
        "turn_id": int(state.get("turn_id", 0)) + 1,
    }


def classify_intent_node(state: HarnessState) -> HarnessState:
    """Store the bounded classification result in graph state."""

    return {"intent": classify_intent(state)}


def route_agent_node(state: HarnessState) -> HarnessState:
    """Record the deterministic selected role agent in graph state."""

    return {"active_agent": route_for_intent(state.get("intent", "unknown"))}


def select_agent_edge(state: HarnessState) -> AgentName:
    """Return the already-recorded graph node name for conditional routing."""

    return state["active_agent"] or route_for_intent(state.get("intent", "unknown"))


def create_agent_node(
    agent: BaseAgent[Any],
    agent_id: AgentName,
    *,
    projector: ContextProjector | None = None,
) -> Callable[[HarnessState], Any]:
    """Create one graph node that invokes exactly one role agent."""

    context_projector = projector or ContextProjector()

    async def invoke_agent(state: HarnessState) -> HarnessState:
        projected_context = context_projector.project(state, agent_id)
        output = await agent.invoke(
            _task_context_for_agent(state, agent_id), projected_context
        )
        return {"candidate_output": output.model_dump(mode="json")}

    return invoke_agent


def finalize_response(state: HarnessState) -> HarnessState:
    """Expose the M4 candidate output without applying later guardrails."""

    return {"final_output": state.get("candidate_output")}


def create_consistency_monitor_node(
    monitor: ConsistencyMonitor,
) -> Callable[[HarnessState], HarnessState]:
    """Create the Layer 2 graph node without mutating candidate output."""

    def consistency_monitor(state: HarnessState) -> HarnessState:
        agent_id = state["active_agent"]
        if agent_id is None:
            raise ValueError("Consistency monitor requires an active agent")
        result = monitor.evaluate(agent_id, state.get("candidate_output"))
        observation = {
            "agent_id": agent_id,
            "turn_id": state.get("turn_id", 0),
            "drift_score": 1.0 - result.role_adherence_score,
            "indicators": result.drift_signals,
            "violations": result.violations,
            "schema_errors": result.schema_errors,
            "action": result.recommended_action,
            "evaluator_version": result.evaluator_version,
        }
        return {
            "consistency_result": result.model_dump(mode="json"),
            "drift_score": observation["drift_score"],
            "drift_indicators": result.drift_signals,
            "drift_history": [observation],
            "guardrail_action": result.recommended_action,
        }

    return consistency_monitor


def select_consistency_edge(state: HarnessState) -> str:
    """Select M5 PASS or non-mutating REPAIR continuation."""

    return "repair" if state.get("guardrail_action") == "repair" else "pass"


def repair_pending(_: HarnessState) -> HarnessState:
    """Preserve the M5 repair branch until M6 implements output repair."""

    return {}


def _task_context_for_agent(state: HarnessState, agent_id: AgentName) -> dict[str, Any]:
    """Adapt shared M4 state into the current agent's task payload."""

    if agent_id == "problem_designer":
        return {
            "learner_level": state.get("learner_level", "beginner"),
            "topic": state.get("user_message", "general programming"),
            "difficulty": "appropriate",
            "constraints": [],
        }
    if agent_id == "code_reviewer":
        return {
            "problem": state.get("current_problem") or {"request": state.get("user_message", "")},
            "learner_code": state.get("learner_code") or "",
            "learner_level": state.get("learner_level", "beginner"),
            "test_results": state.get("latest_test_results", []),
        }
    return {
        "source_code": state.get("learner_code") or "",
        "available_test_cases": state.get("latest_test_results", []),
    }
