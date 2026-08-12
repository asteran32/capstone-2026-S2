"""M4 graph nodes for session setup, agent invocation, and finalization."""

from typing import Any, Callable, Mapping
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from harness.agents.base import BaseAgent
from harness.graph.state import AgentName, HarnessState
from harness.guardrails.consistency import ConsistencyMonitor
from harness.guardrails.repair import RepairManager
from harness.guardrails.reset import ResetPolicy
from harness.graph.routing import classify_intent, route_for_intent
from harness.memory.context import AgentContextMemory
from harness.memory.projector import ContextProjector
from harness.observability.events import TraceEventType
from harness.observability.logger import TraceRecorder
from harness.safety.validator import SafetyValidator


def initialize_session(
    state: HarnessState,
    config: RunnableConfig | None = None,
    *,
    recorder: TraceRecorder | None = None,
) -> HarnessState:
    """Initialize stable session/thread identity and a fresh turn trace."""

    configurable = {} if config is None else config.get("configurable", {})
    thread_id = state.get("thread_id") or configurable.get("thread_id") or str(uuid4())
    update: HarnessState = {
        "session_id": state.get("session_id") or thread_id,
        "thread_id": thread_id,
        "trace_id": str(uuid4()),
        "turn_id": int(state.get("turn_id", 0)) + 1,
        "repair_count": int(state.get("repair_count", 0)),
        "reset_count": int(state.get("reset_count", 0)),
    }
    if recorder is not None:
        event_state = {**state, **update}
        recorder.emit(TraceEventType.SESSION_STARTED, event_state)
        recorder.emit(
            TraceEventType.USER_MESSAGE_RECEIVED,
            event_state,
            metadata={"user_message": state.get("user_message", "")},
        )
    return update


def classify_intent_node(
    state: HarnessState, *, recorder: TraceRecorder | None = None
) -> HarnessState:
    """Store the bounded classification result in graph state."""

    update: HarnessState = {"intent": classify_intent(state)}
    if recorder is not None:
        recorder.emit(
            TraceEventType.INTENT_CLASSIFIED,
            {**state, **update},
            metadata={"intent": update["intent"]},
        )
    return update


def route_agent_node(
    state: HarnessState, *, recorder: TraceRecorder | None = None
) -> HarnessState:
    """Record the deterministic selected role agent in graph state."""

    update: HarnessState = {"active_agent": route_for_intent(state.get("intent", "unknown"))}
    if recorder is not None:
        recorder.emit(
            TraceEventType.AGENT_SELECTED,
            {**state, **update},
            agent=update["active_agent"],
        )
    return update


def select_agent_edge(state: HarnessState) -> AgentName:
    """Return the already-recorded graph node name for conditional routing."""

    return state["active_agent"] or route_for_intent(state.get("intent", "unknown"))


def create_agent_node(
    agent: BaseAgent[Any],
    agent_id: AgentName,
    *,
    projector: ContextProjector | None = None,
    recorder: TraceRecorder | None = None,
) -> Callable[[HarnessState], Any]:
    """Create one graph node that invokes exactly one role agent."""

    context_projector = projector or ContextProjector()

    async def invoke_agent(state: HarnessState) -> HarnessState:
        if recorder is not None:
            recorder.emit(TraceEventType.AGENT_INVOKED, state, agent=agent_id)
        contexts = state.get("agent_contexts", {})
        agent_context = AgentContextMemory(
            agent_id=agent_id, history=contexts.get(agent_id, [])
        )
        projected_context = context_projector.project(
            state, agent_id, agent_context=agent_context
        )
        output = await agent.invoke(
            _task_context_for_agent(state, agent_id), projected_context
        )
        candidate = output.model_dump(mode="json")
        updated_contexts = {key: list(value) for key, value in contexts.items()}
        updated_contexts[agent_id] = [
            *updated_contexts.get(agent_id, []),
            {"candidate_output": candidate},
        ]
        update: HarnessState = {
            "candidate_output": candidate,
            "candidate_output_id": str(uuid4()),
            "agent_contexts": updated_contexts,
        }
        if recorder is not None:
            recorder.emit(
                TraceEventType.AGENT_OUTPUT_RECEIVED,
                {**state, **update},
                agent=agent_id,
                candidate_output=candidate,
                candidate_output_id=update["candidate_output_id"],
                metadata={"output_source": "agent"},
            )
        return update

    return invoke_agent


def finalize_response(
    state: HarnessState, *, recorder: TraceRecorder | None = None
) -> HarnessState:
    """Deliver a candidate only after any applicable safety validation passes."""

    safety_result = state.get("safety_result")
    final_output = (
        None
        if safety_result is not None and not safety_result.get("allowed", False)
        else state.get("candidate_output")
    )
    update: HarnessState = {"final_output": final_output}
    if recorder is not None:
        event_state = {**state, **update}
        recorder.emit(
            TraceEventType.OUTPUT_DELIVERED,
            event_state,
            final_output=final_output,
        )
        recorder.emit(TraceEventType.SESSION_ENDED, event_state)
    return update


def create_consistency_monitor_node(
    monitor: ConsistencyMonitor,
    *,
    recorder: TraceRecorder | None = None,
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
            "phase": state.get("monitor_phase", "initial"),
        }
        update: HarnessState = {
            "consistency_result": result.model_dump(mode="json"),
            "drift_score": observation["drift_score"],
            "drift_indicators": result.drift_signals,
            "drift_history": [observation],
            "guardrail_action": result.recommended_action,
            "monitor_phase": "initial",
        }
        if recorder is not None:
            event_state = {**state, **update}
            recorder.emit(
                TraceEventType.CONSISTENCY_CHECKED,
                event_state,
                evaluator_version=result.evaluator_version,
            )
            recorder.emit(
                TraceEventType.DRIFT_SCORE_UPDATED,
                event_state,
                evaluator_version=result.evaluator_version,
            )
        return update

    return consistency_monitor


def create_intervention_selector(
    reset_policy: ResetPolicy,
) -> Callable[[HarnessState], str]:
    """Create bounded PASS/REPAIR/RESET graph routing for M6."""

    def select_intervention(state: HarnessState) -> str:
        if state.get("guardrail_action") == "pass":
            return "pass"
        decision = reset_policy.decide(state)
        if decision.required:
            return "reset" if state.get("reset_count", 0) < reset_policy.max_resets else "block"
        if state.get("guardrail_action") == "repair" and state.get(
            "repair_count", 0
        ) < reset_policy.max_repairs:
            return "repair"
        return "block"

    return select_intervention


def create_repair_node(
    repair_manager: RepairManager,
    *,
    projector: ContextProjector | None = None,
    recorder: TraceRecorder | None = None,
) -> Callable[[HarnessState], Any]:
    """Create a bounded repair node that preserves the triggering candidate."""

    context_projector = projector or ContextProjector()

    async def repair_output(state: HarnessState) -> HarnessState:
        agent_id = state["active_agent"]
        if agent_id is None:
            raise ValueError("Repair requires an active agent")
        contexts = state.get("agent_contexts", {})
        projected_context = context_projector.project(
            state,
            agent_id,
            agent_context=AgentContextMemory(
                agent_id=agent_id, history=contexts.get(agent_id, [])
            ),
        )
        consistency = state.get("consistency_result") or {}
        original_candidate = state.get("candidate_output")
        repaired = await repair_manager.repair(
            agent_id,
            original_candidate=original_candidate,
            task_context=_task_context_for_agent(state, agent_id),
            projected_context=projected_context.model_dump(mode="json"),
            violations=list(consistency.get("violations", [])),
            schema_errors=list(consistency.get("schema_errors", [])),
        )
        repaired_candidate = repaired.model_dump(mode="json")
        update: HarnessState = {
            "candidate_output": repaired_candidate,
            "candidate_output_id": str(uuid4()),
            "repair_count": state.get("repair_count", 0) + 1,
            "intervention_history": [
                {
                    "kind": "repair",
                    "agent_id": agent_id,
                    "original_candidate": original_candidate,
                    "repaired_candidate": repaired_candidate,
                    "violations": consistency.get("violations", []),
                    "schema_errors": consistency.get("schema_errors", []),
                }
            ],
            "monitor_phase": "post_repair",
        }
        if recorder is not None:
            event_state = {**state, **update}
            recorder.emit(
                TraceEventType.REPAIR_TRIGGERED,
                event_state,
                agent=agent_id,
                caused_by_output_id=state.get("candidate_output_id"),
                metadata={
                    "violations": consistency.get("violations", []),
                    "schema_errors": consistency.get("schema_errors", []),
                },
            )
            recorder.emit(
                TraceEventType.AGENT_OUTPUT_RECEIVED,
                event_state,
                agent=agent_id,
                candidate_output=repaired_candidate,
                candidate_output_id=update["candidate_output_id"],
                caused_by_output_id=state.get("candidate_output_id"),
                metadata={"output_source": "repair"},
            )
        return update

    return repair_output


def create_reset_node(
    reset_policy: ResetPolicy,
    *,
    recorder: TraceRecorder | None = None,
) -> Callable[[HarnessState], HarnessState]:
    """Create selective reset that clears only the active role's context."""

    def reset_agent(state: HarnessState) -> HarnessState:
        agent_id = state["active_agent"]
        if agent_id is None:
            raise ValueError("Reset requires an active agent")
        decision = reset_policy.decide(state)
        contexts = {key: list(value) for key, value in state.get("agent_contexts", {}).items()}
        discarded_context = contexts.get(agent_id, [])
        contexts[agent_id] = []
        update: HarnessState = {
            "agent_contexts": contexts,
            "reset_count": state.get("reset_count", 0) + 1,
            "intervention_history": [
                {
                    "kind": "reset",
                    "agent_id": agent_id,
                    "reason": decision.reason or "consistency_intervention",
                    "discarded_agent_context": discarded_context,
                    "pre_reset_consistency": state.get("consistency_result"),
                }
            ],
            "monitor_phase": "post_reset",
        }
        if recorder is not None:
            recorder.emit(
                TraceEventType.RESET_TRIGGERED,
                {**state, **update},
                agent=agent_id,
                caused_by_output_id=state.get("candidate_output_id"),
                metadata={"reason": decision.reason or "consistency_intervention"},
            )
        return update

    return reset_agent


def intervention_exhausted(
    state: HarnessState, *, recorder: TraceRecorder | None = None
) -> HarnessState:
    """Terminate bounded recovery when configured intervention limits are exhausted."""

    update: HarnessState = {
        "guardrail_action": "block",
        "error_type": "InterventionLimitError",
        "error_message": "Configured repair/reset limit exhausted for this turn.",
    }
    if recorder is not None:
        recorder.emit(
            TraceEventType.ERROR,
            {**state, **update},
            metadata={
                "error_type": update["error_type"],
                "error_message": update["error_message"],
            },
        )
    return update


def create_safety_validator_node(
    validator: SafetyValidator,
    *,
    recorder: TraceRecorder | None = None,
) -> Callable[[HarnessState], HarnessState]:
    """Create the independent M7 safety-validation node."""

    def safety_validator(state: HarnessState) -> HarnessState:
        if recorder is not None and state.get("active_agent") == "test_runner":
            recorder.emit(TraceEventType.SANDBOX_EXECUTION_REQUESTED, state)
        result = validator.validate(state.get("active_agent"), state.get("candidate_output"))
        update: HarnessState = {"safety_result": result.model_dump(mode="json")}
        if result.execution_result is not None:
            update["latest_test_results"] = [
                test_result.model_dump(mode="json")
                for test_result in result.execution_result.test_results
            ]
        if recorder is not None:
            event_state = {**state, **update}
            recorder.emit(
                TraceEventType.SAFETY_CHECKED,
                event_state,
                evaluator_version=result.evaluator_version,
            )
            if result.execution_result is not None:
                recorder.emit(
                    TraceEventType.SANDBOX_EXECUTION_COMPLETED,
                    event_state,
                    evaluator_version=result.evaluator_version,
                    metadata={"status": result.execution_result.status},
                )
        return update

    return safety_validator


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
