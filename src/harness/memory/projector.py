"""Role-specific context projection from shared harness state."""

from __future__ import annotations

from typing import Any, Mapping

from harness.guardrails.contracts import AgentId
from harness.memory.context import AgentContextMemory, ProjectedContext, TaskMemory


class ContextProjector:
    """Exposes only role-relevant task facts and same-role memory."""

    def project(
        self,
        state: Mapping[str, Any],
        agent_id: AgentId,
        *,
        agent_context: AgentContextMemory | None = None,
    ) -> ProjectedContext:
        """Create a role-scoped view without leaking unrelated state fields."""

        if agent_context is not None and agent_context.agent_id != agent_id:
            raise ValueError("agent context can only be projected to its owning agent")

        task_memory = TaskMemory.model_validate(
            {
                field: state[field]
                for field in TaskMemory.model_fields
                if field in state
            }
        )
        task = self._project_task(task_memory, agent_id)
        history = [] if agent_context is None else agent_context.history
        return ProjectedContext(agent_id=agent_id, task=task, agent_context=history)

    @staticmethod
    def _project_task(task_memory: TaskMemory, agent_id: AgentId) -> dict[str, Any]:
        if agent_id == "problem_designer":
            return {
                "learner_level": task_memory.learner_level,
                "user_message": task_memory.user_message,
                "current_problem": task_memory.current_problem,
            }
        if agent_id == "code_reviewer":
            return {
                "learner_level": task_memory.learner_level,
                "user_message": task_memory.user_message,
                "learner_code": task_memory.learner_code,
                "current_problem": task_memory.current_problem,
                "latest_test_results": task_memory.latest_test_results,
            }
        return {
            "learner_code": task_memory.learner_code,
            "current_problem": task_memory.current_problem,
            "latest_test_results": task_memory.latest_test_results,
        }
