"""Role agent for pedagogical learner-code review."""

from harness.agents.base import BaseAgent
from harness.models.schemas import CodeReviewOutput


class CodeReviewAgent(BaseAgent[CodeReviewOutput]):
    """Produces validated feedback and never executes learner code."""

    agent_id = "code_reviewer"
    output_schema = CodeReviewOutput
