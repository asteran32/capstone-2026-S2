"""Role agent for generating or revising programming problems."""

from harness.agents.base import BaseAgent
from harness.models.schemas import ProblemDesignOutput


class ProblemDesignAgent(BaseAgent[ProblemDesignOutput]):
    """Produces validated programming-problem specifications only."""

    agent_id = "problem_designer"
    output_schema = ProblemDesignOutput
