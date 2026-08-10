"""Role agent that requests approved Python test execution."""

from harness.agents.base import BaseAgent
from harness.models.schemas import ExecutionRequest


class TestRunnerAgent(BaseAgent[ExecutionRequest]):
    """Produces an execution request; sandbox execution is intentionally deferred."""

    __test__ = False
    agent_id = "test_runner"
    output_schema = ExecutionRequest
