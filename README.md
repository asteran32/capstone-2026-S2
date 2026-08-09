# Multi-Agent Role Harness

A research prototype for studying role consistency and drift in a fixed,
three-agent LLM tutoring system orchestrated with LangGraph.

## Development setup

Requires Python 3.11 or later.

```bash
python -m pip install -e '.[dev]'
pytest
```

The repository currently provides typed configuration, shared state, and data
schemas. Agents, graph workflow, guardrails, and external model integrations
have not yet been implemented.
