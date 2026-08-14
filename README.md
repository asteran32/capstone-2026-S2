# Multi-Agent Role Harness

A research prototype for studying role consistency and drift in a fixed,
three-agent LLM tutoring system orchestrated with LangGraph.

## Development setup

Requires Python 3.11 or later.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
./.venv/bin/python -m pytest
```

## Run one session

The deterministic mock provider is used by default. It requires no API key and
does not make a network request:

```bash
./.venv/bin/python scripts/run_session.py "두 정수의 합 문제를 만들어 줘"
```

To use the OpenAI provider, export the credentials/configuration and select it
explicitly. Replace `your-model-id` with a model available to your project:

```bash
export OPENAI_API_KEY="your-api-key"
export MODEL_NAME="your-model-id"
./.venv/bin/python scripts/run_session.py \
  "두 정수의 합 문제를 만들어 줘" \
  --provider openai
```

The script reads these values from the process environment; it does not load a
`.env` file automatically. Real API requests may incur usage charges. Never
commit an API key to this repository. `temperature` is optional in
`config/models.yaml`; leave it as `null` for models that do not support that
request parameter.

Async graphs that create a durable SQLite checkpointer own that connection.
Use the returned graph as an async context manager, or call `await graph.aclose()`
when the application finishes:

```python
async with await build_mvp_graph_async(provider, persistence=config) as graph:
    result = await graph.ainvoke(input_state, run_config)
```

For code review, pass learner code as a separate argument:

```bash
./.venv/bin/python scripts/run_session.py \
  "내 코드를 검토해 줘" \
  --learner-code "print(1 + 2)"
```

Run `./.venv/bin/python scripts/run_session.py --help` for all options.

## Run M9 experiments

Experiment commands use the deterministic mock provider by default. Baseline
keeps the same three-agent topology, disables role-guardrail intervention, and
retains passive drift observations:

```bash
./.venv/bin/python scripts/run_baseline.py \
  --repetitions 3 \
  --output data/experiments/baseline.csv
```

Run any supported Layer 1/2/3 combination without editing graph source:

```bash
./.venv/bin/python scripts/run_ablation.py \
  --condition L1_L2 \
  --repetitions 3 \
  --seed 42 \
  --output data/experiments/l1-l2.csv
```

Supported conditions are `BASELINE`, `L1`, `L2`, `L3`, `L1_L2`, `L1_L3`,
`L2_L3`, and `FULL`. Select `--provider openai` explicitly for real, billable
model calls. The CSV export includes condition, trace linkage, raw drift
indicators, model configuration, seed, and a reproducible configuration hash.
