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
commit an API key to this repository.

For code review, pass learner code as a separate argument:

```bash
./.venv/bin/python scripts/run_session.py \
  "내 코드를 검토해 줘" \
  --learner-code "print(1 + 2)"
```

Run `./.venv/bin/python scripts/run_session.py --help` for all options.
