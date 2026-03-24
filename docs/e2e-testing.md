# End-to-End Testing with the PRD Benchmark

The PRD planning benchmark (`../bladnman/planning_benchmark/`) provides a
multi-document reading + planning task that exercises Tinkuy's full request
pipeline: ingest, projection, synthesis, upstream proxy, and streaming
response reconstruction.

## Prerequisites

- Tinkuy installed with server extras: `uv pip install -e ".[serve]"`
- The benchmark repo at `~/projects/bladnman/planning_benchmark/`
- A valid `ANTHROPIC_API_KEY` in the environment

## Steps

### 1. Run unit tests first

```bash
.venv/bin/python -m pytest tests/ -v
```

All tests must pass before proceeding.

### 2. Start the Tinkuy gateway

```bash
.venv/bin/python -m tinkuy serve --port 8080
```

Logs go to stderr. To capture them:

```bash
.venv/bin/python -m tinkuy serve --port 8080 2>tinkuy-stderr.log &
```

### 3. Run the benchmark through Tinkuy

From the tinkuy project root:

```bash
cd ~/projects/bladnman/planning_benchmark

ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
  claude --print \
  --model claude-haiku-4-5-20251001 \
  --max-turns 1 \
  "Read 1-START_HERE.md and follow its instructions. Read all documents in docs/prd/ recursively, then write results/PLAN.md." \
  2>~/projects/tinkuy/tinkuy-prd-stderr.log
```

Notes:
- Use `--max-turns 1` for a quick smoke test (verifies the pipeline works).
- Use `--max-turns 10` (or higher) for a full exercise that produces `results/PLAN.md`.
- `claude-haiku-4-5-20251001` is cheap and fast for validation. Swap for
  `claude-sonnet-4-6` or `claude-opus-4-6` for quality runs.

### 4. Verify success

Check the Tinkuy logs for a clean request cycle:

```bash
tail -20 ~/projects/tinkuy/tinkuy-stderr.log
```

You should see:
- `← request | session=default stream=True ...` — request received
- `checkpoint written | turn=N ...` — projection checkpointed
- `synth | messages=N system=yes` — payload synthesized
- `WIRE -> POST https://api.anthropic.com/v1/messages` — upstream call
- `-> upstream | status=200 (stream)` — successful response
- `ok ...` — summary line with token counts, latency, cache stats

Red flags:
- Any `AttributeError`, `KeyError`, or Python traceback
- `status=4xx` or `status=5xx` from upstream
- Missing `checkpoint written` lines

### 5. Clean up

```bash
# Stop the server (if backgrounded)
kill %1

# Remove test state (optional)
rm -rf /tmp/tinkuy-*
```

## What this exercises

| Component | Coverage |
|-----------|----------|
| `server.py` | HTTP routing, request parsing, streaming proxy |
| `gateway.py` | Session management, prepare_request, response ingestion |
| `orchestrator.py` | Turn lifecycle, checkpointing |
| `adapter.py` | Ingest parsing, payload synthesis, alternation |
| `stream.py` | SSE parsing, message reconstruction |
| `pressure.py` | Pressure calculation (visible in logs) |
| `store.py` | Checkpoint persistence |
| `events.py` | Console status line, telemetry |

## Multi-turn exercise

For deeper coverage (eviction, pressure transitions, tensor creation), run
with more turns and a longer prompt:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
  claude \
  --model claude-haiku-4-5-20251001 \
  --max-turns 20 \
  "Read 1-START_HERE.md and follow its instructions completely."
```

This generates enough tool-use turns to push context usage into moderate
pressure, exercising the eviction scheduler.
