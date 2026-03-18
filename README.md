# Tinkuy

*The meeting place* — a projective gateway for transformer memory hierarchy.

Tinkuy sits between a client (like Claude Code) and the Anthropic API, managing the context window as a structured projection with five stability regions rather than a flat message list. It gives the model cooperative control over what it remembers, what it compresses, and what it lets go.

## The problem

LLM context windows are treated as append-only logs. Every message is preserved verbatim in sequence until the window fills, then the conversation dies or gets naively truncated. This is like running a computer without virtual memory — once RAM is full, you crash.

Tinkuy is virtual memory for transformer context windows. It manages eviction, compression, and recall so conversations can run indefinitely without losing what matters.

## How it works

**Five regions, ordered by stability:**

| Region | Name | Stability | Content |
|--------|------|-----------|---------|
| R0 | Tools | Static | Tool schemas, definitions |
| R1 | System | Stable | System prompt, project context |
| R2 | Durable | Long-lived | Tensors (compressed summaries), reference files |
| R3 | Ephemeral | Volatile | Conversation history, tool results |
| R4 | Current | Turn-scoped | Current user message |

**Pressure-gated eviction:** Context usage determines behavior, not age alone.

| Zone | Usage | Behavior |
|------|-------|----------|
| Low | < 50% | Hold everything |
| Moderate | 50–70% | Schedule candidates, act only at idle |
| Elevated | 70–85% | Request model cooperation |
| Critical | > 85% | Aggressive gateway-initiated removal |

**Cooperative memory:** The model participates in eviction by producing *tensors* — compressed summaries of content it's releasing, with declared losses. Eviction costs the model output tokens, creating natural back-pressure against discarding valuable content.

**Anti-proxy-gravity:** The projection is the source of truth. API payloads are synthesized from the projection, never passed through from the client. This is the core design discipline.

## Quick start

```bash
# Install
pip install tinkuy

# Install with proxy support
pip install tinkuy[serve]

# Start the gateway
python -m tinkuy serve --port 8340

# Point your client at it
export ANTHROPIC_BASE_URL=http://127.0.0.1:8340
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Client     │────▶│    Tinkuy    │────▶│  Anthropic   │
│ (Claude Code) │◀────│   Gateway    │◀────│     API      │
└──────────────┘     └──────┬───────┘     └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐
         │Pressure │ │  Event  │ │  Page   │
         │Scheduler│ │   Bus   │ │  Store  │
         └─────────┘ └─────────┘ └─────────┘
```

**Modules:**

| Module | Purpose |
|--------|---------|
| `regions.py` | Data structures — blocks, regions, projection |
| `pressure.py` | Pressure-gated eviction policy |
| `orchestrator.py` | Turn lifecycle, cooperative signals, eviction |
| `events.py` | Structured event bus, pluggable consumers |
| `store.py` | Page store + checkpoint persistence |
| `adapter.py` | Ingest (rehydrate) + synthesize (live) |
| `gateway.py` | Integration layer |
| `harness.py` | Session driver, signal extraction |
| `proxy.py` | FastAPI HTTP proxy |

## Usage as a library

```python
from tinkuy.gateway import Gateway, GatewayConfig

# Create a gateway
gw = Gateway(GatewayConfig(
    context_limit=200_000,
    data_dir="/tmp/tinkuy-state",
))

# Rehydrate from an existing conversation
gw.rehydrate("/path/to/conversation.json")

# Process a turn
result = gw.process_turn("What files handle authentication?")
# result.api_payload is a synthesized Anthropic API payload

# After calling the API, ingest the response
gw.ingest_response(
    content="The auth module is in src/auth/...",
    signals=[{"type": "release", "handle": "abc123", ...}],
)
```

## Persistence

- **Page store** (workspace-scoped): Verbatim originals of evicted content, shared across sessions. Written eagerly at eviction time.
- **Checkpoint store** (session-scoped): Projection snapshots at turn boundaries, keyed by session ID.

Both support filesystem and in-memory backends.

## Observability

The event bus emits structured events for every mutation:

```python
from tinkuy.events import EventBus, EventLog

log = EventLog()
bus = EventBus()
bus.subscribe(log)

# Later: inspect what happened
for event in log.events_of(EventKind.BLOCK_EVICTED):
    print(f"Evicted {event.data['handle']} at turn {event.turn}")
```

Built-in consumers: `EventLog` (in-memory), `ConsoleStatusConsumer` (terminal status line).

## Development

```bash
git clone <repo-url>
cd tinkuy
uv pip install -e ".[dev]"
python -m pytest tests/ -v
```

110 tests across 10 modules.

## Name

*Tinkuy* is Quechua for "the meeting place" — where different streams converge. In this case, the streams are the client's conversation, the model's memory, and the gateway's management of both.

## License

TBD
