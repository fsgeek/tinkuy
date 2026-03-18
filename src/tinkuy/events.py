"""Structured event stream for observability.

The orchestrator emits events as it works. Consumers subscribe to the
stream and process events however they want — console status lines,
dashboards, structured logs, metrics aggregation.

The event types are the vocabulary of the system's behavior. If you
can't observe it through events, you can't understand it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


# --- Event types ---

class EventKind(Enum):
    """All observable events in the system."""
    # Lifecycle
    BLOCK_CREATED = auto()
    BLOCK_ACCESSED = auto()
    BLOCK_NOMINATED = auto()
    BLOCK_EVICTED = auto()
    BLOCK_RECALLED = auto()
    BLOCK_AGED = auto()         # R4 → R3

    # Pressure
    PRESSURE_READ = auto()
    PRESSURE_CHANGED = auto()

    # Eviction decisions
    CANDIDATE_SCORED = auto()
    EVICTION_DECIDED = auto()
    EVICTION_EXECUTED = auto()

    # Cooperative signals
    SIGNAL_RELEASE = auto()
    SIGNAL_RETAIN = auto()
    SIGNAL_RECALL = auto()

    # Turn lifecycle
    TURN_BEGAN = auto()
    TURN_RESPONSE_INGESTED = auto()
    IDLE_ENTERED = auto()
    IDLE_EXITED = auto()


@dataclass(frozen=True)
class Event:
    """A single observable event."""
    kind: EventKind
    turn: int
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        data_str = ", ".join(f"{k}={v!r}" for k, v in self.data.items())
        return f"Event({self.kind.name}, turn={self.turn}, {data_str})"


# --- Consumer protocol ---

class EventConsumer(Protocol):
    """Protocol for event consumers.

    Implement this to receive events from the orchestrator.
    Consumers should be cheap — the orchestrator calls them
    synchronously in the hot path.
    """

    def on_event(self, event: Event) -> None:
        """Handle an event. Must be fast and non-blocking."""
        ...


# --- Event bus ---

class EventBus:
    """Distributes events to registered consumers.

    The bus is simple by design: synchronous dispatch, no filtering,
    no buffering. Consumers that need async or selective processing
    should handle that internally.
    """

    def __init__(self) -> None:
        self._consumers: list[EventConsumer] = []

    def subscribe(self, consumer: EventConsumer) -> None:
        """Register a consumer."""
        self._consumers.append(consumer)

    def unsubscribe(self, consumer: EventConsumer) -> None:
        """Remove a consumer."""
        self._consumers = [c for c in self._consumers if c is not consumer]

    def emit(self, event: Event) -> None:
        """Dispatch an event to all consumers."""
        for consumer in self._consumers:
            consumer.on_event(event)

    @property
    def consumer_count(self) -> int:
        return len(self._consumers)


# --- Built-in consumers ---

class EventLog(EventConsumer):
    """Append-only in-memory event log.

    Simple consumer that stores all events. Useful for testing,
    debugging, and post-hoc analysis.
    """

    def __init__(self, max_events: int = 10_000) -> None:
        self._events: list[Event] = []
        self._max_events = max_events

    def on_event(self, event: Event) -> None:
        self._events.append(event)
        # Trim from the front if we exceed capacity
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def events_of(self, kind: EventKind) -> list[Event]:
        """Filter events by kind."""
        return [e for e in self._events if e.kind == kind]

    def events_for_handle(self, handle: str) -> list[Event]:
        """All events mentioning a specific content handle."""
        return [
            e for e in self._events
            if e.data.get("handle") == handle
        ]

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


class ConsoleStatusConsumer(EventConsumer):
    """Renders a console status line, similar to Pichay's live status.

    Tracks pressure state and emits formatted status on pressure
    changes and turn boundaries.
    """

    def __init__(
        self,
        context_limit: int = 200_000,
        render_fn: Any = None,
    ) -> None:
        self.context_limit = context_limit
        self._render_fn = render_fn or self._default_render
        self._last_zone: str | None = None
        self._total_tokens: int = 0
        self._turn: int = 0

    def on_event(self, event: Event) -> None:
        if event.kind == EventKind.PRESSURE_READ:
            self._total_tokens = event.data.get("total_tokens", 0)
            zone = event.data.get("zone", "unknown")
            if zone != self._last_zone or event.kind == EventKind.TURN_BEGAN:
                self._last_zone = zone
                self._render()

        elif event.kind == EventKind.TURN_BEGAN:
            self._turn = event.turn
            self._render()

    def _render(self) -> None:
        usage_pct = (
            (self._total_tokens / self.context_limit * 100)
            if self.context_limit > 0 else 100
        )
        line = (
            f"Context: {self._total_tokens:,}/{self.context_limit:,} tok "
            f"({usage_pct:.0f}%) | "
            f"Pressure: {self._last_zone or 'unknown'} | "
            f"Turn: {self._turn}"
        )
        self._render_fn(line)

    @staticmethod
    def _default_render(line: str) -> None:
        print(f"[tinkuy] {line}")
