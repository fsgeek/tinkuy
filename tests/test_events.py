"""Tests for event bus, event consumers, and event log behavior."""

from tinkuy.events import (
    ConsoleStatusConsumer,
    Event,
    EventBus,
    EventKind,
    EventLog,
)


class RecordingConsumer:
    def __init__(self):
        self.events = []

    def on_event(self, event: Event) -> None:
        self.events.append(event)


class TestEventBus:
    def test_subscribe_emit_and_consumer_count(self):
        bus = EventBus()
        c1 = RecordingConsumer()
        c2 = RecordingConsumer()
        event = Event(kind=EventKind.BLOCK_CREATED, turn=1, data={"handle": "h1"})

        bus.subscribe(c1)
        bus.subscribe(c2)
        assert bus.consumer_count == 2

        bus.emit(event)

        assert c1.events == [event]
        assert c2.events == [event]

    def test_unsubscribe_removes_only_target_consumer(self):
        bus = EventBus()
        keep = RecordingConsumer()
        remove = RecordingConsumer()

        bus.subscribe(keep)
        bus.subscribe(remove)
        bus.unsubscribe(remove)

        event = Event(kind=EventKind.BLOCK_ACCESSED, turn=2)
        bus.emit(event)

        assert bus.consumer_count == 1
        assert keep.events == [event]
        assert remove.events == []

    def test_emit_with_no_consumers_is_noop(self):
        bus = EventBus()

        # Should not raise when there are no subscribers.
        bus.emit(Event(kind=EventKind.TURN_BEGAN, turn=3))

        assert bus.consumer_count == 0


class TestEventLog:
    def test_stores_events_and_supports_filtering(self):
        log = EventLog()
        e1 = Event(kind=EventKind.BLOCK_CREATED, turn=1, data={"handle": "a"})
        e2 = Event(kind=EventKind.BLOCK_ACCESSED, turn=1, data={"handle": "a"})
        e3 = Event(kind=EventKind.BLOCK_CREATED, turn=2, data={"handle": "b"})
        e4 = Event(kind=EventKind.BLOCK_EVICTED, turn=2, data={"other": "x"})

        for event in (e1, e2, e3, e4):
            log.on_event(event)

        assert len(log) == 4
        assert log.events_of(EventKind.BLOCK_CREATED) == [e1, e3]
        assert log.events_for_handle("a") == [e1, e2]
        assert log.events_for_handle("b") == [e3]
        assert log.events_for_handle("missing") == []

    def test_max_events_trims_oldest_entries(self):
        log = EventLog(max_events=3)
        events = [Event(kind=EventKind.BLOCK_CREATED, turn=i, data={"handle": str(i)}) for i in range(5)]

        for event in events:
            log.on_event(event)

        assert len(log) == 3
        assert log.events == events[-3:]

    def test_clear_empties_log(self):
        log = EventLog()
        log.on_event(Event(kind=EventKind.BLOCK_CREATED, turn=1))
        log.on_event(Event(kind=EventKind.BLOCK_ACCESSED, turn=2))

        log.clear()

        assert len(log) == 0
        assert log.events == []

    def test_events_property_returns_copy(self):
        log = EventLog()
        event = Event(kind=EventKind.BLOCK_CREATED, turn=1)
        log.on_event(event)

        snapshot = log.events
        snapshot.append(Event(kind=EventKind.BLOCK_EVICTED, turn=2))

        assert log.events == [event]


class TestConsoleStatusConsumer:
    def test_renders_on_pressure_zone_change(self):
        rendered = []
        consumer = ConsoleStatusConsumer(context_limit=1000, render_fn=rendered.append)

        consumer.on_event(
            Event(
                kind=EventKind.PRESSURE_READ,
                turn=1,
                data={"total_tokens": 100, "zone": "low"},
            )
        )
        consumer.on_event(
            Event(
                kind=EventKind.PRESSURE_READ,
                turn=1,
                data={"total_tokens": 200, "zone": "low"},
            )
        )
        consumer.on_event(
            Event(
                kind=EventKind.PRESSURE_READ,
                turn=1,
                data={"total_tokens": 700, "zone": "elevated"},
            )
        )

        assert len(rendered) == 2
        assert "Context: 100/1,000 tok (10%)" in rendered[0]
        assert "Pressure: low" in rendered[0]
        assert "Turn: 0" in rendered[0]
        assert "Context: 700/1,000 tok (70%)" in rendered[1]
        assert "Pressure: elevated" in rendered[1]

    def test_renders_on_turn_began_even_without_pressure_read(self):
        rendered = []
        consumer = ConsoleStatusConsumer(context_limit=200, render_fn=rendered.append)

        consumer.on_event(Event(kind=EventKind.TURN_BEGAN, turn=9))

        assert len(rendered) == 1
        assert rendered[0] == "Context: 0/200 tok (0%) | Pressure: unknown | Turn: 9"

    def test_zero_context_limit_renders_100_percent(self):
        rendered = []
        consumer = ConsoleStatusConsumer(context_limit=0, render_fn=rendered.append)

        consumer.on_event(
            Event(
                kind=EventKind.PRESSURE_READ,
                turn=1,
                data={"total_tokens": 123, "zone": "critical"},
            )
        )

        assert len(rendered) == 1
        assert "Context: 123/0 tok (100%)" in rendered[0]


class TestEventRepr:
    def test_repr_includes_kind_turn_and_data(self):
        event = Event(
            kind=EventKind.SIGNAL_RELEASE,
            turn=5,
            data={"handle": "abc", "declared_losses": "details"},
        )

        text = repr(event)

        assert text.startswith("Event(SIGNAL_RELEASE, turn=5, ")
        assert "handle='abc'" in text
        assert "declared_losses='details'" in text

    def test_repr_for_empty_data(self):
        event = Event(kind=EventKind.TURN_BEGAN, turn=1)

        assert repr(event) == "Event(TURN_BEGAN, turn=1, )"
