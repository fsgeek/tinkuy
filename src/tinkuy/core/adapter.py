"""Adapters for ingesting conversations into the projection.

IngestAdapter — batch-replays a conversation log into a projection.
Supports multiple input formats (Anthropic messages, Claude Code
JSONL, raw message lists). This is how you rehydrate tinkuy from
an existing conversation.

The critical discipline: API payloads are SYNTHESIZED FROM the
projection, never passed through from the client. The projection
is the source of truth. This is the anti-proxy-gravity boundary.
Synthesis adapters live in tinkuy.formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator

from tinkuy.core.orchestrator import (
    EventType,
    InboundEvent,
    Orchestrator,
    ResponseSignal,
    ResponseSignalType,
)
from tinkuy.core.regions import (
    ContentBlock,
    ContentKind,
    ContentStatus,
    Projection,
    RegionID,
)


# --- Conversation formats ---


class ConversationFormat(Enum):
    """Supported conversation log formats."""
    ANTHROPIC_MESSAGES = auto()   # Anthropic API messages format
    JSONL = auto()                # One JSON message per line
    RAW_MESSAGES = auto()         # Simple list of {role, content} dicts


@dataclass
class ConversationMessage:
    """Normalized representation of a single conversation message."""
    role: str              # "user", "assistant", "system", "tool"
    content: str
    turn: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    @property
    def is_assistant(self) -> bool:
        return self.role == "assistant"

    @property
    def is_system(self) -> bool:
        return self.role == "system"

    @property
    def is_tool(self) -> bool:
        return self.role in ("tool", "tool_result")


# --- Format parsers ---


def parse_anthropic_messages(
    data: dict[str, Any],
) -> list[ConversationMessage]:
    """Parse Anthropic API messages format.

    Handles both simple string content and content block arrays.
    Extracts system prompt if present.
    """
    messages: list[ConversationMessage] = []

    # System prompt (top-level in Anthropic format)
    system = data.get("system")
    if system:
        if isinstance(system, str):
            messages.append(ConversationMessage(role="system", content=system))
        elif isinstance(system, list):
            # Content block array
            text = _extract_text_from_blocks(system)
            if text:
                messages.append(
                    ConversationMessage(role="system", content=text)
                )

    # Messages
    for i, msg in enumerate(data.get("messages", [])):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append(
                ConversationMessage(
                    role=role, content=content, turn=i // 2
                )
            )
        elif isinstance(content, list):
            # Content block array — may contain text, tool_use, tool_result
            for block in content:
                block_type = block.get("type", "text")
                if block_type == "text":
                    messages.append(
                        ConversationMessage(
                            role=role,
                            content=block.get("text", ""),
                            turn=i // 2,
                        )
                    )
                elif block_type == "tool_use":
                    messages.append(
                        ConversationMessage(
                            role="tool",
                            content=json.dumps(block.get("input", {})),
                            turn=i // 2,
                            metadata={
                                "tool_name": block.get("name", ""),
                                "tool_use_id": block.get("id", ""),
                            },
                        )
                    )
                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = _extract_text_from_blocks(
                            result_content
                        )
                    messages.append(
                        ConversationMessage(
                            role="tool_result",
                            content=str(result_content),
                            turn=i // 2,
                            metadata={
                                "tool_use_id": block.get(
                                    "tool_use_id", ""
                                ),
                            },
                        )
                    )

    return messages


def parse_jsonl(lines: list[str]) -> list[ConversationMessage]:
    """Parse JSONL format — one JSON message per line."""
    messages: list[ConversationMessage] = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        messages.append(
            ConversationMessage(
                role=data.get("role", "user"),
                content=data.get("content", ""),
                turn=data.get("turn", i // 2),
                metadata=data.get("metadata", {}),
            )
        )
    return messages


def parse_raw_messages(
    messages: list[dict[str, Any]],
) -> list[ConversationMessage]:
    """Parse simple {role, content} message list."""
    return [
        ConversationMessage(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            turn=i // 2,
            metadata=msg.get("metadata", {}),
        )
        for i, msg in enumerate(messages)
    ]


def _has_content(content: str | list[dict[str, Any]]) -> bool:
    """Check if a message content field is non-empty."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return len(content) > 0
    return bool(content)


def _extract_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    """Extract text content from an Anthropic content block array."""
    parts = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


# --- Ingest adapter ---


class IngestAdapter:
    """Batch-replays a conversation log into an orchestrator.

    Walks through normalized messages, grouping by turn, and feeds
    them to the orchestrator. The result is a fully populated
    projection that can then be continued live.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    def ingest_messages(
        self, messages: list[ConversationMessage]
    ) -> None:
        """Replay a list of messages into the orchestrator."""
        # Group messages into turns
        for turn_messages in self._group_by_turn(messages):
            user_events: list[InboundEvent] = []
            assistant_content: list[str] = []

            for msg in turn_messages:
                if msg.is_system:
                    user_events.append(
                        InboundEvent(
                            type=EventType.SYSTEM_UPDATE,
                            content=msg.content,
                            label="system",
                            metadata=msg.metadata,
                        )
                    )
                elif msg.is_user:
                    user_events.append(
                        InboundEvent(
                            type=EventType.USER_MESSAGE,
                            content=msg.content,
                            label="user",
                            metadata=msg.metadata,
                        )
                    )
                elif msg.is_tool:
                    user_events.append(
                        InboundEvent(
                            type=EventType.TOOL_RESULT,
                            content=msg.content,
                            label=msg.metadata.get(
                                "tool_name", "tool_result"
                            ),
                            metadata=msg.metadata,
                        )
                    )
                elif msg.is_assistant:
                    assistant_content.append(msg.content)

            # Begin turn with user-side events
            if user_events:
                self.orchestrator.begin_turn(user_events)

            # Ingest assistant response if present
            if assistant_content:
                self.orchestrator.ingest_response(
                    content="\n".join(assistant_content),
                    label="assistant",
                )

    def ingest_anthropic(self, data: dict[str, Any]) -> None:
        """Ingest from Anthropic API messages format."""
        messages = parse_anthropic_messages(data)
        self.ingest_messages(messages)

    def ingest_jsonl(self, text: str) -> None:
        """Ingest from JSONL format."""
        messages = parse_jsonl(text.strip().split("\n"))
        self.ingest_messages(messages)

    def ingest_file(self, path: str | Path) -> None:
        """Ingest from a file, auto-detecting format."""
        path = Path(path)
        content = path.read_text(encoding="utf-8")

        if path.suffix == ".jsonl":
            self.ingest_jsonl(content)
        elif path.suffix == ".json":
            data = json.loads(content)
            if isinstance(data, dict) and "messages" in data:
                self.ingest_anthropic(data)
            elif isinstance(data, list):
                messages = parse_raw_messages(data)
                self.ingest_messages(messages)
            else:
                raise ValueError(
                    f"Unrecognized JSON format in {path}"
                )
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _group_by_turn(
        self, messages: list[ConversationMessage]
    ) -> Iterator[list[ConversationMessage]]:
        """Group messages into turns.

        A turn is a sequence of messages that belong together —
        typically a user message (possibly with tool results)
        followed by an assistant response. System messages are
        placed in the first turn.
        """
        if not messages:
            return

        current_turn: list[ConversationMessage] = []
        last_role: str | None = None

        for msg in messages:
            # System messages always go with the current group
            if msg.is_system:
                current_turn.append(msg)
                continue

            # Start a new turn when we transition from assistant to user
            if last_role == "assistant" and not msg.is_assistant:
                if current_turn:
                    yield current_turn
                    current_turn = []

            current_turn.append(msg)
            last_role = "assistant" if msg.is_assistant else msg.role

        if current_turn:
            yield current_turn


# --- Page table coalescing ---


def coalesce_episodes(
    entries: list[dict[str, Any]],
    current_turn: int,
) -> list[dict[str, Any]]:
    """Group entries into temporal episodes.

    Consecutive entries (by created turn) within 3 turns of each other
    are merged into one episode. This is the simplest coalescing
    strategy — temporal proximity only, no topic analysis.
    """
    if not entries:
        return []

    # Sort by age descending (oldest first)
    sorted_entries = sorted(entries, key=lambda e: -e["age_turns"])

    episodes: list[dict[str, Any]] = []
    current_ep: list[dict[str, Any]] = [sorted_entries[0]]

    for entry in sorted_entries[1:]:
        prev = current_ep[-1]
        # Merge if within 3 turns of the previous entry
        if abs(entry["age_turns"] - prev["age_turns"]) <= 3:
            current_ep.append(entry)
        else:
            episodes.append(_summarize_episode(current_ep, current_turn))
            current_ep = [entry]

    if current_ep:
        episodes.append(_summarize_episode(current_ep, current_turn))

    return episodes


def _summarize_episode(
    entries: list[dict[str, Any]],
    current_turn: int,
) -> dict[str, Any]:
    """Produce a summary for a group of entries."""
    ages = [e["age_turns"] for e in entries]
    oldest = max(ages)
    newest = min(ages)

    if oldest == newest:
        turn_range = f"{current_turn - oldest}"
    else:
        turn_range = f"{current_turn - oldest}-{current_turn - newest}"

    kinds = set(e["kind"] for e in entries)

    return {
        "turn_range": turn_range,
        "block_count": len(entries),
        "total_tokens": sum(e["size_tokens"] for e in entries),
        "kinds": "+".join(sorted(kinds)),
    }
