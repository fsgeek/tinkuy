"""Chat session — the core loop for long-horizon conversations.

Owns an Orchestrator, a LiteLLMAdapter, and the litellm API call.
Each turn: read user input → begin_turn → synthesize → API call →
ingest response → print. The projection handles all memory management.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from tinkuy.core.orchestrator import (
    EventType,
    InboundEvent,
    Orchestrator,
)
from tinkuy.core.pressure import PressureZone
from tinkuy.core.store import FileCheckpointStore, FilePageStore
from tinkuy.formats.litellm import LiteLLMAdapter

_DEFAULT_SYSTEM = (
    "You are a thoughtful conversational partner in a long-horizon chat session. "
    "This conversation is managed by Tinkuy, a virtual memory system for "
    "transformer context windows. You may see a <yuyay-page-table> in the system "
    "prompt showing what content is in memory. If you need to recall evicted "
    "content, mention the handle and Tinkuy will fault it back in.\n\n"
    "Be natural. This is a conversation, not a task."
)


class ChatSession:
    """Drives a long-horizon chat through the Tinkuy memory system."""

    def __init__(
        self,
        model: str,
        context_limit: int = 200_000,
        system_prompt: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM

        # Set up persistence if data_dir provided
        page_store = None
        checkpoint_store = None
        if data_dir:
            data_path = Path(data_dir)
            data_path.mkdir(parents=True, exist_ok=True)
            page_store = FilePageStore(data_path / "pages")
            checkpoint_store = FileCheckpointStore(data_path / "checkpoints")

        self.orchestrator = Orchestrator(
            context_limit=context_limit,
            page_store=page_store,
            checkpoint_store=checkpoint_store,
        )
        self.adapter = LiteLLMAdapter(self.orchestrator)
        self.turns = 0
        self.total_tokens_sent = 0

        # Seed the system prompt
        self.orchestrator.begin_turn([
            InboundEvent(
                type=EventType.SYSTEM_UPDATE,
                content=self.system_prompt,
                label="system",
            ),
        ])

    def run(self) -> None:
        """Main input loop. Reads from stdin, prints to stdout."""
        while True:
            # Read user input (multi-line: blank line submits)
            user_input = self._read_input()
            if user_input is None:
                break

            # Handle commands
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            # Run a turn
            response = self.turn(user_input)
            print(f"\n{response}\n")

    def turn(self, user_input: str) -> str:
        """Execute one conversation turn."""
        import litellm

        # Feed user message to orchestrator
        self.orchestrator.begin_turn([
            InboundEvent(
                type=EventType.USER_MESSAGE,
                content=user_input,
                label="user",
            ),
        ])

        # Inject page table into system content if there's memory state
        page_table = self.adapter.synthesize_page_table()
        if page_table:
            self.orchestrator.begin_turn([
                InboundEvent(
                    type=EventType.SYSTEM_UPDATE,
                    content=page_table,
                    label="page_table",
                ),
            ])

        # Synthesize messages
        payload = self.adapter.synthesize_messages()
        messages = payload["messages"]
        self.total_tokens_sent += sum(
            len(m.get("content", "")) // 4 for m in messages  # rough estimate
        )

        # Call API
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                stream=False,
            )
            content = response.choices[0].message.content or ""

            # Track actual token usage
            if hasattr(response, "usage") and response.usage:
                self.total_tokens_sent = getattr(
                    response.usage, "prompt_tokens", self.total_tokens_sent
                )

        except Exception as e:
            content = f"[API error: {e}]"

        # Ingest response
        self.orchestrator.ingest_response(
            content=content,
            label="assistant",
        )

        self.turns += 1
        return content

    def _read_input(self) -> str | None:
        """Read user input from stdin. Returns None on EOF."""
        try:
            line = input("you: ")
        except EOFError:
            return None
        return line

    def _handle_command(self, cmd: str) -> None:
        """Handle slash commands."""
        cmd = cmd.strip().lower()

        if cmd == "/status":
            proj = self.orchestrator.projection
            zone = self.orchestrator._scheduler.current_zone(proj)
            print(f"\n  Turn: {self.orchestrator.turn}")
            print(f"  Tokens: {proj.total_tokens:,}")
            print(f"  Pressure: {zone.name}")
            print(f"  Blocks: {len([b for r in proj.regions.values() for b in r.blocks])}")

            page_table = self.adapter.synthesize_page_table()
            if page_table:
                print(f"\n{page_table}")
            print()

        elif cmd == "/help":
            print("\n  /status — show memory state")
            print("  /help   — this message")
            print("  Ctrl+D  — exit\n")

        else:
            print(f"\n  Unknown command: {cmd}\n")
