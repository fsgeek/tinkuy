"""Long-horizon chat with Tinkuy memory management.

Usage:
    python -m tinkuy.chat [--model MODEL] [--context-limit TOKENS]

A minimal chat interface that demonstrates Tinkuy's virtual memory
system. Conversations can run indefinitely — the projection manages
eviction and page tables transparently.

No file system, no tools, no git. Just conversation with memory.
"""

from __future__ import annotations

import argparse
import os
import sys

from tinkuy.chat.session import ChatSession


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tinkuy-chat",
        description="Long-horizon chat with context-window memory management",
    )
    parser.add_argument(
        "--model", default=os.environ.get("TINKUY_MODEL", "claude-sonnet-4-20250514"),
        help="Model name for litellm (default: claude-sonnet-4-20250514 or $TINKUY_MODEL)",
    )
    parser.add_argument(
        "--context-limit", type=int, default=200_000,
        help="Context window token limit (default: 200000)",
    )
    parser.add_argument(
        "--system", default=None,
        help="System prompt (default: built-in)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory for persistent state (pages, checkpoints)",
    )
    args = parser.parse_args()

    session = ChatSession(
        model=args.model,
        context_limit=args.context_limit,
        system_prompt=args.system,
        data_dir=args.data_dir,
    )

    print(f"tinkuy-chat | model={args.model} | context={args.context_limit:,} tokens")
    print(f"Type your message. Ctrl+D to exit. /status for memory info.\n")

    try:
        session.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except EOFError:
        pass

    print(f"\nSession ended. {session.turns} turns, {session.total_tokens_sent:,} tokens sent.")


if __name__ == "__main__":
    main()
