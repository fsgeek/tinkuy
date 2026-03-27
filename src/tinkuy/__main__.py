"""CLI entry point for tinkuy.

Usage:
    python -m tinkuy serve [--port PORT] [--upstream URL] [--data-dir DIR]
    python -m tinkuy status [--port PORT]
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tinkuy",
        description="Projective gateway for transformer memory hierarchy",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    serve_p = sub.add_parser("serve", help="Start the tinkuy gateway")
    serve_p.add_argument(
        "--port", type=int, default=None,
        help="Port to listen on (default: auto-detect free port)",
    )
    serve_p.add_argument(
        "--upstream", default=None,
        help="Anthropic upstream URL (default: env or https://api.anthropic.com)",
    )
    serve_p.add_argument(
        "--gemini-upstream", default=None,
        help="Gemini upstream URL (default: env or https://generativelanguage.googleapis.com)",
    )
    serve_p.add_argument(
        "--data-dir", default=None,
        help="Directory for persistent state (pages, checkpoints)",
    )
    serve_p.add_argument(
        "--context-limit", type=int, default=200_000,
        help="Context window token limit (default: 200000)",
    )
    serve_p.add_argument(
        "--taste", action="store_true",
        help="Use taste-native gateway (model self-curates via tensor)",
    )

    args = parser.parse_args()

    if args.command == "serve":
        from tinkuy.server import serve
        serve(
            port=args.port,
            upstream=args.upstream,
            gemini_upstream=args.gemini_upstream,
            data_dir=args.data_dir,
            context_limit=args.context_limit,
            taste=args.taste,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
