"""Evaluation runner — executes tasks in multiple modes and compares results.

Usage:
    python -m tinkuy.eval.runner --task needle_in_haystack --modes baseline,full

The runner executes the same task in each mode, captures full transcripts,
and writes them to disk for analysis. Comparison is a separate step —
the runner's job is to produce clean, complete data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tinkuy.eval.driver import ConversationDriver, Task, Transcript
from tinkuy.eval.tasks import TASK_REGISTRY
from tinkuy.gateway import Gateway, GatewayConfig

log = logging.getLogger(__name__)


def _make_gateway(context_limit: int = 200_000) -> Gateway:
    """Create a fresh gateway with no prior state."""
    config = GatewayConfig(
        context_limit=context_limit,
        enable_console=False,
        enable_event_log=True,
    )
    return Gateway(config)


async def run_task(
    task: Task,
    model: str,
    modes: list[str],
    context_limit: int = 200_000,
    max_tokens: int = 4096,
) -> dict[str, Transcript]:
    """Run a task in multiple modes, return transcripts keyed by mode."""
    results: dict[str, Transcript] = {}

    for mode in modes:
        log.info("--- running %s in mode=%s ---", task.name, mode)

        # Fresh gateway per run — no contamination between modes
        gw = _make_gateway(context_limit)
        driver = ConversationDriver(
            gateway=gw,
            model=model,
            max_tokens=max_tokens,
        )

        transcript = await driver.run(task, mode=mode)
        results[mode] = transcript

        log.info(
            "completed %s/%s: %d turns, final pressure=%s",
            task.name,
            mode,
            len(transcript.turns),
            transcript.turns[-1].pressure_zone if transcript.turns else "N/A",
        )

    return results


def save_transcripts(
    results: dict[str, Transcript],
    output_dir: str | Path,
) -> list[Path]:
    """Write transcripts to JSON files. Returns paths written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for mode, transcript in results.items():
        filename = f"{transcript.task_name}_{mode}_{transcript.model}.json"
        path = output_dir / filename
        path.write_text(json.dumps(asdict(transcript), indent=2, default=str))
        paths.append(path)
        log.info("wrote %s (%d turns)", path, len(transcript.turns))

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Tinkuy evaluation tasks",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_REGISTRY.keys()),
        help="Task to run",
    )
    parser.add_argument(
        "--modes",
        default="baseline,full",
        help="Comma-separated modes (default: baseline,full)",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model to use (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=200_000,
        help="Context window limit (default: 200000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64000,
        help="Max output tokens per turn (default: 64000)",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Output directory for transcripts (default: eval_results)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="[eval] %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    task_factory = TASK_REGISTRY[args.task]
    task = task_factory()
    modes = [m.strip() for m in args.modes.split(",")]

    results = asyncio.run(
        run_task(
            task=task,
            model=args.model,
            modes=modes,
            context_limit=args.context_limit,
            max_tokens=args.max_tokens,
        )
    )

    save_transcripts(results, args.output_dir)


if __name__ == "__main__":
    main()
