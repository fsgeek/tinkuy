"""Streaming buffer simulator — validates chunking invariance.

Feeds canned SSE streams through the StreamBuffer with different
chunking strategies and validates that the reconstructed message
is identical regardless of how bytes arrive.

Run: python -m tinkuy.stream_sim
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass

from tinkuy.stream import (
    BlockType,
    ReconstructedBlock,
    ReconstructedMessage,
    StreamBuffer,
    build_sse_stream,
)


# --- Canned fixtures ---


def _fixture_text_only() -> tuple[str, bytes]:
    """Simple text-only response."""
    return "text_only", build_sse_stream([
        {"type": "message_start", "message": {
            "id": "msg_text", "model": "claude-3", "role": "assistant",
            "usage": {"input_tokens": 100},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Hello "}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "world!"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])


def _fixture_thinking_text() -> tuple[str, bytes]:
    """Thinking block followed by text block."""
    return "thinking_text", build_sse_stream([
        {"type": "message_start", "message": {
            "id": "msg_think", "model": "claude-opus-4-6",
            "role": "assistant", "usage": {"input_tokens": 5000},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta",
                   "thinking": "Let me analyze this carefully."}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta",
                   "signature": "EvECCkYI...signature..."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "text_delta",
                   "text": "Based on my analysis, the answer is 42."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta",
         "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 50}},
        {"type": "message_stop"},
    ])


def _fixture_full_claude_code() -> tuple[str, bytes]:
    """Full Claude Code pattern: thinking + text + tool_use."""
    return "full_claude_code", build_sse_stream([
        {"type": "message_start", "message": {
            "id": "msg_full", "model": "claude-opus-4-6",
            "role": "assistant",
            "usage": {"input_tokens": 12000},
        }},
        # Thinking
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta",
                   "thinking": "I need to read the file first. "}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta",
                   "thinking": "Then I'll analyze the contents."}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta",
                   "signature": "sig_opaque_bytes_here"}},
        {"type": "content_block_stop", "index": 0},
        # Text
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "text_delta",
                   "text": "Let me read the configuration."}},
        {"type": "content_block_stop", "index": 1},
        # Tool use
        {"type": "content_block_start", "index": 2,
         "content_block": {"type": "tool_use", "id": "toolu_abc123",
                           "name": "Read"}},
        {"type": "content_block_delta", "index": 2,
         "delta": {"type": "input_json_delta",
                   "partial_json": '{"file_'}},
        {"type": "content_block_delta", "index": 2,
         "delta": {"type": "input_json_delta",
                   "partial_json": 'path": "/etc/'}},
        {"type": "content_block_delta", "index": 2,
         "delta": {"type": "input_json_delta",
                   "partial_json": 'hosts"}'}},
        {"type": "content_block_stop", "index": 2},
        # End
        {"type": "message_delta",
         "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 300}},
        {"type": "message_stop"},
    ])


def _fixture_multi_tool() -> tuple[str, bytes]:
    """Multiple tool_use blocks in one response."""
    return "multi_tool", build_sse_stream([
        {"type": "message_start", "message": {
            "id": "msg_multi", "model": "claude-opus-4-6",
            "role": "assistant", "usage": {"input_tokens": 8000},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta",
                   "text": "I'll check both files."}},
        {"type": "content_block_stop", "index": 0},
        # Tool 1
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_1",
                           "name": "Read"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta",
                   "partial_json": '{"file_path": "/a.txt"}'}},
        {"type": "content_block_stop", "index": 1},
        # Tool 2
        {"type": "content_block_start", "index": 2,
         "content_block": {"type": "tool_use", "id": "toolu_2",
                           "name": "Grep"}},
        {"type": "content_block_delta", "index": 2,
         "delta": {"type": "input_json_delta",
                   "partial_json": '{"pattern": "TODO", "path": "/b.txt"}'}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta",
         "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 100}},
        {"type": "message_stop"},
    ])


def _fixture_with_pings() -> tuple[str, bytes]:
    """Stream with ping events interspersed."""
    return "with_pings", build_sse_stream([
        {"type": "message_start", "message": {
            "id": "msg_ping", "model": "claude-3", "role": "assistant",
            "usage": {"input_tokens": 100},
        }},
        {"type": "ping"},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "ping"},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "pong"}},
        {"type": "ping"},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])


def _fixture_redacted_thinking() -> tuple[str, bytes]:
    """Response with a redacted thinking block."""
    return "redacted_thinking", build_sse_stream([
        {"type": "message_start", "message": {
            "id": "msg_redact", "model": "claude-opus-4-6",
            "role": "assistant", "usage": {"input_tokens": 100},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "redacted_thinking"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "text_delta", "text": "Here is the answer."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta",
         "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 10}},
        {"type": "message_stop"},
    ])


ALL_FIXTURES = [
    _fixture_text_only,
    _fixture_thinking_text,
    _fixture_full_claude_code,
    _fixture_multi_tool,
    _fixture_with_pings,
    _fixture_redacted_thinking,
]


# --- Chunking strategies ---


def chunk_byte_at_a_time(data: bytes) -> list[bytes]:
    return [data[i:i + 1] for i in range(len(data))]


def chunk_line_at_a_time(data: bytes) -> list[bytes]:
    chunks = []
    for line in data.split(b"\n"):
        chunks.append(line + b"\n")
    return chunks


def chunk_random(data: bytes, seed: int = 42) -> list[bytes]:
    rng = random.Random(seed)
    chunks = []
    pos = 0
    while pos < len(data):
        size = rng.randint(1, min(512, len(data) - pos))
        chunks.append(data[pos:pos + size])
        pos += size
    return chunks


def chunk_whole(data: bytes) -> list[bytes]:
    return [data]


def chunk_adversarial(data: bytes, seed: int = 99) -> list[bytes]:
    """Split at the worst places: mid-UTF-8, mid-keyword, mid-JSON."""
    rng = random.Random(seed)
    chunks = []
    pos = 0
    while pos < len(data):
        # Small chunks to maximize boundary problems
        size = rng.randint(1, min(7, len(data) - pos))
        chunks.append(data[pos:pos + size])
        pos += size
    return chunks


ALL_STRATEGIES = {
    "whole": chunk_whole,
    "byte_at_a_time": chunk_byte_at_a_time,
    "line_at_a_time": chunk_line_at_a_time,
    "random": chunk_random,
    "adversarial": chunk_adversarial,
}


# --- Comparison ---


def _msg_fingerprint(msg: ReconstructedMessage) -> dict:
    """Extract a comparable fingerprint from a message."""
    return {
        "id": msg.id,
        "model": msg.model,
        "role": msg.role,
        "stop_reason": msg.stop_reason,
        "usage": dict(msg.usage),
        "blocks": [
            {
                "index": b.index,
                "type": b.block_type.value,
                "text": b.text,
                "thinking": b.thinking,
                "signature": b.signature,
                "tool_id": b.tool_id,
                "tool_name": b.tool_name,
                "input_parsed": b.input_parsed,
            }
            for b in msg.blocks
        ],
    }


# --- Main ---


def run_sim() -> bool:
    """Run the simulator. Returns True if all tests pass."""
    all_pass = True

    for fixture_fn in ALL_FIXTURES:
        name, stream_data = fixture_fn()

        # Parse with whole-stream first to get reference
        buf_ref = StreamBuffer()
        buf_ref.feed(stream_data)
        ref_msg = buf_ref.finish()
        ref_fp = _msg_fingerprint(ref_msg)

        for strat_name, chunker in ALL_STRATEGIES.items():
            chunks = chunker(stream_data)
            buf = StreamBuffer()
            for chunk in chunks:
                buf.feed(chunk)
            try:
                msg = buf.finish()
            except Exception as e:
                print(f"  FAIL {name}/{strat_name}: {e}")
                all_pass = False
                continue

            fp = _msg_fingerprint(msg)
            if fp != ref_fp:
                print(f"  FAIL {name}/{strat_name}: fingerprint mismatch")
                all_pass = False
            else:
                print(f"  ok   {name}/{strat_name}")

    return all_pass


def main() -> None:
    print("tinkuy stream simulator\n")
    passed = run_sim()
    print()
    if passed:
        print("ALL PASSED")
    else:
        print("SOME FAILURES")
        sys.exit(1)


if __name__ == "__main__":
    main()
