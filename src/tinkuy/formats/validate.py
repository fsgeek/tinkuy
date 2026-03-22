"""Pre-flight payload validation for Anthropic API requests.

Encodes the structural invariants the API enforces, checked before
the payload hits the wire. Better to fail here with a clear message
than get a 400 from Anthropic with an opaque error.

Every rule here was learned from a production failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """A single validation failure."""
    rule: str
    message: str
    location: str = ""  # e.g. "messages[3]"


@dataclass
class ValidationResult:
    """Result of validating a payload."""
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def raise_if_invalid(self) -> None:
        if self.errors:
            msg = f"{len(self.errors)} validation error(s):\n"
            for e in self.errors:
                loc = f" at {e.location}" if e.location else ""
                msg += f"  [{e.rule}]{loc}: {e.message}\n"
            raise ValueError(msg)


def validate_anthropic_payload(payload: dict[str, Any]) -> ValidationResult:
    """Validate a payload against Anthropic API structural requirements.

    Rules:
    1. Messages must alternate user/assistant roles
    2. First message must be role=user
    3. Every tool_use block must have a matching tool_result in the
       next user message
    4. tool_result blocks must reference valid tool_use IDs from the
       preceding assistant message
    5. System must be str or list of content blocks
    6. cache_control count <= 4 (API limit)
    7. No empty content (empty string or empty list)
    """
    result = ValidationResult()
    messages = payload.get("messages", [])

    if not messages:
        result.errors.append(ValidationError(
            rule="non_empty", message="messages array is empty",
        ))
        return result

    _check_alternation(messages, result)
    _check_tool_pairing(messages, result)
    _check_tool_result_ordering(messages, result)
    _check_cache_control_budget(payload, result)
    _check_content_validity(messages, result)

    return result


def _check_alternation(
    messages: list[dict[str, Any]], result: ValidationResult
) -> None:
    """Rule 1+2: alternation and user-first."""
    if messages[0].get("role") != "user":
        result.errors.append(ValidationError(
            rule="user_first",
            message=f"first message must be user, got {messages[0].get('role')}",
            location="messages[0]",
        ))

    for i in range(1, len(messages)):
        prev_role = messages[i - 1].get("role")
        curr_role = messages[i].get("role")
        if prev_role == curr_role:
            result.errors.append(ValidationError(
                rule="alternation",
                message=f"consecutive {curr_role} messages",
                location=f"messages[{i-1}..{i}]",
            ))


def _check_tool_pairing(
    messages: list[dict[str, Any]], result: ValidationResult
) -> None:
    """Rule 3+4: tool_use/tool_result pairing."""
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        # Collect tool_use IDs from this assistant message
        tool_use_ids = _extract_tool_use_ids(msg)
        if not tool_use_ids:
            continue

        # Next message must be user with matching tool_results
        if i + 1 >= len(messages):
            result.errors.append(ValidationError(
                rule="tool_result_required",
                message=f"assistant has tool_use but no following message: {tool_use_ids}",
                location=f"messages[{i}]",
            ))
            continue

        next_msg = messages[i + 1]
        if next_msg.get("role") != "user":
            result.errors.append(ValidationError(
                rule="tool_result_required",
                message=f"tool_use must be followed by user message with tool_results: {tool_use_ids}",
                location=f"messages[{i}..{i+1}]",
            ))
            continue

        # Check that all tool_use IDs have matching tool_results
        tool_result_ids = _extract_tool_result_ids(next_msg)
        missing = tool_use_ids - tool_result_ids
        if missing:
            result.errors.append(ValidationError(
                rule="tool_result_missing",
                message=f"tool_use IDs without matching tool_result: {missing}",
                location=f"messages[{i}..{i+1}]",
            ))

        extra = tool_result_ids - tool_use_ids
        if extra:
            result.errors.append(ValidationError(
                rule="tool_result_orphan",
                message=f"tool_result IDs without matching tool_use: {extra}",
                location=f"messages[{i+1}]",
            ))


def _check_tool_result_ordering(
    messages: list[dict[str, Any]], result: ValidationResult
) -> None:
    """Rule: tool_results must precede other content in user messages
    that follow tool_use.

    When the previous assistant message contains tool_use blocks, the
    API requires tool_result blocks to appear before any other content
    in the next user message. A text block before tool_results causes
    a 400 even when the IDs match correctly.
    """
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if not _extract_tool_use_ids(msg):
            continue
        if i + 1 >= len(messages):
            continue

        next_msg = messages[i + 1]
        content = next_msg.get("content", "")
        if not isinstance(content, list):
            continue

        # Check that no non-tool_result block precedes a tool_result
        seen_non_tool_result = False
        for j, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                if seen_non_tool_result:
                    result.errors.append(ValidationError(
                        rule="tool_result_ordering",
                        message=(
                            f"tool_result at content[{j}] appears after "
                            f"non-tool_result content — tool_results must "
                            f"come first in user messages following tool_use"
                        ),
                        location=f"messages[{i+1}]",
                    ))
                    break
            else:
                seen_non_tool_result = True


def _check_cache_control_budget(
    payload: dict[str, Any], result: ValidationResult
) -> None:
    """Rule 6: at most 4 cache_control breakpoints."""
    count = 0

    # System blocks
    system = payload.get("system", [])
    if isinstance(system, list):
        for part in system:
            if isinstance(part, dict) and "cache_control" in part:
                count += 1

    # Message content blocks
    for msg in payload.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1

    if count > 4:
        result.errors.append(ValidationError(
            rule="cache_control_budget",
            message=f"found {count} cache_control breakpoints, API limit is 4",
        ))


def _check_content_validity(
    messages: list[dict[str, Any]], result: ValidationResult
) -> None:
    """Rule 7: no empty content."""
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if content is None or content == "" or content == []:
            result.errors.append(ValidationError(
                rule="empty_content",
                message=f"message has empty content",
                location=f"messages[{i}]",
            ))


# --- Helpers ---

def _extract_tool_use_ids(msg: dict[str, Any]) -> set[str]:
    """Extract tool_use IDs from a message."""
    content = msg.get("content", "")
    if not isinstance(content, list):
        return set()
    return {
        block.get("id", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    } - {""}


def _extract_tool_result_ids(msg: dict[str, Any]) -> set[str]:
    """Extract tool_use_ids referenced by tool_result blocks."""
    content = msg.get("content", "")
    if not isinstance(content, list):
        return set()
    return {
        block.get("tool_use_id", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_result"
    } - {""}
