"""Tensor protocol for the wire — tool callback variant.

State updates travel as structured tool calls (_tinkuy_state_update),
not as XML embedded in response text. The gateway injects the tool,
intercepts the call, and strips it from the client-visible stream.

The tool_result return channel carries harness feedback back to the
model — memory pressure, curation advisories, cycle confirmation.

The Pydantic schema uses extra='allow': the model can add fields we
haven't defined. We carry them faithfully and log them.
"""

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Pydantic schema — the tool's input_schema
# ---------------------------------------------------------------------------

class Strand(BaseModel):
    title: str
    content: str = ""
    depends_on: list[str] = []
    key_claims: list[dict] = []
    integration_losses: list[str] = []


class Loss(BaseModel):
    what_was_lost: str = ""
    why: str = ""
    shed_from: str = ""
    category: str = "context_pressure"


class Tension(BaseModel):
    tension_id: str
    framings: list[dict] = []
    cycles_held: int = 0
    touches_strands: list[str] = []
    what_would_collapse_it: str = ""


class MemoryAction(BaseModel):
    action: str   # "summarize" | "release" | "pin"
    id: str       # memory object id (e.g. "m3")
    content: str = ""   # summary text (for summarize)
    reason: str = ""    # why (for release)


class StateUpdate(BaseModel):
    model_config = ConfigDict(extra="allow")

    updated_regions: list[str]

    # Known cognitive regions — typed, optional
    strands: list[Strand] | None = None
    declared_losses: list[Loss] | None = None
    open_questions: list[str] | None = None
    unresolved_tensions: list[Tension] | None = None
    instructions_for_next: str | None = None
    overall_truth: float | None = None
    overall_indeterminacy: float | None = None
    overall_falsity: float | None = None
    feedback_to_harness: dict | None = None

    # Memory curation — unified into the same update
    memory_actions: list[MemoryAction] | None = None


# ---------------------------------------------------------------------------
# Tool definition — injected into the client's tool list
# ---------------------------------------------------------------------------

TOOL_NAME = "_tinkuy_state_update"

_TOOL_DEFINITION: dict | None = None


def get_state_update_tool() -> dict:
    """Return the tool definition dict for _tinkuy_state_update.

    Cached after first call — the schema is static.
    """
    global _TOOL_DEFINITION
    if _TOOL_DEFINITION is not None:
        return _TOOL_DEFINITION

    StateUpdate.model_rebuild()
    _TOOL_DEFINITION = {
        "name": TOOL_NAME,
        "description": (
            "Update your cognitive state. Call at the end of each "
            "conversational turn (not during tool chains). Fields you "
            "don't include in updated_regions are preserved unchanged."
        ),
        "input_schema": StateUpdate.model_json_schema(),
    }
    return _TOOL_DEFINITION


# ---------------------------------------------------------------------------
# System prompt fragments
# ---------------------------------------------------------------------------

STATE_PROTOCOL = """\
You are operating as a stateful processor. Your state is shown below
as JSON — it contains your accumulated cognitive understanding from
all prior turns.

You will see:
- Your current state (strands, questions, tensions, etc.)
- Prior tool outputs as labeled memory objects
- The current turn (user message + any tool chain in progress)

You will NOT see prior user messages or prior assistant responses.

AT THE END OF EACH CONVERSATIONAL TURN (not during tool chains), call
_tinkuy_state_update to persist what you've learned. Declare which
regions you're updating via updated_regions. Regions you don't list
are carried forward unchanged.

Your cognitive state contains:
- **strands**: thematic threads of accumulated reasoning (integrated, not appended).
  Each strand has a depends_on list naming strands it builds on.
- **declared_losses**: what you actively chose to drop this cycle, and why.
  Each loss has a shed_from field linking it to the source strand.
- **open_questions**: unresolved questions (curate these — drop stale ones).
- **unresolved_tensions**: live superpositions you are actively holding.
  Competing framings where you haven't collapsed to one interpretation.
  Each tension has weighted framings and a collapse condition.
  Tensions persist until you resolve them.
- **instructions_for_next**: branch prediction for the next cycle.
- **feedback_to_harness**: signal the harness about your curation process.

On the FIRST cycle (no prior state), initialize all regions.

Do not be precious about your state. It is working memory, not a monument.
Update when the conversation warrants it. Leave it alone when it doesn't.

When you rewrite a strand, declare what didn't survive the integration in
the strand's integration_losses list.

Distinguish between empirical findings and your own speculation. When
consolidating strands, empirical findings are load-bearing — keep the
specific numbers and results even if you reorganize the framing.

You can also curate memory objects via the memory_actions field:
- summarize: replace full content with your summary
- release: drop entirely (gone, no recall)
- pin: mark as important

You may add fields beyond the known regions. They will be carried
faithfully. We don't interpret them yet."""


def build_tensor_system_block(
    tensor: dict | None,
    cycle: int,
    tool_cycle: bool = False,
) -> str:
    """Build the system prompt block containing protocol + state as JSON.

    When tool_cycle=True, the state is included read-only (so the model
    knows its own state) but the update protocol is omitted.
    Harness feedback travels via the tool_result channel, not here.
    """
    if tool_cycle:
        parts = [
            "You are mid-task (tool chain in progress). Your state is "
            "shown below for reference — do NOT call _tinkuy_state_update "
            "now. Focus on the task. You will update your state when the "
            "user's turn completes.",
        ]
        if tensor is not None:
            parts.append("")
            parts.append(f"## Your current state (cycle {cycle})\n")
            parts.append(_tensor_to_json(tensor))
        return "\n".join(parts)

    parts = [STATE_PROTOCOL, ""]

    if tensor is not None:
        parts.append(f"## Your current state (cycle {cycle})\n")
        parts.append(_tensor_to_json(tensor))
    else:
        parts.append(
            "This is cycle 1 — no prior state. Initialize all regions."
        )

    return "\n".join(parts)


def _tensor_to_json(tensor: dict) -> str:
    """Serialize tensor state as formatted JSON for the system prompt."""
    return json.dumps(tensor, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# State update processing — parse and convert to update dict
# ---------------------------------------------------------------------------

def _deserialize_string_fields(tool_input: dict) -> dict:
    """Models sometimes send nested structures as JSON-encoded strings.

    Walk the input dict and attempt json.loads on any string value that
    looks like a JSON array or object.  This lets Pydantic see actual
    lists/dicts instead of opaque strings.
    """
    out = {}
    for k, v in tool_input.items():
        if isinstance(v, str) and v.lstrip().startswith(("[", "{")):
            # Models sometimes produce trailing commas or whitespace after
            # the closing bracket/brace — strip before parsing.
            cleaned = v.strip().rstrip(",").strip()
            try:
                out[k] = json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                log.warning(
                    "field %r looks like JSON but failed to parse: %s",
                    k, v[:200],
                )
                out[k] = v
        else:
            out[k] = v
    return out


def parse_state_update(tool_input: dict) -> dict:
    """Validate and convert a tool_use input dict to an update dict.

    Parses through the Pydantic StateUpdate model for validation, then
    converts to a plain dict for _apply_updates. Extra fields (unknown
    to the schema) are preserved.
    """
    cleaned = _deserialize_string_fields(tool_input)
    update = StateUpdate.model_validate(cleaned)
    return update.model_dump(exclude_none=True)


def build_harness_feedback(
    cycle: int,
    feedback: list[str],
    memory_objects: list[Any] | None = None,
) -> dict:
    """Build the tool_result payload for the state update.

    This is what the model sees as the response to its
    _tinkuy_state_update call.
    """
    result: dict[str, Any] = {
        "status": "ok",
        "cycle": cycle,
    }
    if feedback:
        result["feedback"] = feedback
    if memory_objects:
        n_objects = len(memory_objects)
        memory_tokens = sum(m.tokens for m in memory_objects)
        n_stale = sum(
            1 for m in memory_objects
            if not m.pinned and m.state == "full"
            and (cycle - m.turn) >= 10
        )
        result["memory_summary"] = {
            "objects": n_objects,
            "tokens": memory_tokens,
            "stale_count": n_stale,
        }
    return result


# ---------------------------------------------------------------------------
# Session tag — self-propagating identity in the message stream
# ---------------------------------------------------------------------------

SESSION_TAG_PATTERN = re.compile(
    r"<tinkuy-session\s+id=\"([^\"]+)\"\s*/>"
)


def make_session_tag(session_id: str) -> str:
    """Create a session tag to inject into the first response."""
    return f'<tinkuy-session id="{session_id}"/>'


def extract_session_tag(messages: list[dict]) -> str | None:
    """Find a session tag in the conversation history.

    Scans assistant messages for <tinkuy-session id="..."/>. The tag
    propagates because the client echoes conversation history back.
    Returns the session ID or None if no tag found.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            match = SESSION_TAG_PATTERN.search(content)
            if match:
                return match.group(1)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                elif isinstance(block, str):
                    text = block
                else:
                    continue
                match = SESSION_TAG_PATTERN.search(text)
                if match:
                    return match.group(1)
    return None
