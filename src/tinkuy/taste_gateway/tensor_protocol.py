"""Tensor protocol for the wire.

Taste.py uses tool_use for structured tensor output. The gateway can't —
the client's tools must pass through. So the tensor uses XML signals
embedded in the response text. The model writes its response naturally,
then appends the tensor update as XML. The gateway strips the XML before
the client sees it.

The protocol instructions go into the system prompt. The tensor state
is presented as XML in the system prompt. Updates come back as XML in
the response.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any


# ---------------------------------------------------------------------------
# System prompt fragments
# ---------------------------------------------------------------------------

TENSOR_PROTOCOL = """\
You are operating as a stateful processor. Your tensor is your memory
for prior conversation. You will see:
- Your tensor (accumulated cognitive state from all prior turns)
- Prior tool outputs (what tools were called and what they returned)
- The current turn (user message + any tool chain in progress)

You will NOT see prior user messages or prior assistant responses — only
the tensor carries that. Tool outputs persist so you can reference prior
results without re-running tools.

YOUR RESPONSIBILITIES EACH CYCLE:
1. Read your tensor (your accumulated cognitive state)
2. Read the current input (user message, tool results, etc.)
3. Produce your response (text, tool calls — whatever the task requires)
4. Update your tensor to integrate everything you need from this cycle
5. Declare what you chose not to integrate (declared-losses)

If information from the current input matters for future cycles, you MUST
integrate it into your tensor now. There is no conversation history to
fall back on. The tensor is your only memory between cycles.

Your cognitive state contains:
- **strands**: thematic threads of accumulated reasoning (integrated, not appended).
  Each strand has a `depends-on` attribute listing titles of strands it builds on.
- **declared-losses**: what you actively chose to drop this cycle, and why.
  Each loss has a `shed-from` attribute linking it to the source strand.
- **open-questions**: unresolved questions (curate these — drop stale ones).
- **unresolved-tensions**: live superpositions you are actively holding.
  Competing framings where you haven't collapsed to one interpretation.
  Each tension has weighted framings and a collapse condition.
  Tensions persist until you resolve them.
- **instructions-for-next**: branch prediction for the next cycle.
- **feedback-to-harness**: signal the harness about your curation process.

The tensor is DEFAULT-STABLE. Declare which regions you are updating via
the `updated-regions` attribute. Regions you don't list are carried forward
unchanged by the harness. If you want to remove a strand, update strands
(with it removed) AND add a declared-loss.

On the FIRST cycle (no prior tensor), initialize all regions.

Do not be precious about the tensor. It is working memory, not a monument.
Update when the conversation warrants it. Leave it alone when it doesn't.

When you rewrite a strand, declare what didn't survive the integration in
an `integration-losses` element within the strand.

Distinguish between empirical findings and your own speculation. When
consolidating strands, empirical findings are load-bearing — keep the
specific numbers and results even if you reorganize the framing.

Emit the tensor update AFTER your response text, inside <yuyay-tensor> tags.
This update will be stripped from what the user sees — it is between you
and the wire.

<yuyay-tensor updated-regions="strands,declared-losses">
  <strands>
    <strand title="example_strand" depends-on="other_strand">
      <content>Integrated understanding of the topic...</content>
      <key-claim truth="0.8" indeterminacy="0.15" falsity="0.05">
        Specific assertion with epistemic values
      </key-claim>
      <integration-loss>Detail that was compressed away</integration-loss>
    </strand>
  </strands>
  <declared-losses>
    <loss shed-from="old_strand" category="context_pressure">
      <what>What was dropped</what>
      <why>Why it was dropped</why>
    </loss>
  </declared-losses>
</yuyay-tensor>

## Memory Management

Prior tool outputs are presented as labeled memory objects. Each has:
- **id**: reference handle (m1, m2, ...)
- **tool**: which tool produced it
- **label**: what it contains (file path, command, etc.)
- **tokens**: approximate size
- **turn**: which cycle created it
- **state**: "full" (original) or "summary" (compressed)

You can curate memory objects by emitting <yuyay-memory> AFTER your
response (alongside <yuyay-tensor> if you're also updating the tensor):

<yuyay-memory>
  <summarize id="m3">
    Your summary replacing the full content. Include key findings,
    specific numbers, and anything you might need to reference later.
  </summarize>
  <release id="m7">Why this is no longer needed</release>
  <pin id="m12"/>
</yuyay-memory>

- **summarize**: Replace full content with your summary. Use when you've
  integrated key findings into your tensor but may need specific details.
- **release**: Drop entirely. Use when fully integrated into tensor and
  you won't reference it again. Released objects are gone — no recall.
- **pin**: Mark as important. The harness won't suggest releasing it.

Memory curation is optional. Don't curate during tool chains — focus on
the task. Curate at turn boundaries when the harness advises it or when
you notice stale objects."""


def build_tensor_system_block(
    tensor: dict | None,
    cycle: int,
    feedback: list[str] | None = None,
    tool_cycle: bool = False,
) -> str:
    """Build the system prompt block containing protocol + tensor state.

    When tool_cycle=True, the tensor is included read-only (so the model
    knows its own state) but the update protocol and feedback are omitted.
    The model should focus on the task, not on curation. Tensor updates
    happen at turn boundaries, not during tool chains.
    """
    if tool_cycle:
        # Read-only: tensor state visible, no update protocol
        parts = [
            "You are mid-task (tool chain in progress). Your tensor is "
            "shown below for reference — do NOT update it now. Focus on "
            "the task. You will update your tensor when the user's turn "
            "completes.",
        ]
        if tensor is not None:
            parts.append("")
            parts.append(f"## Your current tensor (cycle {cycle})\n")
            parts.append(_tensor_to_xml(tensor))
        return "\n".join(parts)

    parts = [TENSOR_PROTOCOL, ""]

    if tensor is not None:
        parts.append(f"## Your current tensor (cycle {cycle})\n")
        parts.append(_tensor_to_xml(tensor))
    else:
        parts.append(
            "This is cycle 1 — no prior tensor. Initialize all regions."
        )

    if feedback:
        parts.append("\n## Harness Feedback\n")
        for f in feedback:
            parts.append(f)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tensor ↔ XML serialization
# ---------------------------------------------------------------------------

def _tensor_to_xml(tensor: dict) -> str:
    """Serialize tensor state as XML for the system prompt."""
    lines = [f'<yuyay-tensor-state cycle="{tensor.get("cycle", 0)}">']

    # Strands
    for strand in tensor.get("strands", []):
        deps = strand.get("depends_on", [])
        dep_attr = f' depends-on="{",".join(deps)}"' if deps else ""
        lines.append(f'  <strand title="{_esc(strand.get("title", ""))}"'
                     f'{dep_attr}>')
        lines.append(f'    <content>{_esc(strand.get("content", ""))}</content>')
        for claim in strand.get("key_claims", []):
            lines.append(
                f'    <key-claim truth="{claim.get("truth", 0)}" '
                f'indeterminacy="{claim.get("indeterminacy", 0)}" '
                f'falsity="{claim.get("falsity", 0)}">'
                f'{_esc(claim.get("text", ""))}</key-claim>'
            )
        lines.append("  </strand>")

    # Open questions
    questions = tensor.get("open_questions", [])
    if questions:
        lines.append("  <open-questions>")
        for q in questions:
            lines.append(f"    <question>{_esc(q)}</question>")
        lines.append("  </open-questions>")

    # Unresolved tensions
    tensions = tensor.get("unresolved_tensions", [])
    if tensions:
        lines.append("  <unresolved-tensions>")
        for t in tensions:
            touches = t.get("touches_strands", [])
            touches_attr = (f' touches-strands="{",".join(touches)}"'
                           if touches else "")
            lines.append(
                f'    <tension id="{_esc(t.get("tension_id", ""))}" '
                f'cycles-held="{t.get("cycles_held", 0)}"'
                f'{touches_attr}>'
            )
            for framing in t.get("framings", []):
                lines.append(
                    f'      <framing label="{_esc(framing.get("label", ""))}" '
                    f'weight="{framing.get("weight", 0.5)}">'
                    f'{_esc(framing.get("statement", ""))}</framing>'
                )
            collapse = t.get("what_would_collapse_it", "")
            if collapse:
                lines.append(
                    f"      <collapse-condition>{_esc(collapse)}</collapse-condition>"
                )
            lines.append("    </tension>")
        lines.append("  </unresolved-tensions>")

    # Instructions for next
    ifn = tensor.get("instructions_for_next", "")
    if ifn:
        lines.append(
            f"  <instructions-for-next>{_esc(ifn)}</instructions-for-next>"
        )

    # Epistemic state
    for key in ["overall_truth", "overall_indeterminacy", "overall_falsity"]:
        val = tensor.get(key)
        if val is not None:
            tag = key.replace("overall_", "")
            lines.append(f"  <{tag}>{val}</{tag}>")

    lines.append("</yuyay-tensor-state>")
    return "\n".join(lines)


def _esc(text: str) -> str:
    """Escape XML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ---------------------------------------------------------------------------
# Response parsing — extract tensor update from model output
# ---------------------------------------------------------------------------

_TENSOR_PATTERN = re.compile(
    r"<yuyay-tensor\b[^>]*>(.*?)</yuyay-tensor>",
    re.DOTALL,
)


def parse_tensor_update(response_text: str) -> tuple[str, dict | None]:
    """Extract tensor update from response, return (clean_text, update).

    The clean text has the <yuyay-tensor> block removed — this is what
    the client sees. The update dict follows taste.py semantics.

    Uses the LAST match — the model may echo example tags from the
    protocol instructions earlier in its response. The actual update
    is always appended at the end.

    Returns (response_text, None) if no tensor block found.
    """
    matches = list(_TENSOR_PATTERN.finditer(response_text))
    if not matches:
        return response_text, None

    # Try matches from last to first — the actual protocol output is
    # typically at the end. Earlier matches may be the model discussing
    # the protocol in conversation.
    for match in reversed(matches):
        full_xml = match.group(0)
        try:
            root = ET.fromstring(full_xml)
        except ET.ParseError:
            # Not valid XML — the model is mentioning the tag in
            # conversation, not emitting protocol output. Skip it.
            continue

        # Valid XML — this is a real tensor update. Strip it.
        clean = (
            response_text[:match.start()].rstrip()
            + response_text[match.end():]
        ).rstrip()

        # Extract updated-regions from the opening tag
        opening_tag = full_xml[:full_xml.index(">") + 1]
        regions_match = re.search(r'updated-regions="([^"]*)"', opening_tag)
        updated_regions = []
        if regions_match:
            updated_regions = [
                r.strip() for r in regions_match.group(1).split(",")
            ]

        update = _xml_to_tensor_update(root, updated_regions)
        return clean, update

    # All matches were invalid XML — leave the response untouched
    return response_text, None


def _xml_to_tensor_update(root: ET.Element, updated_regions: list[str]) -> dict:
    """Convert parsed XML element to a tensor update dict."""
    update: dict[str, Any] = {"updated_regions": updated_regions}

    # Strands
    strands = []
    for strand_el in root.findall(".//strand"):
        strand: dict[str, Any] = {
            "title": strand_el.get("title", ""),
            "content": "",
            "key_claims": [],
        }
        deps = strand_el.get("depends-on", "")
        if deps:
            strand["depends_on"] = [d.strip() for d in deps.split(",")]

        content_el = strand_el.find("content")
        if content_el is not None and content_el.text:
            strand["content"] = content_el.text.strip()

        for claim_el in strand_el.findall("key-claim"):
            claim = {
                "text": (claim_el.text or "").strip(),
                "truth": float(claim_el.get("truth", 0)),
                "indeterminacy": float(claim_el.get("indeterminacy", 0)),
                "falsity": float(claim_el.get("falsity", 0)),
            }
            strand["key_claims"].append(claim)

        integration_losses = []
        for loss_el in strand_el.findall("integration-loss"):
            if loss_el.text:
                integration_losses.append(loss_el.text.strip())
        if integration_losses:
            strand["integration_losses"] = integration_losses

        strands.append(strand)
    if strands:
        update["strands"] = strands

    # Declared losses
    losses = []
    for loss_el in root.findall(".//declared-losses/loss"):
        loss: dict[str, Any] = {
            "category": loss_el.get("category", "context_pressure"),
        }
        shed = loss_el.get("shed-from")
        if shed:
            loss["shed_from"] = shed
        what_el = loss_el.find("what")
        why_el = loss_el.find("why")
        if what_el is not None and what_el.text:
            loss["what_was_lost"] = what_el.text.strip()
        if why_el is not None and why_el.text:
            loss["why"] = why_el.text.strip()
        losses.append(loss)
    if losses:
        update["declared_losses"] = losses

    # Open questions
    questions = []
    for q_el in root.findall(".//open-questions/question"):
        if q_el.text:
            questions.append(q_el.text.strip())
    if questions:
        update["open_questions"] = questions

    # Unresolved tensions
    tensions = []
    for t_el in root.findall(".//unresolved-tensions/tension"):
        tension: dict[str, Any] = {
            "tension_id": t_el.get("id", ""),
            "framings": [],
            "cycles_held": int(t_el.get("cycles-held", 0)),
        }
        touches = t_el.get("touches-strands", "")
        if touches:
            tension["touches_strands"] = [s.strip() for s in touches.split(",")]

        for f_el in t_el.findall("framing"):
            framing = {
                "label": f_el.get("label", ""),
                "statement": (f_el.text or "").strip(),
                "weight": float(f_el.get("weight", 0.5)),
            }
            tension["framings"].append(framing)

        collapse_el = t_el.find("collapse-condition")
        if collapse_el is not None and collapse_el.text:
            tension["what_would_collapse_it"] = collapse_el.text.strip()

        tensions.append(tension)
    if tensions:
        update["unresolved_tensions"] = tensions

    # Instructions for next
    ifn_el = root.find("instructions-for-next")
    if ifn_el is not None and ifn_el.text:
        update["instructions_for_next"] = ifn_el.text.strip()

    # Epistemic values
    for tag, key in [
        ("truth", "overall_truth"),
        ("indeterminacy", "overall_indeterminacy"),
        ("falsity", "overall_falsity"),
    ]:
        el = root.find(tag)
        if el is not None and el.text:
            try:
                update[key] = float(el.text.strip())
            except ValueError:
                pass

    # Feedback to harness
    fb_requests = []
    fb_observations = []
    for el in root.findall(".//feedback-to-harness/request"):
        if el.text:
            fb_requests.append(el.text.strip())
    for el in root.findall(".//feedback-to-harness/observation"):
        if el.text:
            fb_observations.append(el.text.strip())
    if fb_requests or fb_observations:
        update["feedback_to_harness"] = {
            "requests": fb_requests,
            "process_observations": fb_observations,
        }

    return update


# ---------------------------------------------------------------------------
# Response parsing — extract memory curation signals from model output
# ---------------------------------------------------------------------------

_MEMORY_PATTERN = re.compile(
    r"<yuyay-memory>(.*?)</yuyay-memory>",
    re.DOTALL,
)


def parse_memory_signals(response_text: str) -> tuple[str, list[dict]]:
    """Extract memory curation signals from response, return (clean_text, signals).

    The clean text has the <yuyay-memory> block removed. Each signal is a dict:
        {"action": "summarize"|"release"|"pin", "id": "m3", "content": "..."}

    If a match isn't valid protocol content (no recognizable signals),
    it's treated as the model mentioning the tag in conversation and
    left in the response untouched.

    Returns (response_text, []) if no memory block found.
    """
    matches = list(_MEMORY_PATTERN.finditer(response_text))
    if not matches:
        return response_text, []

    for match in reversed(matches):
        xml_content = match.group(1)
        signals: list[dict] = []

        # Parse summarize signals
        for m in re.finditer(
            r'<summarize\s+id="([^"]+)">(.*?)</summarize>',
            xml_content, re.DOTALL,
        ):
            signals.append({
                "action": "summarize",
                "id": m.group(1),
                "content": m.group(2).strip(),
            })

        # Parse release signals
        for m in re.finditer(
            r'<release\s+id="([^"]+)">(.*?)</release>',
            xml_content, re.DOTALL,
        ):
            signals.append({
                "action": "release",
                "id": m.group(1),
                "reason": m.group(2).strip(),
            })

        # Parse pin signals
        for m in re.finditer(
            r'<pin\s+id="([^"]+)"\s*/>', xml_content,
        ):
            signals.append({
                "action": "pin",
                "id": m.group(1),
            })

        if not signals:
            # No recognizable signals — the model is mentioning
            # the tag in conversation. Leave it in the response.
            continue

        # Valid signals found — strip this block from the response
        clean = (
            response_text[:match.start()].rstrip()
            + response_text[match.end():]
        ).rstrip()
        return clean, signals

    # No valid memory blocks found
    return response_text, []


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
