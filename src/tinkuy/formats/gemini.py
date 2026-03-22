"""Gemini API payload synthesis and ingestion.

GeminiLiveAdapter: Synthesis (Projection -> Gemini API)
GeminiInboundAdapter: Request parsing (Gemini API -> Tinkuy Events)
GeminiResponseIngester: Response parsing (Gemini API -> Projection)

THIS IS THE ANTI-PROXY-GRAVITY BOUNDARY.
"""

from __future__ import annotations
import json
import logging
from typing import Any
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

log = logging.getLogger("tinkuy.formats.gemini")

class GeminiLiveAdapter:
    """Synthesizes Gemini API payloads from the projection."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    def synthesize_request(self) -> dict[str, Any]:
        """Synthesize a complete Gemini GenerateContentRequest."""
        projection = self.orchestrator.projection
        
        # R1 → system_instruction
        system_instruction = self._collect_system(projection)
        
        # R2 + R3 + R4 → contents
        contents = self._collect_contents(projection)
        
        # R0 → tools
        tools = self._collect_tools(projection)

        request = {
            "contents": contents,
        }
        if system_instruction:
            request["system_instruction"] = system_instruction
        if tools:
            request["tools"] = tools
            
        return request

    def _collect_system(self, projection: Projection) -> dict[str, Any] | None:
        """Collect system instruction from R1."""
        parts = []
        region = projection.region(RegionID.SYSTEM)
        for block in region.blocks:
            if block.status == ContentStatus.PRESENT:
                parts.append({"text": block.content})
        
        if not parts:
            return None
        return {"parts": parts}

    def _collect_contents(self, projection: Projection) -> list[dict[str, Any]]:
        """Collect conversation contents from R2, R3, R4."""
        contents = []
        
        current_role = None
        current_parts = []

        for rid in (RegionID.DURABLE, RegionID.EPHEMERAL, RegionID.CURRENT):
            region = projection.region(rid)
            for block in region.blocks:
                if block.kind == ContentKind.SYSTEM:
                    continue
                
                role = self._get_gemini_role(block)
                part = self._block_to_part(block)
                
                if role != current_role:
                    if current_parts:
                        contents.append({"role": current_role, "parts": current_parts})
                    current_role = role
                    current_parts = []
                
                current_parts.append(part)
        
        if current_parts:
            contents.append({"role": current_role, "parts": current_parts})
            
        return contents

    def _get_gemini_role(self, block: ContentBlock) -> str:
        if block.kind == ContentKind.TOOL_RESULT:
            return "user"
        if "assistant" in block.label or block.kind == ContentKind.TENSOR:
            return "model"
        return "user"

    def _block_to_part(self, block: ContentBlock) -> dict[str, Any]:
        if block.status == ContentStatus.AVAILABLE:
            return {"text": f"[tensor:{block.handle[:8]} — {block.label}]"}
        
        if block.kind == ContentKind.TOOL_RESULT:
            return {
                "function_response": {
                    "name": block.metadata.get("tool_name", "tool"),
                    "response": {"result": block.content}
                }
            }
        
        # If it was originally a tool call
        if block.metadata.get("function_call"):
             fc = block.metadata.get("function_call")
             return {
                 "function_call": {
                     "name": fc.get("name"),
                     "args": fc.get("args", {})
                 }
             }

        return {"text": block.content}

    def _collect_tools(self, projection: Projection) -> list[dict[str, Any]]:
        return []

    def synthesize_page_table(self) -> str:
        # We can implement a Gemini-specific page table if needed
        return ""


class GeminiInboundAdapter:
    """Parses Gemini API requests into Tinkuy events."""

    def parse_request(self, body: dict[str, Any]) -> list[InboundEvent]:
        """Parse a Gemini GenerateContentRequest into InboundEvents."""
        events = []
        contents = body.get("contents", [])
        
        # We process the messages from the client.
        # In a true gateway, we focus on the new content being added.
        if contents and contents[-1].get("role") == "user":
            last_turn = contents[-1]
            for part in last_turn.get("parts", []):
                if "text" in part:
                    events.append(InboundEvent(
                        type=EventType.USER_MESSAGE,
                        content=part["text"],
                        label="user",
                    ))
                elif "function_response" in part:
                    fr = part["function_response"]
                    tool_name = fr.get("name", "tool")
                    events.append(InboundEvent(
                        type=EventType.TOOL_RESULT,
                        content=json.dumps(fr.get("response", {})),
                        label=tool_name,
                        metadata={"tool_name": tool_name},
                    ))
        
        return events


class GeminiResponseIngester:
    """Parses Gemini API responses into projection mutations."""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator

    def ingest_response(self, response_json: dict[str, Any]) -> Any:
        """Ingest a complete Gemini GenerateContentResponse."""
        # Note: Gemini often sends a list of candidates
        candidates = response_json.get("candidates", [])
        if not candidates:
            return None
        
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        text_parts = []
        content_blocks = []
        
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
                content_blocks.append({"type": "text", "text": part["text"]})
            elif "function_call" in part:
                fc = part["function_call"]
                content_blocks.append({
                    "type": "tool_use", # Mapping to internal tool_use kind
                    "name": fc.get("name"),
                    "input": fc.get("args", {}),
                    "metadata": {"function_call": fc}
                })

        full_text = "\n".join(text_parts)
        
        # Extract signals (if any) from the text
        from tinkuy.gateway.harness import extract_signals, strip_signals
        signals = extract_signals(full_text)
        clean_text = strip_signals(full_text)

        parsed_signals = []
        for s in signals:
            signal_type = s.get("type", "").lower()
            handle = s.get("handle", "")
            if signal_type == "release":
                parsed_signals.append(ResponseSignal(
                    type=ResponseSignalType.RELEASE,
                    handle=handle,
                    tensor_content=s.get("tensor_content"),
                    declared_losses=s.get("declared_losses"),
                ))
            elif signal_type == "retain":
                parsed_signals.append(ResponseSignal(
                    type=ResponseSignalType.RETAIN,
                    handle=handle,
                ))
            elif signal_type == "recall":
                parsed_signals.append(ResponseSignal(
                    type=ResponseSignalType.RECALL,
                    handle=handle,
                ))

        return self.orchestrator.ingest_response(
            content=clean_text,
            label="assistant",
            signals=parsed_signals,
            content_blocks=content_blocks,
        )
