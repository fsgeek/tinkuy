"""Projection region data structures.

The projection is a set of structured regions, each with different
stability and caching characteristics. Regions are the source of truth —
API payloads are generated FROM the projection, never from the client's
message history.

Regions:
  R0: Tools      — tool definitions, never mutates
  R1: System     — system instructions + absorbed client context
  R2: Durable    — model-curated tensors with declared losses
  R3: Ephemeral  — recent content aging toward eviction or promotion
  R4: Current    — current turn, replaced every turn
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class RegionID(Enum):
    """Projection region identifiers, ordered by stability."""
    TOOLS = 0       # R0: tool definitions
    SYSTEM = 1      # R1: system instructions
    DURABLE = 2     # R2: model-curated tensors
    EPHEMERAL = 3   # R3: recent content, mutable
    CURRENT = 4     # R4: current turn


class ContentStatus(Enum):
    """Lifecycle status of content within a region."""
    PRESENT = auto()           # live in the projection
    AVAILABLE = auto()         # evicted, tensor in R2, recallable
    PENDING_REMOVAL = auto()   # nominated for eviction, awaiting tensor


class ContentKind(Enum):
    """Classification of content for eviction/promotion decisions."""
    FILE = auto()              # file read — faultable
    TOOL_RESULT = auto()       # command/search output — ephemeral
    CONVERSATION = auto()      # user/assistant message
    TENSOR = auto()            # model-curated projection
    SYSTEM = auto()            # system-level content


@dataclass
class AccessRecord:
    """Tracks access patterns for eviction/promotion decisions.

    This is the 'accessed bit' — we collect data now and optimize
    later as workload patterns emerge.
    """
    created_turn: int = 0
    last_access_turn: int = 0
    access_count: int = 0
    fault_count: int = 0       # times recalled after eviction

    def touch(self, turn: int) -> None:
        """Record an access."""
        self.last_access_turn = turn
        self.access_count += 1

    def record_fault(self, turn: int) -> None:
        """Record a fault (recall after eviction)."""
        self.fault_count += 1
        self.touch(turn)

    @property
    def age(self) -> int:
        """Turns since last access (requires current turn from caller)."""
        return 0  # caller must compute: current_turn - last_access_turn


@dataclass
class ContentBlock:
    """A unit of content within a projection region.

    Content blocks are the atoms of the projection. They carry their
    own access records, metadata, and lifecycle state.
    """
    handle: str                           # 8-char hex, content-addressed
    kind: ContentKind
    label: str                            # human-readable (~40 chars)
    content: str                          # the actual content
    status: ContentStatus = ContentStatus.PRESENT
    region: RegionID = RegionID.EPHEMERAL
    size_tokens: int = 0                  # approximate
    access: AccessRecord = field(default_factory=AccessRecord)
    metadata: dict[str, Any] = field(default_factory=dict)

    # When evicted, the tensor that replaced this content
    tensor_handle: str | None = None

    @staticmethod
    def compute_handle(content: str) -> str:
        """Content-addressed handle: first 8 hex chars of SHA-256."""
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    @classmethod
    def create(
        cls,
        content: str,
        kind: ContentKind,
        label: str,
        region: RegionID = RegionID.EPHEMERAL,
        turn: int = 0,
        **metadata: Any,
    ) -> ContentBlock:
        """Create a new content block with computed handle."""
        handle = cls.compute_handle(content)
        # Rough token estimate: ~4 chars per token
        size_tokens = len(content) // 4
        return cls(
            handle=handle,
            kind=kind,
            label=label,
            content=content,
            region=region,
            size_tokens=size_tokens,
            access=AccessRecord(created_turn=turn, last_access_turn=turn),
            metadata=metadata,
        )


@dataclass
class RemovalNomination:
    """Advisory nomination for content removal.

    Multiple sources can nominate content for removal. The space
    scheduler decides what actually gets evicted and when.
    """
    handle: str                    # content block handle
    source: str                    # who nominated: "model", "gateway", "pressure", "client"
    reason: str                    # why
    timestamp: float = field(default_factory=time.time)
    tensor_provided: bool = False  # has the model provided a replacement tensor?


@dataclass
class Region:
    """A projection region containing content blocks.

    Regions are ordered by stability (R0 most stable, R4 least).
    The API adapter places cache breakpoints at region boundaries.
    """
    id: RegionID
    blocks: list[ContentBlock] = field(default_factory=list)
    nominations: list[RemovalNomination] = field(default_factory=list)

    @property
    def size_tokens(self) -> int:
        """Total tokens in this region."""
        return sum(
            b.size_tokens for b in self.blocks
            if b.status == ContentStatus.PRESENT
        )

    @property
    def waste_tokens(self) -> int:
        """Tokens in blocks marked for removal."""
        return sum(
            b.size_tokens for b in self.blocks
            if b.status == ContentStatus.PENDING_REMOVAL
        )

    @property
    def block_count(self) -> int:
        """Number of present blocks."""
        return sum(
            1 for b in self.blocks
            if b.status == ContentStatus.PRESENT
        )

    def add(self, block: ContentBlock) -> None:
        """Add a content block to this region."""
        block.region = self.id
        self.blocks.append(block)

    def find(self, handle: str) -> ContentBlock | None:
        """Find a block by handle."""
        for b in self.blocks:
            if b.handle == handle:
                return b
        return None

    def nominate_removal(self, handle: str, source: str, reason: str) -> None:
        """Nominate a block for removal. Advisory only."""
        self.nominations.append(
            RemovalNomination(handle=handle, source=source, reason=reason)
        )

    def remove(self, handle: str) -> ContentBlock | None:
        """Remove a block from the region. Returns the block."""
        for i, b in enumerate(self.blocks):
            if b.handle == handle:
                return self.blocks.pop(i)
        return None

    def present_blocks(self) -> list[ContentBlock]:
        """All blocks with PRESENT status, in order."""
        return [b for b in self.blocks if b.status == ContentStatus.PRESENT]


class Projection:
    """The complete projection — source of truth for the gateway.

    The projection contains all regions and provides the interface
    for the orchestrator to examine, mutate, and generate from.
    """

    def __init__(self, turn: int = 0) -> None:
        self.turn = turn
        self.regions: dict[RegionID, Region] = {
            rid: Region(id=rid) for rid in RegionID
        }
        # Page store: handle → verbatim original content
        # (Storage is cheap; always keep the original)
        self.page_store: dict[str, str] = {}

    def region(self, rid: RegionID) -> Region:
        """Get a region by ID."""
        return self.regions[rid]

    @property
    def total_tokens(self) -> int:
        """Total tokens across all regions."""
        return sum(r.size_tokens for r in self.regions.values())

    @property
    def waste_tokens(self) -> int:
        """Total waste tokens across all regions."""
        return sum(r.waste_tokens for r in self.regions.values())

    def add_content(
        self,
        content: str,
        kind: ContentKind,
        label: str,
        region: RegionID = RegionID.EPHEMERAL,
        **metadata: Any,
    ) -> ContentBlock:
        """Add content to a region. Returns the created block."""
        block = ContentBlock.create(
            content=content,
            kind=kind,
            label=label,
            region=region,
            turn=self.turn,
            **metadata,
        )
        self.regions[region].add(block)
        self.page_store[block.handle] = content
        return block

    def touch(self, handle: str) -> bool:
        """Record an access to a content block. Returns False if not found."""
        for region in self.regions.values():
            block = region.find(handle)
            if block is not None:
                block.access.touch(self.turn)
                return True
        return False

    def evict(self, handle: str, tensor: ContentBlock) -> bool:
        """Evict content, replacing it with a tensor in R2.

        The original content is always preserved in the page store.
        The tensor goes into R2. The evicted block's status changes
        to AVAILABLE.
        """
        for region in self.regions.values():
            block = region.find(handle)
            if block is None:
                continue
            block.status = ContentStatus.AVAILABLE
            block.tensor_handle = tensor.handle
            # Tensor goes into R2 (durable)
            self.regions[RegionID.DURABLE].add(tensor)
            return True
        return False

    def recall(self, handle: str) -> str | None:
        """Fault: recall evicted content from page store.

        Returns the verbatim original. Updates access records.
        """
        original = self.page_store.get(handle)
        if original is None:
            return None

        for region in self.regions.values():
            block = region.find(handle)
            if block is not None:
                block.access.record_fault(self.turn)
                block.status = ContentStatus.PRESENT
                block.content = original
                return original

        return original

    def advance_turn(self) -> None:
        """Advance to the next turn."""
        self.turn += 1

    def snapshot(self) -> dict[str, Any]:
        """Serialize projection state for checkpointing."""
        return {
            "turn": self.turn,
            "regions": {
                rid.name: {
                    "blocks": [
                        {
                            "handle": b.handle,
                            "kind": b.kind.name,
                            "label": b.label,
                            "status": b.status.name,
                            "size_tokens": b.size_tokens,
                            "access": {
                                "created_turn": b.access.created_turn,
                                "last_access_turn": b.access.last_access_turn,
                                "access_count": b.access.access_count,
                                "fault_count": b.access.fault_count,
                            },
                            "tensor_handle": b.tensor_handle,
                            "metadata": b.metadata,
                        }
                        for b in region.blocks
                    ],
                    "nominations": [
                        {
                            "handle": n.handle,
                            "source": n.source,
                            "reason": n.reason,
                            "tensor_provided": n.tensor_provided,
                        }
                        for n in region.nominations
                    ],
                }
                for rid, region in self.regions.items()
            },
        }
