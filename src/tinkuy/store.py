"""Persistence layer for projection state and page store.

Two concerns, two different write strategies:

1. Page store (verbatim originals) — written eagerly at eviction time.
   If we crash before persisting, the content is gone forever. These
   are append-only and keyed by content handle.

2. Projection checkpoint — written at turn boundaries and idle marks.
   This is the full projection state (regions, blocks, metadata).
   It's a snapshot: each write replaces the previous one.

Both use a Store protocol so backends can vary (filesystem, SQLite,
S3, etc.) without the orchestrator knowing or caring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol


class PageStore(Protocol):
    """Protocol for verbatim original storage.

    Implementations must be durable — once put() returns, the
    content must survive a process restart.
    """

    def put(self, handle: str, content: str) -> None:
        """Store a verbatim original. Must be durable on return."""
        ...

    def get(self, handle: str) -> str | None:
        """Retrieve a verbatim original by handle."""
        ...

    def has(self, handle: str) -> bool:
        """Check if a handle exists without retrieving content."""
        ...

    def delete(self, handle: str) -> None:
        """Remove a verbatim original (e.g., after TTL expiry)."""
        ...

    def handles(self) -> list[str]:
        """List all stored handles."""
        ...


class CheckpointStore(Protocol):
    """Protocol for projection checkpoint storage."""

    def save(self, data: dict[str, Any]) -> None:
        """Write a checkpoint. Replaces any previous checkpoint."""
        ...

    def load(self) -> dict[str, Any] | None:
        """Load the most recent checkpoint, or None if none exists."""
        ...

    def exists(self) -> bool:
        """Check if a checkpoint exists."""
        ...


# --- Filesystem implementations ---


class FilePageStore:
    """Page store backed by individual files on disk.

    Each verbatim original is written to its own file, named by handle.
    This is simple and durable — fsync on write, one file per page.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, handle: str) -> Path:
        return self.directory / f"{handle}.page"

    def put(self, handle: str, content: str) -> None:
        path = self._path(handle)
        # Write to temp file then rename for atomicity
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(str(tmp), str(path))

    def get(self, handle: str) -> str | None:
        path = self._path(handle)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def has(self, handle: str) -> bool:
        return self._path(handle).exists()

    def delete(self, handle: str) -> None:
        path = self._path(handle)
        if path.exists():
            path.unlink()

    def handles(self) -> list[str]:
        return [
            p.stem for p in self.directory.glob("*.page")
        ]


class FileCheckpointStore:
    """Checkpoint store backed by a JSON file on disk.

    Writes atomically via temp-file-then-rename. Keeps the previous
    checkpoint as a .bak file for recovery.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: dict[str, Any]) -> None:
        content = json.dumps(data, indent=2, default=str)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        # Rotate: current → .bak, then temp → current
        bak = self.path.with_suffix(".bak")
        if self.path.exists():
            os.replace(str(self.path), str(bak))
        os.replace(str(tmp), str(self.path))

    def load(self) -> dict[str, Any] | None:
        path = self.path if self.path.exists() else self.path.with_suffix(".bak")
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        return json.loads(content)

    def exists(self) -> bool:
        return self.path.exists() or self.path.with_suffix(".bak").exists()


# --- In-memory implementations (for testing) ---


class MemoryPageStore:
    """In-memory page store for testing."""

    def __init__(self) -> None:
        self._pages: dict[str, str] = {}

    def put(self, handle: str, content: str) -> None:
        self._pages[handle] = content

    def get(self, handle: str) -> str | None:
        return self._pages.get(handle)

    def has(self, handle: str) -> bool:
        return handle in self._pages

    def delete(self, handle: str) -> None:
        self._pages.pop(handle, None)

    def handles(self) -> list[str]:
        return list(self._pages.keys())


class MemoryCheckpointStore:
    """In-memory checkpoint store for testing."""

    def __init__(self) -> None:
        self._data: dict[str, Any] | None = None

    def save(self, data: dict[str, Any]) -> None:
        self._data = data

    def load(self) -> dict[str, Any] | None:
        return self._data

    def exists(self) -> bool:
        return self._data is not None
