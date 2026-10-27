"""Core data contracts shared across the pipeline (spec.md §0, §8).

`Candidate` is the in-memory unit that flows ingest -> reject -> cluster -> metrics ->
score -> persist. `Insight` is the engine-authored, structured claim that the LLM
narrates (the memory graph is an insight graph). Both are defined here in P0 so every
later phase shares ONE definition rather than retrofitting.

Design rules encoded here:
- A thumbnail is ephemeral working memory, never persisted as a full-resolution file.
- `original_ref` is a uuid (library) or path (hdd); it is NEVER an openable master path
  (opening a Photos master would trigger an iCloud download — forbidden).
- An Insight's `direction`/`magnitude` are committed by CODE before any LLM runs, so
  narration can never contradict them. `allowed_entities` is the exact set of names
  (above the recognition precision bar) and places narration may use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# PIL is a runtime dependency, but we avoid importing it at module import time so that
# non-image code (config, db, tests) does not pull in Pillow. The thumb is typed loosely.
ThumbImage = Any  # PIL.Image.Image at runtime


class Source(StrEnum):
    LIBRARY = "library"
    HDD = "hdd"


@dataclass
class Candidate:
    """A single photo flowing through the pipeline (spec.md §0)."""

    id: str                      # stable, source-prefixed: "lib:<uuid>" / "hdd:<sha1>"
    source: Source
    original_ref: str            # uuid (library) or path (hdd); never a master to open
    timestamp: float | None = None   # capture time (epoch); None if unknown
    thumb: ThumbImage | None = None  # ephemeral in-memory RGB; released after use
    exif: dict[str, Any] = field(default_factory=dict)
    # Library-only metadata (favorite, named persons, face attrs, albums, keywords).
    # None for HDD candidates (which have no native Library metadata).
    library_meta: dict[str, Any] | None = None

    def release_thumb(self) -> None:
        """Drop the in-memory thumbnail once metrics have been extracted."""
        self.thumb = None


@dataclass(frozen=True)
class Insight:
    """An engine-authored structured claim — a node in the insight graph (spec.md §8).

    The LLM narrates this; it does not author it. `direction` and `magnitude` are fixed
    by code from the data before any LLM runs, so narration cannot contradict them.
    """

    id: str
    kind: str                    # e.g. "era_style", "resonance_gap"
    claim: str                   # plain machine statement
    direction: str | None = None   # committed by code, e.g. "higher" / "lower"
    magnitude: float | None = None # committed by code
    evidence_ids: tuple[str, ...] = ()   # supporting photo ids
    resonance_angle: str | None = None  # how resonance figures in
    # Exact names (above the recognition bar) + places narration may use; nothing else.
    allowed_entities: tuple[str, ...] = ()
    era: str | None = None
    created_at: float = 0.0
    # Retrieval keywords for the insight graph: `ask` matches a question's tokens against
    # these to SELECT which nodes are relevant. Not shown to the owner; narration ignores
    # them. Empty for critique-only insights that are fetched by era rather than by topic.
    tags: tuple[str, ...] = ()
