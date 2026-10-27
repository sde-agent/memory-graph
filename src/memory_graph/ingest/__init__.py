"""Ingest adapters: produce a uniform stream of Candidates from each source.

Two sources, one contract. Both adapters yield `Candidate` objects (contracts.py) with
an in-memory thumbnail and normalized EXIF. Guardrails that live here:
  - HDD: the source must be read-only; each file is opened exactly once.
  - Library: master paths are NEVER accessed (that would trigger an iCloud download);
    only the Photos database and local derivative caches are read.
"""

from memory_graph.ingest.hdd import stream_hdd
from memory_graph.ingest.library import LibraryMasterAccessError, stream_library

__all__ = ["stream_hdd", "stream_library", "LibraryMasterAccessError"]
