"""Core contract behavior (P0)."""

from __future__ import annotations

from memory_graph.contracts import Candidate, Insight, Source


def test_candidate_release_thumb_drops_reference() -> None:
    c = Candidate(id="hdd:x", source=Source.HDD, original_ref="/v/x.jpg", thumb=object())
    assert c.thumb is not None
    c.release_thumb()
    assert c.thumb is None


def test_source_is_string_enum() -> None:
    assert Source.LIBRARY == "library"
    assert Source.HDD == "hdd"


def test_insight_is_frozen_and_hashable() -> None:
    ins = Insight(id="i1", kind="era_style", claim="leaned on 50mm wide open")
    # Frozen: committed facts cannot be mutated after creation.
    try:
        ins.claim = "something else"  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised
    # Hashable (frozen dataclass) so insights can live in sets / dict keys.
    assert ins in {ins}


def test_candidate_defaults() -> None:
    c = Candidate(id="lib:u", source=Source.LIBRARY, original_ref="u")
    assert c.timestamp is None
    assert c.exif == {}
    assert c.library_meta is None
