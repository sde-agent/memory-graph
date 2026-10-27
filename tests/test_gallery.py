"""Identity gallery: matching with a precision bar + pHash bridging (P5)."""

from __future__ import annotations

from memory_graph.identity.gallery import Gallery, Match, bridge_by_phash, cosine


def _vec(*xs) -> list[float]:
    return list(xs)


def test_cosine_basic() -> None:
    assert abs(cosine([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-6
    assert abs(cosine([1, 0, 0], [0, 1, 0])) < 1e-6
    assert cosine([0, 0, 0], [1, 1, 1]) == 0.0  # degenerate -> 0


def test_match_names_above_bar() -> None:
    g = Gallery(min_similarity=0.55)
    g.add("Ada", _vec(1, 0, 0, 0))
    g.add("Grace", _vec(0, 1, 0, 0))
    # A query very close to Ada's vector matches Ada.
    m = g.match(_vec(0.98, 0.02, 0, 0))
    assert isinstance(m, Match)
    assert m.name == "Ada"
    assert m.similarity > 0.9
    assert m.present


def test_match_below_bar_is_present_but_unnamed() -> None:
    g = Gallery(min_similarity=0.9)   # a demanding bar
    g.add("Ada", _vec(1, 0, 0, 0))
    # A query only loosely similar: present, but NOT named (wrong name is worse).
    m = g.match(_vec(0.5, 0.5, 0.5, 0.5))
    assert m.name is None
    assert m.present
    assert m.similarity < 0.9


def test_match_empty_gallery() -> None:
    g = Gallery(min_similarity=0.5)
    m = g.match(_vec(1, 0, 0))
    assert m.name is None
    assert not m.present


def test_match_none_embedding() -> None:
    g = Gallery(min_similarity=0.5)
    g.add("Ada", _vec(1, 0, 0))
    m = g.match(None)
    assert m.name is None
    assert m.present   # the gallery exists; we just had nothing to compare


def test_multiple_references_per_person() -> None:
    g = Gallery(min_similarity=0.55)
    g.add("Ada", _vec(1, 0, 0, 0))
    g.add("Ada", _vec(0, 0, 1, 0))   # a second exemplar of Ada
    assert "Ada" in g.names
    assert g.size == 2
    # A query near the SECOND exemplar still matches Ada.
    m = g.match(_vec(0.1, 0, 0.99, 0))
    assert m.name == "Ada"


def test_add_degenerate_vector_ignored() -> None:
    g = Gallery(min_similarity=0.5)
    g.add("Ada", _vec(0, 0, 0))  # zero vector: cannot normalize, ignored
    assert g.size == 0


def test_bridge_by_phash_propagates_library_names() -> None:
    index = {"ff00ff00": ["Ada", "Grace"], "0000ffff": ["Bob"]}
    assert bridge_by_phash("ff00ff00", index) == ["Ada", "Grace"]
    assert bridge_by_phash("deadbeef", index) == []


def test_normalization_makes_scale_invariant() -> None:
    g = Gallery(min_similarity=0.55)
    g.add("Ada", _vec(2, 0, 0))          # unnormalized reference
    m = g.match(_vec(10, 0, 0))          # unnormalized query, same direction
    assert m.name == "Ada"
    assert abs(m.similarity - 1.0) < 1e-6
