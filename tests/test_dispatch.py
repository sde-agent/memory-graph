"""Daily dispatch selection + guardrails (P7)."""

from __future__ import annotations

from memory_graph.config import load_config
from memory_graph.scoring.dispatch import select_dispatch


def dcfg():
    return load_config().dispatch


def _row(rid, resonance, *, q=1, favorite=False, private=False, source="hdd",
         canonical=1, last_dispatched=None) -> dict:
    return {
        "id": rid,
        "resonance": resonance,
        "dominant_quadrant": q,
        "favorite": 1 if favorite else 0,
        "private": 1 if private else 0,
        "source": source,
        "is_canonical": canonical,
        "last_dispatched": last_dispatched,
    }


def test_picks_highest_resonance() -> None:
    rows = [_row("a", 0.1, q=1), _row("b", 0.9, q=2), _row("c", 0.5, q=3)]
    out = select_dispatch(rows, dcfg(), now=1_000_000.0, n=1)
    assert [i.id for i in out] == ["b"]


def test_respects_count_range() -> None:
    rows = [_row(f"p{i}", 1.0 - i * 0.1, q=(i % 5) + 1) for i in range(10)]
    out = select_dispatch(rows, dcfg(), now=1_000_000.0, n=3)
    assert len(out) == 3


def test_private_never_dispatched() -> None:
    rows = [_row("secret", 5.0, private=True, q=1), _row("ok", 0.2, q=2)]
    out = select_dispatch(rows, dcfg(), now=1_000_000.0, n=2)
    assert "secret" not in [i.id for i in out]
    assert "ok" in [i.id for i in out]


def test_recently_dispatched_excluded() -> None:
    now = 1_000_000.0
    recent = now - 5 * 86400           # 5 days ago, inside the 60-day window
    rows = [_row("fresh", 0.3, q=1), _row("stale", 0.9, q=2, last_dispatched=recent)]
    out = select_dispatch(rows, dcfg(), now=now, n=1)
    assert [i.id for i in out] == ["fresh"]   # the recently-shown higher one is skipped


def test_favorite_share_capped() -> None:
    # Many favorites; the dispatch must not be all favorites (DR-3).
    rows = [_row(f"f{i}", 1.0 - i * 0.01, q=1, favorite=True) for i in range(5)]
    rows += [_row(f"n{i}", 0.5 - i * 0.01, q=2, favorite=False) for i in range(5)]
    out = select_dispatch(rows, dcfg(), now=1_000_000.0, n=3)
    fav_count = sum(1 for i in out if i.favorite)
    # max_favorite_share default 0.5 of 3 -> at most 1 favorite.
    assert fav_count <= 1


def test_diversity_prefers_distinct_quadrants() -> None:
    # Top resonances are all the same quadrant; a slightly lower other-quadrant photo
    # should be preferred for variety when the cap allows.
    rows = [
        _row("q1a", 0.95, q=1),
        _row("q1b", 0.94, q=1),
        _row("q3", 0.80, q=3),
    ]
    out = select_dispatch(rows, dcfg(), now=1_000_000.0, n=2)
    quads = {i.dominant_quadrant for i in out}
    assert len(quads) == 2   # two different quadrants, not two Q1s


def test_deterministic_for_same_inputs() -> None:
    rows = [_row(f"p{i}", 0.5, q=(i % 5) + 1) for i in range(8)]   # tie on resonance
    a = select_dispatch(rows, dcfg(), now=1_000_000.0, n=3)
    b = select_dispatch(rows, dcfg(), now=1_000_000.0, n=3)
    assert [i.id for i in a] == [i.id for i in b]   # seeded, reproducible


def test_empty_when_nothing_eligible() -> None:
    rows = [_row("p", 0.5, private=True)]
    assert select_dispatch(rows, dcfg(), now=1_000_000.0) == []
