"""Stratified sampling (P-opt)."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_graph.config import load_config
from memory_graph.contracts import Candidate, Source
from memory_graph.sampling import plan_grid, stratified_sample


def eras():
    return load_config().eras


def _ts(year):
    return datetime(year, 6, 1, tzinfo=UTC).timestamp()


def _pool():
    pool = []
    # 2000s/hdd: 100, late 2010s/library: 100, 2020s/library: 40
    for i in range(100):
        pool.append(Candidate(id=f"h{i}", source=Source.HDD, original_ref=f"h{i}",
                              timestamp=_ts(2005)))
    for i in range(100):
        pool.append(Candidate(id=f"l{i}", source=Source.LIBRARY, original_ref=f"l{i}",
                              timestamp=_ts(2017)))
    for i in range(40):
        pool.append(Candidate(id=f"m{i}", source=Source.LIBRARY, original_ref=f"m{i}",
                              timestamp=_ts(2022)))
    return pool


def test_plan_reports_available_per_cell() -> None:
    plan = plan_grid(_pool(), eras(), target=120)
    assert plan.available[("2000s", "hdd")] == 100
    assert plan.available[("late 2010s", "library")] == 100
    assert plan.available[("2020s", "library")] == 40


def test_allocation_capped_by_availability() -> None:
    # Target larger than a thin cell: that cell is capped, others absorb the rest.
    plan = plan_grid(_pool(), eras(), target=180)
    for key, alloc in plan.allocation.items():
        assert alloc <= plan.available[key]
    assert sum(plan.allocation.values()) <= 180


def test_sample_size_and_cleanliness() -> None:
    sample = stratified_sample(_pool(), eras(), target=120, seed=1)
    assert len(sample) <= 120
    assert len({c.id for c in sample}) == len(sample)   # no duplicates in the sample


def test_sample_is_deterministic() -> None:
    a = stratified_sample(_pool(), eras(), target=100, seed=3)
    b = stratified_sample(_pool(), eras(), target=100, seed=3)
    assert [c.id for c in a] == [c.id for c in b]


def test_sample_spans_multiple_cells() -> None:
    sample = stratified_sample(_pool(), eras(), target=120, seed=2)
    cells = {(c.source.value, datetime.fromtimestamp(c.timestamp, UTC).year // 10) for c in sample}
    assert len(cells) >= 2   # draws from more than one era/source, not all from one bucket


def test_undated_excluded_from_sample() -> None:
    pool = _pool() + [Candidate(id="nd", source=Source.HDD, original_ref="nd", timestamp=None)]
    sample = stratified_sample(pool, eras(), target=300, seed=0)
    assert "nd" not in {c.id for c in sample}
