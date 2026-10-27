"""Global near-duplicate dedup via LSH banding (clutter leakage fix)."""

from __future__ import annotations

from memory_graph.cluster.bursts import Shot, global_dedup
from memory_graph.config import load_config
from memory_graph.contracts import Candidate, Source


def cfg():
    return load_config()


def _shot(pid, phash, ts=1000.0) -> Shot:
    c = Candidate(id=pid, source=Source.HDD, original_ref=pid, timestamp=ts,
                  exif={"phash": phash})
    return Shot(burst_id=f"b:{pid}", representative=c, members=[c])


def test_identical_hashes_merge() -> None:
    shots = [_shot("a", "ffff0000"), _shot("b", "ffff0000"), _shot("c", "00000000")]
    out = global_dedup(shots, cfg())
    assert len(out) == 2   # a+b merged, c separate


def test_near_duplicate_within_threshold_merges() -> None:
    # Two hashes differing in a few bits (<=6) should merge.
    a = "ffffffffffffffff"
    b = "fffffffffffffff0"   # differs in the low nibble (4 bits)
    out = global_dedup([_shot("a", a), _shot("b", b)], cfg())
    assert len(out) == 1


def test_distant_hashes_do_not_merge() -> None:
    a = "ffffffffffffffff"
    b = "0000000000000000"   # 64 bits apart
    out = global_dedup([_shot("a", a), _shot("b", b)], cfg())
    assert len(out) == 2


def test_dupe_across_time_and_source_merges() -> None:
    # The key fix: near-dups NOT time-adjacent (would survive burst clustering) still merge.
    a = Candidate(id="hdd:x", source=Source.HDD, original_ref="x", timestamp=1000.0,
                  exif={"phash": "aaaabbbbccccdddd"})
    b = Candidate(id="lib:y", source=Source.LIBRARY, original_ref="y", timestamp=9_000_000.0,
                  exif={"phash": "aaaabbbbccccdddd"})
    out = global_dedup([Shot("b:x", a, [a]), Shot("b:y", b, [b])], cfg())
    assert len(out) == 1
    assert out[0].size == 2   # members folded together, nothing lost


def test_no_hash_shots_passthrough() -> None:
    c = Candidate(id="n", source=Source.HDD, original_ref="n", exif={})
    out = global_dedup([Shot("b:n", c, [c])], cfg())
    assert len(out) == 1


def test_transitive_grouping() -> None:
    # a~b and b~c (each within threshold) => all three in one group (union-find).
    shots = [
        _shot("a", "ffffffffffffffff"),
        _shot("b", "fffffffffffffff0"),   # ~a
        _shot("c", "ffffffffffffff00"),   # ~b
    ]
    out = global_dedup(shots, cfg())
    assert len(out) == 1
    assert out[0].size == 3
