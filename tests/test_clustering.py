"""Burst clustering + canonical keyframe election (P3)."""

from __future__ import annotations

from PIL import Image, ImageFilter

from memory_graph.cluster.bursts import (
    cluster_and_elect,
    elect_canonical,
    group_bursts,
)
from memory_graph.config import load_config
from memory_graph.contracts import Candidate, Source
from memory_graph.ingest.imaging import phash


def cfg():
    return load_config()


def _crisp() -> Image.Image:
    import numpy as np

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    return Image.fromarray(noise, mode="L").convert("RGB")


def _blurry() -> Image.Image:
    return _crisp().filter(ImageFilter.GaussianBlur(radius=5))


def _cand(cid, ts, thumb, *, source=Source.HDD, favorite=False, focal=50.0,
          w=4000, h=3000, burst_group=None, phash_override=None) -> Candidate:
    meta = None
    if source == Source.LIBRARY or favorite or burst_group:
        meta = {"favorite": favorite, "burst_group": burst_group, "faces": []}
    exif = {
        "focal35": focal,
        "width": w,
        "height": h,
        "phash": phash_override or phash(thumb),
    }
    return Candidate(id=cid, source=source, original_ref=cid, timestamp=ts,
                     thumb=thumb, exif=exif, library_meta=meta)


def test_near_identical_frames_within_window_form_one_burst() -> None:
    t = _crisp()
    # Same image content (same pHash), 2s apart => one burst of 3.
    cands = [_cand(f"hdd:{i}", 1000.0 + i * 2, t.copy()) for i in range(3)]
    shots = cluster_and_elect(cands, cfg())
    assert len(shots) == 1
    assert shots[0].size == 3


def test_frames_far_apart_in_time_are_separate() -> None:
    t = _crisp()
    cands = [
        _cand("hdd:a", 1000.0, t.copy()),
        _cand("hdd:b", 1000.0 + 999, t.copy()),  # way beyond the window
    ]
    shots = cluster_and_elect(cands, cfg())
    assert len(shots) == 2


def test_different_content_not_merged_even_if_adjacent() -> None:
    a = _cand("hdd:a", 1000.0, _crisp())
    b = _cand("hdd:b", 1001.0, Image.new("RGB", (256, 256), (0, 0, 0)))  # very different pHash
    shots = cluster_and_elect([a, b], cfg())
    assert len(shots) == 2


def test_different_framing_bucket_not_merged() -> None:
    t = _crisp()
    wide = _cand("hdd:a", 1000.0, t.copy(), focal=16.0)   # ultra-wide bucket
    tele = _cand("hdd:b", 1001.0, t.copy(), focal=100.0)  # tele bucket
    shots = cluster_and_elect([wide, tele], cfg())
    assert len(shots) == 2


def test_apple_burst_group_honored() -> None:
    t = _crisp()
    # Two library photos sharing a burst_group are grouped even without pHash checks.
    a = _cand("lib:a", 1000.0, t.copy(), source=Source.LIBRARY, burst_group="B1")
    b = _cand("lib:b", 1000.0 + 500, t.copy(), source=Source.LIBRARY, burst_group="B1")
    shots = cluster_and_elect([a, b], cfg())
    assert len(shots) == 1
    assert shots[0].size == 2


def test_canonical_is_sharpest_frame() -> None:
    sharp = _cand("hdd:sharp", 1000.0, _crisp())
    blur = _cand("hdd:blur", 1002.0, _blurry(), phash_override=sharp.exif["phash"])
    # Force same pHash/shot-key so they cluster; the sharp one must win.
    rep = elect_canonical([blur, sharp], cfg())
    assert rep.id == "hdd:sharp"


def test_favorite_breaks_near_tie_but_not_sharpness() -> None:
    c = cfg()
    t = _crisp()
    # Two equally sharp frames; the favorited one should win the tie.
    plain = _cand("hdd:plain", 1000.0, t.copy())
    fav = _cand("hdd:fav", 1001.0, t.copy(), favorite=True)
    rep = elect_canonical([plain, fav], c)
    assert rep.id == "hdd:fav"

    # But a favorited BLURRY frame must NOT beat a clearly sharper plain frame.
    sharp = _cand("hdd:sharp", 1000.0, _crisp())
    favblur = _cand("hdd:favblur", 1001.0, _blurry(), favorite=True)
    rep2 = elect_canonical([sharp, favblur], c)
    assert rep2.id == "hdd:sharp"


def test_burst_id_is_stable_and_shared() -> None:
    t = _crisp()
    cands = [_cand(f"hdd:{i}", 1000.0 + i, t.copy()) for i in range(3)]
    shots = cluster_and_elect(cands, cfg())
    assert shots[0].burst_id == "burst:hdd:0"  # smallest member id


def test_group_bursts_returns_all_members() -> None:
    t = _crisp()
    cands = [_cand(f"hdd:{i}", 1000.0 + i, t.copy()) for i in range(5)]
    groups = group_bursts(cands, cfg())
    assert sum(len(g) for g in groups) == 5
