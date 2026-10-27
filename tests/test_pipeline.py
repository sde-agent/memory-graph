"""End-to-end pipeline: analyze -> persist -> dispatch (P7).

Uses synthetic candidates with real thumbnails and `use_vision=False` so the test is
deterministic and does not depend on Apple Vision.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from memory_graph.config import load_config
from memory_graph.contracts import Candidate, Source
from memory_graph.db import connect, dispatch_candidates, init_db, upsert_photo
from memory_graph.pipeline import analyze_candidate
from memory_graph.scoring.dispatch import select_dispatch


def cfg():
    return load_config()


def _thumb() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(
        rng.integers(0, 256, (256, 256), dtype=np.uint8), "L"
    ).convert("RGB")


def _cand(cid="hdd:1", *, focal=50.0, favorite=False, ts=1_200_000_000.0) -> Candidate:
    meta = {"favorite": favorite, "faces": [], "albums": []} if favorite else None
    return Candidate(
        id=cid, source=Source.HDD, original_ref=f"/v/{cid}.jpg", timestamp=ts,
        thumb=_thumb(),
        exif={"focal35": focal, "aperture": 1.8, "shutter": 1 / 125, "iso": 200,
              "width": 4000, "height": 3000, "phash": "ff00ff00"},
        library_meta=meta,
    )


def test_analyze_produces_complete_row() -> None:
    row = analyze_candidate(_cand(), cfg(), now=1_700_000_000.0, use_vision=False)
    # Required/expected keys present.
    for key in ("id", "source", "phash", "sharpness", "is_canonical",
                "q1_score", "q3_score", "dominant_quadrant", "resonance", "private"):
        assert key in row
    assert row["focal_length_35mm"] == 50.0
    assert row["aperture"] == 1.8
    assert 1 <= row["dominant_quadrant"] <= 5
    assert row["resonance"] >= 0.0


def test_analyze_persists_and_reads_back(tmp_path) -> None:
    conn = connect(tmp_path / "p.db")
    init_db(conn)
    row = analyze_candidate(_cand(), cfg(), use_vision=False)
    upsert_photo(conn, row)
    rows = dispatch_candidates(conn)
    assert len(rows) == 1
    assert rows[0]["id"] == "hdd:1"
    conn.close()


def test_full_chain_analyze_then_dispatch(tmp_path) -> None:
    conn = connect(tmp_path / "p.db")
    init_db(conn)
    # A normal-lens recent favorite (should rank well) and a plain older wide shot.
    for c in (
        _cand("hdd:fav", focal=50.0, favorite=True),
        _cand("hdd:plain", focal=16.0),
    ):
        upsert_photo(conn, analyze_candidate(c, cfg(), use_vision=False))
    rows = dispatch_candidates(conn)
    out = select_dispatch(rows, cfg().dispatch, now=1_700_000_000.0, n=1)
    conn.close()
    assert len(out) == 1  # a dispatch was produced from persisted analysis


def _cand_raw_focal(cid="hdd:k", *, raw_focal=6.3, model="KODAK TEST",
                    ts=1_000_000_000.0) -> Candidate:
    # A camera that reports only the RAW focal (no focal35) — the crop-table case.
    return Candidate(
        id=cid, source=Source.HDD, original_ref=f"/v/{cid}.jpg", timestamp=ts,
        thumb=_thumb(),
        exif={"focal": raw_focal, "model": model, "aperture": 4.0,
              "shutter": 1 / 250, "iso": 100, "width": 2000, "height": 1500,
              "phash": "abcd"},
        library_meta=None,
    )


def test_rescore_matches_full_analysis_when_nothing_changes() -> None:
    # B2 faithfulness: re-scoring a persisted row (no pixels) reproduces the quadrant
    # vector + resonance that the full pixel analysis produced.
    from memory_graph.pipeline import rescorable, rescore_row

    c = cfg()
    row = analyze_candidate(_cand("hdd:r", focal=50.0), c, now=1_700_000_000.0,
                            use_vision=False)
    assert rescorable(row)
    re = rescore_row(row, c, now=1_700_000_000.0)
    for q in ("q1_score", "q2_score", "q3_score", "q4_score", "q5_score"):
        assert abs(re[q] - row[q]) < 1e-9
    assert re["dominant_quadrant"] == row["dominant_quadrant"]
    assert abs(re["resonance"] - row["resonance"]) < 1e-9


def test_rescore_applies_crop_factor_and_shifts_focal() -> None:
    # Analyze with NO crop entry (raw 6.3mm stored), then add a crop factor and re-score:
    # the focal must jump to the corrected equivalent without touching pixels.
    import dataclasses

    from memory_graph.pipeline import rescore_row

    base = load_config()
    row = analyze_candidate(_cand_raw_focal(model="KODAK TEST", raw_focal=6.3),
                            base, use_vision=False)
    assert row["focal_raw"] == 6.3
    assert abs(row["focal_length_35mm"] - 6.3) < 1e-9   # no crop yet -> raw stored

    # Config now knows the camera's 4.762x crop factor.
    framing2 = dataclasses.replace(base.framing, crop={"KODAK TEST": 4.762})
    cfg2 = dataclasses.replace(base, framing=framing2)
    re = rescore_row(row, cfg2)
    assert abs(re["focal_length_35mm"] - 6.3 * 4.762) < 1e-6   # ~30mm equiv


def test_rescore_prefers_exif_focal35_over_crop() -> None:
    # A camera that DID supply its own 35mm-equiv must never be crop-scaled.
    import dataclasses

    from memory_graph.pipeline import rescore_row

    base = load_config()
    row = analyze_candidate(_cand("hdd:ff", focal=85.0), base, use_vision=False)
    assert row["exif_focal35"] == 85.0
    framing2 = dataclasses.replace(base.framing, crop={"whatever": 2.0})
    cfg2 = dataclasses.replace(base, framing=framing2)
    re = rescore_row(row, cfg2)
    assert abs(re["focal_length_35mm"] - 85.0) < 1e-9   # untouched


def test_pre_b2_row_is_not_rescorable() -> None:
    # A row lacking the persisted re-score signals must be flagged for re-analysis.
    from memory_graph.pipeline import rescorable
    legacy = {"id": "hdd:old", "focal_length_35mm": 6.3, "camera_model": "X"}
    assert not rescorable(legacy)


def test_private_flag_from_hidden_album() -> None:
    c = Candidate(
        id="lib:h", source=Source.LIBRARY, original_ref="uuid-h", timestamp=1_200_000_000.0,
        thumb=_thumb(),
        exif={"focal35": 50.0, "phash": "aa"},
        library_meta={"favorite": False, "faces": [], "albums": ["Hidden"]},
    )
    row = analyze_candidate(c, cfg(), use_vision=False)
    assert row["private"] == 1   # a Hidden-album photo is marked private
