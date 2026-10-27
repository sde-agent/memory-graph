"""Optics extraction from EXIF (P4) — exact values against known inputs."""

from __future__ import annotations

import math

from memory_graph.config import load_config
from memory_graph.metrics.optics import extract_optics


def cfg():
    return load_config()


def test_focal35_used_directly_when_present() -> None:
    o = extract_optics({"focal35": 75.0, "focal": 50.0}, cfg())
    assert o.focal_35mm == 75.0


def test_focal35_from_crop_factor_when_missing(tmp_path) -> None:
    # Local overlay adds a crop factor for a specific camera model.
    import textwrap

    from memory_graph.paths import DEFAULT_CONFIG

    local = tmp_path / "config.local.toml"
    local.write_text(
        textwrap.dedent(
            """
            [framing.crop]
            "NIKON D7000" = 1.5
            """
        )
    )
    c = load_config(default_path=DEFAULT_CONFIG, local_path=local)
    o = extract_optics({"focal": 35.0, "model": "NIKON D7000"}, c)
    assert o.focal_35mm == 52.5  # 35 * 1.5


def test_raw_focal_fallback() -> None:
    o = extract_optics({"focal": 28.0}, cfg())
    assert o.focal_35mm == 28.0


def test_aperture_shutter_iso_passthrough() -> None:
    o = extract_optics({"aperture": 1.8, "shutter": 0.004, "iso": 200}, cfg())
    assert o.aperture == 1.8
    assert o.shutter == 0.004
    assert o.iso == 200


def test_ev_computation_at_iso100() -> None:
    # f/8, 1/250s, ISO 100: EV = log2(64 / (1/250)) = log2(16000) ~= 13.97
    o = extract_optics({"aperture": 8.0, "shutter": 1 / 250, "iso": 100}, cfg())
    assert o.ev is not None
    assert math.isclose(o.ev, math.log2(64 * 250), rel_tol=1e-6)


def test_ev_normalized_for_iso() -> None:
    # Doubling ISO lowers normalized EV by 1 stop.
    base = extract_optics({"aperture": 4.0, "shutter": 1 / 60, "iso": 100}, cfg()).ev
    hi = extract_optics({"aperture": 4.0, "shutter": 1 / 60, "iso": 200}, cfg()).ev
    assert math.isclose(base - hi, 1.0, rel_tol=1e-6)


def test_ev_none_without_exposure() -> None:
    assert extract_optics({"aperture": 2.8}, cfg()).ev is None
    assert extract_optics({"shutter": 0.01}, cfg()).ev is None


def test_empty_exif_yields_all_none() -> None:
    o = extract_optics({}, cfg())
    assert o.focal_35mm is None and o.aperture is None and o.ev is None
