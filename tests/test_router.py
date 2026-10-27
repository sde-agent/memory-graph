"""The 5-quadrant soft scorer (P6).

Each quadrant's canonical example (from the intent matrix) should score that quadrant
dominant. Crucially, the scorer is SOFT — a boundary focal length yields partial
membership in two quadrants, never a hard flip.
"""

from __future__ import annotations

from memory_graph.config import load_config
from memory_graph.metrics.faces import FaceAttributes
from memory_graph.metrics.geometry import Horizon
from memory_graph.metrics.light import Light
from memory_graph.metrics.optics import Optics
from memory_graph.metrics.router import score_quadrants


def quads():
    return load_config().quadrants


def _optics(focal=None, aperture=None, shutter=None, iso=None, ev=None) -> Optics:
    return Optics(focal_35mm=focal, aperture=aperture, shutter=shutter, iso=iso, ev=ev)


def _horizon(angle=0.0, y=0.5) -> Horizon:
    dutch = abs(angle) >= 4.5 if angle is not None else False
    return Horizon(angle_deg=angle, y_ratio=y, dutch=dutch)


def _light(golden=False, warm_cool=0.0) -> Light:
    band = "golden hour" if golden else "daylight"
    return Light(elevation=5.0, band=band, golden_hour=golden, warm_cool=warm_cool)


def _faces(count=0, ratio=None, candid=None) -> FaceAttributes:
    return FaceAttributes(face_count=count, face_ratio=ratio, eyes_open=None, candid_score=candid)


def test_q1_nostalgic_normal_wide_level_warm() -> None:
    v = score_quadrants(
        _optics(focal=50.0, aperture=1.8), _horizon(angle=0.0), _light(golden=True),
        _faces(), quads(), warm=True,
    )
    assert v.dominant == 1


def test_q2_visceral_ultrawide_fast_shutter() -> None:
    v = score_quadrants(
        _optics(focal=16.0, aperture=8.0, shutter=1 / 2000), _horizon(angle=0.0, y=0.2),
        _light(), _faces(), quads(),
    )
    assert v.dominant == 2


def test_q3_empathetic_tele_tight_face_candid() -> None:
    v = score_quadrants(
        _optics(focal=100.0, aperture=1.8), _horizon(angle=0.0),
        _light(), _faces(count=1, ratio=0.35, candid=0.8), quads(),
    )
    assert v.dominant == 3


def test_q4_solitary_negative_space() -> None:
    # A landscape: vast negative space, focal not the defining feature.
    v = score_quadrants(
        _optics(focal=85.0), _horizon(angle=0.0), _light(), _faces(), quads(),
        negative_space=0.85,
    )
    assert v.dominant == 4


def test_q4_solitary_long_exposure() -> None:
    v = score_quadrants(
        _optics(focal=85.0, shutter=4.0), _horizon(angle=0.0), _light(), _faces(), quads(),
    )
    assert v.dominant == 4


def test_q5_provocative_dutch_tilt() -> None:
    v = score_quadrants(
        _optics(focal=85.0), _horizon(angle=15.0), _light(), _faces(), quads(),
    )
    assert v.dominant == 5


def test_q5_provocative_shutter_drag() -> None:
    v = score_quadrants(
        _optics(focal=85.0, shutter=0.25), _horizon(angle=0.0), _light(), _faces(), quads(),
    )
    assert v.dominant == 5


def test_soft_no_cliff_at_focal_boundary() -> None:
    # A 50mm shot is clearly Q1; a 100mm clearly Q3. A 65mm sits BETWEEN — it should
    # carry partial membership in both, not snap entirely to one.
    q = quads()
    mid = score_quadrants(
        _optics(focal=65.0, aperture=1.8), _horizon(angle=0.0),
        _light(), _faces(count=1, ratio=0.22, candid=0.5), q,
    )
    assert mid.q1 > 0.0 and mid.q3 > 0.0           # both present (no cliff)


def test_scores_in_unit_range() -> None:
    v = score_quadrants(
        _optics(focal=50.0, aperture=1.8, shutter=1 / 125), _horizon(angle=2.0),
        _light(golden=True), _faces(count=2, ratio=0.3, candid=0.6), quads(),
        negative_space=0.2, warm=True,
    )
    for s in v.as_tuple():
        assert 0.0 <= s <= 1.0


def test_empty_metrics_all_low() -> None:
    v = score_quadrants(
        _optics(), Horizon(None, None, False), _light(), _faces(), quads(),
    )
    # No inputs => everything near zero; dominant is whatever, but no score is high.
    assert max(v.as_tuple()) < 0.3
