"""Content cues in the router (Part B): faces/scenery drive Q1/Q3/Q2/Q4."""

from __future__ import annotations

from memory_graph.config import load_config
from memory_graph.metrics.faces import FaceAttributes
from memory_graph.metrics.geometry import Horizon
from memory_graph.metrics.light import Light
from memory_graph.metrics.optics import Optics
from memory_graph.metrics.router import score_quadrants


def quads():
    return load_config().quadrants


def _o(focal=None, aperture=None, shutter=None):
    return Optics(focal_35mm=focal, aperture=aperture, shutter=shutter, iso=None, ev=None)


def _h():
    return Horizon(angle_deg=0.0, y_ratio=0.5, dutch=False, confidence=0.8)


def _light(golden=False):
    return Light(elevation=5.0, band="golden hour" if golden else "daylight",
                 golden_hour=golden, warm_cool=0.1)


def _faces(count=0, ratio=None, candid=None):
    return FaceAttributes(face_count=count, face_ratio=ratio, eyes_open=None,
                          candid_score=candid)


def test_people_scene_lifts_q1_without_normal_focal() -> None:
    # A warm photo of people at an unknown/wide focal should still register nostalgia,
    # which the pure focal-core router missed.
    v = score_quadrants(
        _o(focal=28.0), _h(), _light(golden=True), _faces(count=2), quads(),
        warm=True, scene_labels=[("people", 0.9), ("child", 0.8)],
    )
    assert v.q1 > 0.4


def test_face_presence_anchors_q3_without_tight_crop() -> None:
    # A candid of a person at normal focal (not a tele tight-crop) still reads tender.
    v = score_quadrants(
        _o(focal=50.0), _h(), _light(), _faces(count=1, ratio=0.1, candid=0.8), quads(),
        scene_labels=[("people", 0.9), ("adult", 0.85)],
    )
    assert v.q3 > 0.4


def test_scenery_lifts_q2_awe() -> None:
    v = score_quadrants(
        _o(focal=24.0), _h(), _light(), _faces(), quads(),
        scene_labels=[("mountain", 0.9), ("sky", 0.85), ("outdoor", 0.9)],
    )
    assert v.q2 > 0.3


def test_lonely_scenery_lifts_q4() -> None:
    # Scenery with no people reinforces solitude.
    v = score_quadrants(
        _o(focal=50.0), _h(), _light(), _faces(count=0), quads(),
        negative_space=0.5, scene_labels=[("landscape", 0.9), ("sky", 0.8)],
    )
    assert v.q4 > 0.3


def test_crowd_suppresses_solitude() -> None:
    # Same scenery but with a crowd should NOT read as solitary.
    lonely = score_quadrants(
        _o(focal=50.0), _h(), _light(), _faces(count=0), quads(),
        scene_labels=[("landscape", 0.9)],
    )
    crowd = score_quadrants(
        _o(focal=50.0), _h(), _light(), _faces(count=5), quads(),
        scene_labels=[("landscape", 0.9), ("crowd", 0.8)],
    )
    assert crowd.q4 <= lonely.q4


def test_no_scene_labels_still_works() -> None:
    # Backward compatible: no content cues => the EXIF path still scores (canonical Q1).
    v = score_quadrants(
        _o(focal=50.0, aperture=1.8), _h(), _light(golden=True), _faces(), quads(),
        warm=True, scene_labels=None,
    )
    assert v.dominant == 1
