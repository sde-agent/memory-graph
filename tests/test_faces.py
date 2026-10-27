"""Face attributes + candidness proxy + face cropping (P5).

Face detections are synthetic dicts so these tests are deterministic and need no Vision.
"""

from __future__ import annotations

from PIL import Image

from memory_graph.metrics.faces import (
    crop_face,
    face_attributes,
    from_library_meta,
    from_vision,
    named_people,
)


def _face(x, y, w, h, yaw=None, eye=None) -> dict:
    d = {"x": x, "y": y, "w": w, "h": h, "area": w * h}
    if yaw is not None:
        d["yaw"] = yaw
    if eye is not None:
        d["eye_openness"] = eye
    return d


def test_no_faces_is_neutral() -> None:
    a = from_vision([])
    assert a.face_count == 0
    assert a.face_ratio is None
    assert a.candid_score is None


def test_face_count_and_ratio() -> None:
    faces = [_face(0.1, 0.1, 0.2, 0.3), _face(0.5, 0.5, 0.1, 0.1)]
    a = from_vision(faces)
    assert a.face_count == 2
    # Largest face area = 0.2*0.3 = 0.06
    assert abs(a.face_ratio - 0.06) < 1e-6


def test_eyes_open_from_landmarks() -> None:
    open_face = from_vision([_face(0.2, 0.2, 0.3, 0.3, eye=0.3)])
    shut_face = from_vision([_face(0.2, 0.2, 0.3, 0.3, eye=0.05)])
    assert open_face.eyes_open is True
    assert shut_face.eyes_open is False


def test_single_frontal_face_reads_posed() -> None:
    a = from_vision([_face(0.3, 0.2, 0.4, 0.5, yaw=0.0)])
    assert a.candid_score is not None
    assert a.candid_score <= 0.3   # a lone frontal face is the canonical posed portrait


def test_turned_head_reads_candid() -> None:
    a = from_vision([_face(0.3, 0.2, 0.4, 0.5, yaw=0.8)])  # head clearly turned
    assert a.candid_score is not None
    assert a.candid_score > 0.5


def test_unknown_yaw_is_conservative_not_candid() -> None:
    # No gaze info: we must NOT over-claim candid (asymmetric; posed default is safer).
    a = from_vision([_face(0.3, 0.2, 0.4, 0.5)])
    assert a.candid_score is not None
    assert a.candid_score <= 0.3


def test_library_meta_faces() -> None:
    meta = {"faces": [{"size": 0.3}, {"size": 0.1}]}
    a = from_library_meta(meta)
    assert a.face_count == 2
    assert abs(a.face_ratio - 0.09) < 1e-6  # largest size^2 = 0.3^2
    assert a.eyes_open is None               # Library has no reliable eye data


def test_library_candidness_posed_vs_candid() -> None:
    # A single frontal smiling face => posed => low candidness.
    posed = from_library_meta({"faces": [{"size": 0.4, "yaw": 0.0, "has_smile": True}]})
    assert posed.candid_score is not None and posed.candid_score <= 0.3
    # A turned, non-smiling face => candid => high.
    candid = from_library_meta({"faces": [{"size": 0.4, "yaw": 0.8, "has_smile": False}]})
    assert candid.candid_score is not None and candid.candid_score > 0.5


def test_named_people_reads_authoritative_names() -> None:
    meta = {
        "faces": [
            {"size": 0.4, "name": "Ada Lovelace"},
            {"size": 0.1, "name": "Grace Hopper"},
            {"size": 0.1, "name": None},          # unnamed face ignored
            {"size": 0.1, "name": "Ada Lovelace"},  # duplicate collapsed
        ]
    }
    assert named_people(meta) == ["Ada Lovelace", "Grace Hopper"]


def test_named_people_empty() -> None:
    assert named_people({"faces": []}) == []
    assert named_people(None) == []


def test_face_attributes_prefers_vision() -> None:
    a = face_attributes(vision_faces=[_face(0.1, 0.1, 0.2, 0.2)],
                        library_meta={"faces": [{"size": 0.9}]})
    assert a.face_count == 1
    assert abs(a.face_ratio - 0.04) < 1e-6   # used the Vision face, not the library one


def test_face_attributes_falls_back_to_library() -> None:
    a = face_attributes(vision_faces=None, library_meta={"faces": [{"size": 0.5}]})
    assert a.face_count == 1


def test_crop_face_returns_subregion() -> None:
    img = Image.new("RGB", (400, 400), (100, 100, 100))
    crop = crop_face(img, _face(0.25, 0.25, 0.5, 0.5), pad=0.0)
    assert crop.size == (200, 200)


def test_crop_face_degenerate_returns_whole() -> None:
    img = Image.new("RGB", (400, 400))
    crop = crop_face(img, _face(0.0, 0.0, 0.0, 0.0))
    assert crop.size == (400, 400)
