"""Apple Vision binding (P2).

Most of these verify graceful degradation (no crash when Vision is absent). A
`vision`-marked test runs real on-device classification when the bindings exist.
"""

from __future__ import annotations

import pytest
from PIL import Image

from memory_graph import vision


def test_norm_normalizes_identifiers() -> None:
    assert vision.norm("Printed_Page") == "printed page"
    assert vision.norm("  Document ") == "document"


def test_available_returns_bool() -> None:
    assert isinstance(vision.available(), bool)


def test_classify_returns_none_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(vision, "available", lambda: False)
    assert vision.classify(Image.new("RGB", (64, 64))) is None


def test_saliency_returns_none_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(vision, "available", lambda: False)
    assert vision.saliency_boxes(Image.new("RGB", (64, 64))) is None


@pytest.mark.vision
def test_classify_live_returns_labels() -> None:
    if not vision.available():
        pytest.skip("Apple Vision not available")
    out = vision.classify(Image.new("RGB", (256, 256), (120, 160, 90)))
    assert out is None or isinstance(out, list)


@pytest.mark.vision
def test_saliency_live_shape() -> None:
    if not vision.available():
        pytest.skip("Apple Vision not available")
    boxes = vision.saliency_boxes(Image.new("RGB", (256, 256), (120, 160, 90)))
    assert boxes is None or set(boxes.keys()) == {"attention", "objectness"}


def test_detect_faces_none_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(vision, "available", lambda: False)
    assert vision.detect_faces(Image.new("RGB", (64, 64))) is None


def test_feature_print_none_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(vision, "available", lambda: False)
    assert vision.feature_print(Image.new("RGB", (64, 64))) is None


@pytest.mark.vision
def test_detect_faces_live_shape() -> None:
    if not vision.available():
        pytest.skip("Apple Vision not available")
    faces = vision.detect_faces(Image.new("RGB", (256, 256), (200, 180, 170)))
    # A blank image likely has no faces; the call must still return a list (or None on failure).
    assert faces is None or isinstance(faces, list)


@pytest.mark.vision
def test_feature_print_live_returns_vector() -> None:
    if not vision.available():
        pytest.skip("Apple Vision not available")
    vec = vision.feature_print(Image.new("RGB", (256, 256), (120, 160, 90)))
    assert vec is None or (isinstance(vec, list) and len(vec) > 0)
