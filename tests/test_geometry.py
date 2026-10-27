"""Geometry metrics: horizon tilt and negative space (P4)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from memory_graph.metrics.geometry import horizon, negative_space


def _horizontal_edge_image(tilt_deg: float = 0.0, size: int = 200) -> Image.Image:
    """An image split into a dark top and light bottom by a line at `tilt_deg`.

    The boundary is the dominant edge; its angle should be recovered as ~tilt_deg.
    """
    arr = np.zeros((size, size), dtype=np.uint8)
    slope = np.tan(np.radians(tilt_deg))
    for x in range(size):
        boundary = int(size / 2 + slope * (x - size / 2))
        arr[max(0, boundary):, x] = 255
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_level_horizon_near_zero_angle() -> None:
    h = horizon(_horizontal_edge_image(0.0))
    assert h.angle_deg is not None
    assert abs(h.angle_deg) < 2.0
    assert not h.dutch


def test_tilted_horizon_detected_as_dutch() -> None:
    h = horizon(_horizontal_edge_image(10.0))
    assert h.angle_deg is not None
    assert abs(h.angle_deg) > 4.5          # clearly tilted
    assert h.dutch                         # within the Dutch range


def test_horizon_y_ratio_is_mid_for_centered_split() -> None:
    h = horizon(_horizontal_edge_image(0.0))
    assert h.y_ratio is not None
    assert 0.4 < h.y_ratio < 0.6


def test_flat_image_has_no_horizon() -> None:
    flat = Image.new("RGB", (100, 100), (128, 128, 128))
    h = horizon(flat)
    assert h.angle_deg is None
    assert not h.dutch


def test_clean_line_is_high_confidence() -> None:
    h = horizon(_horizontal_edge_image(10.0))
    assert h.confidence > 0.7          # a real, coherent line
    assert h.dutch


def test_scattered_edges_low_confidence_suppresses_dutch() -> None:
    # Random noise has strong but SCATTERED vertical gradients: a line fit is meaningless,
    # so confidence must be low and the Dutch flag must NOT fire (the P6 over-firing fix).
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, size=(200, 200), dtype=np.uint8)
    h = horizon(Image.fromarray(noise, mode="L").convert("RGB"))
    assert h.confidence < 0.5
    assert not h.dutch


def test_negative_space_none_without_boxes() -> None:
    assert negative_space(None) is None


def test_negative_space_empty_boxes_is_all_space() -> None:
    # Vision ran but found nothing salient => entirely negative space.
    assert negative_space({"attention": [], "objectness": []}) == 1.0


def test_negative_space_quarter_box() -> None:
    boxes = {"attention": [{"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5, "confidence": 0.9}]}
    ns = negative_space(boxes)
    assert ns is not None
    assert abs(ns - 0.75) < 0.02  # a quarter covered => 75% negative space


def test_negative_space_overlapping_boxes_not_double_counted() -> None:
    boxes = {
        "attention": [
            {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5, "confidence": 0.9},
            {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "confidence": 0.8},
        ]
    }
    ns = negative_space(boxes)
    # Union of the two overlapping quarter-boxes is less than 2*0.25 = 0.5 coverage.
    assert ns is not None
    assert ns > 0.5


def test_negative_space_prefers_attention_over_objectness() -> None:
    boxes = {
        "attention": [{"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.1, "confidence": 0.9}],
        "objectness": [{"x": 0.0, "y": 0.0, "w": 0.9, "h": 0.9, "confidence": 0.9}],
    }
    ns = negative_space(boxes)
    assert ns is not None and ns > 0.9  # used the small attention box, not objectness
