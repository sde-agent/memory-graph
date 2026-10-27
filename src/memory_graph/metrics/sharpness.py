"""Image sharpness via the variance of the Laplacian (pure, numpy-based).

A blurry frame has little high-frequency detail, so the Laplacian (a second-derivative
edge operator) has low variance; a crisp frame has high variance. This is the classic,
cheap focus measure used to pick the sharpest frame of a burst (spec.md §3). No OpenCV
dependency — a 3x3 Laplacian kernel applied with numpy is enough and keeps the engine
portable.

`region_variance` lets the caller weight sharpness inside a sub-rectangle (e.g. the
largest face box) so burst election favors the frame whose SUBJECT is sharp, not one
that happens to have a crisp background. Face boxes arrive in P5; until then the whole
frame is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 3x3 discrete Laplacian (matches cv2.Laplacian's default kernel).
_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _laplacian(gray: np.ndarray) -> np.ndarray:
    """Convolve a 2-D grayscale array with the Laplacian kernel (valid region)."""
    # Manual 3x3 convolution via shifted slices — avoids a scipy dependency.
    p = np.pad(gray, 1, mode="reflect")
    out = (
        _LAPLACIAN[0, 1] * p[:-2, 1:-1]
        + _LAPLACIAN[1, 0] * p[1:-1, :-2]
        + _LAPLACIAN[1, 1] * p[1:-1, 1:-1]
        + _LAPLACIAN[1, 2] * p[1:-1, 2:]
        + _LAPLACIAN[2, 1] * p[2:, 1:-1]
    )
    return out


def _gray(thumb: Any) -> np.ndarray:
    return np.asarray(thumb.convert("L"), dtype=np.float32)


def sharpness(thumb: Any) -> float:
    """Variance of the Laplacian over the whole thumbnail. Higher = sharper."""
    return float(_laplacian(_gray(thumb)).var())


def region_variance(thumb: Any, box: tuple[float, float, float, float] | None) -> float:
    """Laplacian variance inside a normalized box (x, y, w, h) in [0,1]; whole image if None.

    Used to weight burst election toward a sharp SUBJECT. An empty or degenerate box
    falls back to the whole frame.
    """
    gray = _gray(thumb)
    if box is None:
        return float(_laplacian(gray).var())
    h, w = gray.shape
    x0 = max(0, int(box[0] * w))
    y0 = max(0, int(box[1] * h))
    x1 = min(w, int((box[0] + box[2]) * w))
    y1 = min(h, int((box[1] + box[3]) * h))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return float(_laplacian(gray).var())
    return float(_laplacian(gray[y0:y1, x0:x1]).var())
