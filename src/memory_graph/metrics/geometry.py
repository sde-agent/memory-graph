"""Geometry metrics: horizon tilt and negative space (spec.md §4).

- `horizon(thumb)` estimates the dominant near-horizontal line angle (Dutch-angle cue
  for Q5) and the vertical position of the strongest horizontal edge band (low-horizon
  cue for Q2). Implemented with numpy Sobel gradients — no OpenCV dependency — so it is
  pure and portable (and iOS-friendly for V3).
- `negative_space(boxes)` is 1 minus the fraction of the frame covered by salient
  objects (Q4 cue). It consumes the Apple Vision saliency boxes from `vision.py`, so it
  degrades to None when Vision is unavailable rather than guessing.

Both are deterministic given their inputs (evaluation.md B.5): a programmatically
tilted image yields its known angle; a known box layout yields its known coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Horizon:
    angle_deg: float | None       # dominant near-horizontal tilt, signed, [-45, 45]
    y_ratio: float | None         # vertical position of the horizon band, 0 (top)..1 (bottom)
    dutch: bool                   # True if a CONFIDENT tilt is in the Dutch-angle range
    confidence: float = 0.0       # how line-like the detected edge is, 0..1


def _sobel_y(gray: np.ndarray) -> np.ndarray:
    """Vertical-gradient Sobel response (strong on horizontal edges)."""
    p = np.pad(gray, 1, mode="reflect")
    return (
        (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
        - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    )


def horizon(
    thumb: Any,
    dutch_from: float = 4.5,
    dutch_to: float = 35.0,
    min_strength: float = 0.15,
    min_confidence: float = 0.5,
    band_frac: float = 0.03,
) -> Horizon:
    """Estimate the dominant near-horizontal line's tilt, position, and how line-like it is.

    Approach: for each column, the horizon crosses at the row with the strongest
    horizontal-edge response (large vertical gradient |gy|). Fitting a straight line to
    those per-column crossing rows gives the tilt (slope -> angle) and the mean crossing
    gives the vertical position. This recovers a global line robustly, where a per-pixel
    gradient-angle median collapses to 0 on rasterized ('staircased') edges.

    CONFIDENCE guards against false tilt: on a telephoto shot with no true horizon, the
    per-column edges are scattered and the fit is meaningless. We measure the fraction
    of selected columns whose edge row lies within a thin band of the fitted line; only
    a confident fit may set the Dutch flag. (Finding from P6: without this, tele nature
    shots spuriously scored Q5 'provocative'.)
    """
    gray = np.asarray(thumb.convert("L"), dtype=np.float32)
    h, w = gray.shape
    if h < 4 or w < 4:
        return Horizon(angle_deg=None, y_ratio=None, dutch=False, confidence=0.0)
    gy = _sobel_y(gray)
    vmag = np.abs(gy)
    if vmag.max() <= 0:
        return Horizon(angle_deg=None, y_ratio=None, dutch=False, confidence=0.0)

    rows = np.argmax(vmag, axis=0).astype(np.float32)
    strength = vmag.max(axis=0)
    threshold = min_strength * vmag.max()
    sel = strength >= threshold
    if sel.sum() < max(4, w // 10):
        return Horizon(angle_deg=None, y_ratio=None, dutch=False, confidence=0.0)

    xs = np.arange(w, dtype=np.float32)[sel]
    ys = rows[sel]
    wts = strength[sel]

    sw = wts.sum()
    mx = (wts * xs).sum() / sw
    my = (wts * ys).sum() / sw
    cov = (wts * (xs - mx) * (ys - my)).sum()
    var = (wts * (xs - mx) ** 2).sum()
    slope = cov / var if var > 0 else 0.0
    intercept = my - slope * mx

    # Confidence: fraction of selected columns whose edge row sits within a thin band
    # (band_frac of the image height) of the fitted line. A real horizon => most columns
    # agree; scattered tele edges => few agree.
    predicted = slope * xs + intercept
    band = band_frac * h
    within = np.abs(ys - predicted) <= band
    confidence = float(within.mean())

    angle = float(np.degrees(np.arctan(slope)))
    angle = max(-45.0, min(45.0, angle))
    y_ratio = float(my / max(1, h - 1))
    dutch = (
        confidence >= min_confidence and dutch_from <= abs(angle) <= dutch_to
    )
    return Horizon(angle_deg=angle, y_ratio=y_ratio, dutch=dutch, confidence=confidence)


def negative_space(boxes: dict[str, list[dict[str, float]]] | None) -> float | None:
    """Fraction of the frame NOT covered by salient objects, 0..1, or None if no boxes.

    Uses the 'attention' request when present (where a viewer looks), else 'objectness'.
    Coverage is the union area of the salient boxes on a coarse grid (so overlapping
    boxes are not double-counted).
    """
    if not boxes:
        return None
    chosen = boxes.get("attention") or boxes.get("objectness") or []
    if not chosen:
        return 1.0  # Vision ran but found nothing salient => all negative space.
    grid = 100
    mask = np.zeros((grid, grid), dtype=bool)
    for b in chosen:
        x0 = max(0, min(grid, int(b["x"] * grid)))
        y0 = max(0, min(grid, int(b["y"] * grid)))
        x1 = max(0, min(grid, int((b["x"] + b["w"]) * grid)))
        y1 = max(0, min(grid, int((b["y"] + b["h"]) * grid)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    covered = mask.mean()
    return float(1.0 - covered)
