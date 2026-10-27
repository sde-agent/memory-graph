"""Pixel statistics over the thumbnail (pure, numpy-based).

Used first by the rejection filter's "flat / nothing" check: a blank wall, an
over-exposed scan, or a near-featureless document tends to have almost no luminance
contrast and almost no color saturation. Also provides the warm-cool tint the light
metrics and the Q1 warmth signal reuse later.

All values are normalized to [0, 1] (or centered around 0 for warm_cool), so thresholds
in config are resolution-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PixelStats:
    contrast: float      # std-dev of luminance, 0..1
    saturation: float    # mean saturation of lit pixels, 0..1
    warm_cool: float     # mean(R) - mean(B), roughly -1..1 (positive = warm)
    uniformity: float    # 1 - spatial color variation; ~1 = a flat single-color frame


def pixel_stats(thumb: Any) -> PixelStats:
    """Compute contrast, saturation, and warm-cool tint from a PIL RGB thumbnail."""
    arr = np.asarray(thumb.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Luminance (Rec. 601) and its spread.
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    contrast = float(luma.std())

    # HSV-style saturation = (max - min) / max, over lit pixels (avoid 0/0 on black).
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    lit = mx > 0.05
    if lit.any():
        sat = np.zeros_like(mx)
        sat[lit] = (mx[lit] - mn[lit]) / mx[lit]
        saturation = float(sat[lit].mean())
    else:
        saturation = 0.0

    warm_cool = float(r.mean() - b.mean())

    # Spatial uniformity: a flat single-color frame (black, purple, a blank wall) has
    # near-zero per-channel spatial std-dev REGARDLESS of hue — this is what separates a
    # uniform purple wall from a vivid purple sunset (which varies across the frame).
    # uniformity -> 1.0 for a perfectly flat frame, -> 0 for a varied scene.
    spatial_std = float(np.mean([r.std(), g.std(), b.std()]))
    uniformity = max(0.0, 1.0 - spatial_std / 0.15)  # 0.15 std ~ clearly varied

    return PixelStats(
        contrast=contrast, saturation=saturation, warm_cool=warm_cool, uniformity=uniformity
    )
