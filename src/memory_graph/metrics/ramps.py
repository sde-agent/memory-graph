"""Smooth membership ramps for the quadrant scorer (spec.md §4, product.md).

Quadrant membership must be SOFT — no boundary cliffs. A value just outside a range
should score a little, not zero; a 1mm change in focal length must not flip a hard
class. These helpers turn a scalar into a [0,1] membership with linear tapers.

Two shapes cover every quadrant cue:
  - `plateau(x, lo, hi, fade)`: 1.0 inside [lo, hi], tapering linearly to 0 across
    `fade` on each side. (A focal "normal" range, an aperture band, etc.)
  - `ramp_at_least(x, knee, fade)` / `ramp_at_most(x, knee, fade)`: a one-sided soft
    threshold — 0.5 at the knee, saturating to 1/0 across `fade`. (Fast shutter and
    wide aperture use `ramp_at_most`; large negative space and horizon tilt use
    `ramp_at_least`.)
"""

from __future__ import annotations


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def plateau(x: float | None, lo: float, hi: float, fade: float) -> float:
    """1.0 within [lo, hi]; linear taper to 0 across `fade` on each side; 0 if x is None."""
    if x is None:
        return 0.0
    if fade <= 0:
        return 1.0 if lo <= x <= hi else 0.0
    if x < lo:
        return _clamp01(1.0 - (lo - x) / fade)
    if x > hi:
        return _clamp01(1.0 - (x - hi) / fade)
    return 1.0


def ramp_at_least(x: float | None, knee: float, fade: float) -> float:
    """Softly 'x >= knee': 0 at knee-fade, 0.5 at knee, 1 at knee+fade."""
    if x is None:
        return 0.0
    if fade <= 0:
        return 1.0 if x >= knee else 0.0
    return _clamp01((x - knee) / (2 * fade) + 0.5)


def ramp_at_most(x: float | None, knee: float, fade: float) -> float:
    """Softly 'x <= knee': 1 at knee-fade, 0.5 at knee, 0 at knee+fade.

    Used for 'fast shutter' (shutter <= knee) and 'wide aperture' (f-number <= knee).
    """
    if x is None:
        return 0.0
    if fade <= 0:
        return 1.0 if x <= knee else 0.0
    return _clamp01((knee - x) / (2 * fade) + 0.5)
