"""Smooth membership ramps (P6)."""

from __future__ import annotations

from memory_graph.metrics.ramps import plateau, ramp_at_least, ramp_at_most


def test_plateau_inside_is_one() -> None:
    assert plateau(45, 35, 58, 12) == 1.0
    assert plateau(35, 35, 58, 12) == 1.0
    assert plateau(58, 35, 58, 12) == 1.0


def test_plateau_tapers_to_zero() -> None:
    # 12 below lo (35) => 0; halfway (6 below) => 0.5.
    assert plateau(23, 35, 58, 12) == 0.0
    assert abs(plateau(29, 35, 58, 12) - 0.5) < 1e-6
    # Above hi similarly.
    assert plateau(70, 35, 58, 12) == 0.0
    assert abs(plateau(64, 35, 58, 12) - 0.5) < 1e-6


def test_plateau_none_is_zero() -> None:
    assert plateau(None, 0, 1, 1) == 0.0


def test_ramp_at_least() -> None:
    assert abs(ramp_at_least(0.6, 0.6, 0.15) - 0.5) < 1e-6   # 0.5 at the knee
    assert ramp_at_least(0.6 + 0.15, 0.6, 0.15) == 1.0       # +fade => 1
    assert ramp_at_least(0.6 - 0.15, 0.6, 0.15) == 0.0       # -fade => 0
    assert ramp_at_least(None, 0.6, 0.15) == 0.0


def test_ramp_at_most() -> None:
    # Fast shutter: shutter <= knee. 0.5 at knee, 1 well below, 0 well above.
    assert abs(ramp_at_most(0.001, 0.001, 0.001) - 0.5) < 1e-6
    assert ramp_at_most(0.0, 0.001, 0.001) == 1.0
    assert ramp_at_most(0.01, 0.001, 0.001) == 0.0
    assert ramp_at_most(None, 1.0, 1.0) == 0.0


def test_no_cliff_monotonic() -> None:
    # Membership changes smoothly, never jumps by the full range over a tiny step.
    prev = plateau(58, 35, 58, 12)
    for x in range(59, 71):
        cur = plateau(float(x), 35, 58, 12)
        assert cur <= prev
        assert prev - cur < 0.2  # small steps => small changes (no cliff)
        prev = cur
