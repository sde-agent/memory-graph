"""Resonance scoring (P7)."""

from __future__ import annotations

from memory_graph.config import ResonanceModifiers, ResonanceWeights
from memory_graph.metrics.router import QuadrantVector
from memory_graph.scoring.resonance import ResonanceInputs, age_boost, resonance


def cfg():
    class TestConfig:
        resonance_weights = ResonanceWeights(q1=0.55, q2=0.45, q3=0.80, q4=0.55, q5=0.25)
        resonance_modifiers = ResonanceModifiers(
            age_full_years=15.0, age_max_boost=1.8, warm_gate_cold=0.4,
            candid_bonus=1.05, posed_penalty=0.95, favorite_bonus=1.5,
            known_person_bonus=2.0, face_present_bonus=1.3, quadrant_weight=0.6,
            taste_weight=1.0, blur_floor=60.0, blur_deweight=1.0,
        )
    return TestConfig()


def _res(inp: ResonanceInputs) -> float:
    c = cfg()
    return resonance(inp, c.resonance_weights, c.resonance_modifiers)


def _q(q1=0, q2=0, q3=0, q4=0, q5=0) -> QuadrantVector:
    return QuadrantVector(q1=q1, q2=q2, q3=q3, q4=q4, q5=q5)


def test_q3_outranks_q5_for_equal_scores() -> None:
    # Personal priority: Q3 (tenderness) >> Q5 (tension).
    q3 = _res(ResonanceInputs(quadrants=_q(q3=1.0)))
    q5 = _res(ResonanceInputs(quadrants=_q(q5=1.0)))
    assert q3 > q5


def test_age_boost_only_lifts_q1() -> None:
    young_q1 = _res(ResonanceInputs(quadrants=_q(q1=1.0), age_years=0))
    old_q1 = _res(ResonanceInputs(quadrants=_q(q1=1.0), age_years=20))
    assert old_q1 > young_q1   # age amplifies nostalgia

    # ...but age does NOT lift a non-Q1 (e.g. Q2) photo.
    young_q2 = _res(ResonanceInputs(quadrants=_q(q2=1.0), age_years=0, warm=True))
    old_q2 = _res(ResonanceInputs(quadrants=_q(q2=1.0), age_years=20, warm=True))
    assert abs(old_q2 - young_q2) < 1e-9


def test_age_boost_curve() -> None:
    mods = cfg().resonance_modifiers
    assert age_boost(None, mods) == 1.0
    assert age_boost(0, mods) == 1.0
    assert abs(age_boost(mods.age_full_years, mods) - mods.age_max_boost) < 1e-6
    # Beyond full years, capped at max.
    assert abs(age_boost(mods.age_full_years * 3, mods) - mods.age_max_boost) < 1e-6


def test_warm_gate_dampens_cold_q2() -> None:
    warm = _res(ResonanceInputs(quadrants=_q(q2=1.0), warm=True))
    cold = _res(ResonanceInputs(quadrants=_q(q2=1.0), warm=False))
    assert warm > cold   # cold ultra-wide is not "warm aliveness"


def test_favorite_boosts() -> None:
    plain = _res(ResonanceInputs(quadrants=_q(q3=1.0), favorite=False))
    fav = _res(ResonanceInputs(quadrants=_q(q3=1.0), favorite=True))
    assert fav > plain


def test_known_person_boosts() -> None:
    plain = _res(ResonanceInputs(quadrants=_q(q3=1.0), known_person=False))
    known = _res(ResonanceInputs(quadrants=_q(q3=1.0), known_person=True))
    assert known > plain


def test_candid_beats_posed() -> None:
    posed = _res(ResonanceInputs(quadrants=_q(q3=1.0), candid_score=0.0))
    candid = _res(ResonanceInputs(quadrants=_q(q3=1.0), candid_score=1.0))
    assert candid > posed


def test_missing_signals_are_neutral() -> None:
    # No favorite / known person / candid / taste => all multipliers are 1.0 (no penalty).
    # Score = floor + quadrant_weight * (weighted quadrant affinity).
    c = cfg()
    base = _res(ResonanceInputs(quadrants=_q(q3=1.0)))
    expected = 0.4 + c.resonance_modifiers.quadrant_weight * (c.resonance_weights.q3 * 1.0)
    assert abs(base - expected) < 1e-6


def test_resonance_non_negative() -> None:
    assert _res(ResonanceInputs(quadrants=_q())) >= 0.0


def test_blur_multiplier_ramps_when_enabled() -> None:
    # Test the blur de-weight function directly with an ENABLED config (default ships
    # neutral=1.0 after the strict-relabel finding that blur was anti-predictive).
    import dataclasses

    from memory_graph.scoring.resonance import _blur_multiplier

    mods = dataclasses.replace(cfg().resonance_modifiers, blur_floor=60.0, blur_deweight=0.7)
    assert _blur_multiplier(1000.0, mods) == 1.0        # above floor: no effect
    assert _blur_multiplier(5.0, mods) == 0.7           # well below: full de-weight
    assert 0.7 < _blur_multiplier(45.0, mods) < 1.0     # ramp region
    assert _blur_multiplier(None, mods) == 1.0          # unknown: neutral


def test_blur_disabled_by_default() -> None:
    # The shipped default neutralizes blur (owner's worthy photos skew softer).
    assert cfg().resonance_modifiers.blur_deweight == 1.0
    flat = _res(ResonanceInputs(quadrants=_q(q3=1.0), sharpness=5.0))
    sharp = _res(ResonanceInputs(quadrants=_q(q3=1.0), sharpness=1000.0))
    assert abs(flat - sharp) < 1e-9   # no blur effect with the default config
