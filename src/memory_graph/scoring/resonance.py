"""The resonance score (spec.md §6, evaluation.md Part A).

A per-photo signal that ranks the daily dispatch (and feeds the LLM critique). It is a
multiplicative blend so each factor degrades gracefully to 1.0 when its signal is
unavailable — a photo is never penalized for something we could not compute.

    resonance = base_quadrant_affinity
              * candid_multiplier
              * favorite_multiplier
              * known_person_boost
              * taste_model_multiplier        # V1.5 (defaults to 1.0 in V1)

Two per-term modifiers live INSIDE base_quadrant_affinity (not as global multipliers):
  - age_boost amplifies ONLY the Q1 (nostalgia) term — age does not make a clinical
    ultra-wide resonant.
  - warm_gate scales the Q2 term down when the photo is NOT warm-lit, so a cold
    ultra-wide does not read as "warm aliveness".

All weights and modifier magnitudes come from config (V2 portability, the tuning
surface). Nothing here is a final value — these are a pre-registered first draft.
"""

from __future__ import annotations

from dataclasses import dataclass

from memory_graph.config import ResonanceModifiers, ResonanceWeights
from memory_graph.metrics.router import QuadrantVector


@dataclass(frozen=True)
class ResonanceInputs:
    quadrants: QuadrantVector
    age_years: float | None = None       # photo age; None if capture date unknown
    warm: bool = False                   # light.is_warm (gates the Q2 term)
    candid_score: float | None = None    # 0 posed .. 1 candid; None if no faces
    favorite: bool = False               # Library favorite flag
    known_person: bool = False           # a named person above the precision bar
    face_present: bool = False           # any face detected (strong strict-dispatch lift)
    taste_score: float | None = None     # V1.5 model output in [0,1]; None in V1
    sharpness: float | None = None       # Laplacian variance; gentle blur de-weight only


def age_boost(age_years: float | None, mods: ResonanceModifiers) -> float:
    """Linear nostalgia amplifier: 1.0 now, rising to age_max_boost by age_full_years.

    Applied ONLY to the Q1 term. None age (unknown date) => no boost (1.0).
    """
    if age_years is None or age_years <= 0:
        return 1.0
    frac = min(1.0, age_years / mods.age_full_years)
    return 1.0 + frac * (mods.age_max_boost - 1.0)


def _base_quadrant_affinity(
    inp: ResonanceInputs, w: ResonanceWeights, mods: ResonanceModifiers
) -> float:
    q = inp.quadrants
    boost = age_boost(inp.age_years, mods)
    warm_gate = 1.0 if inp.warm else mods.warm_gate_cold
    return (
        w.q1 * q.q1 * boost
        + w.q2 * q.q2 * warm_gate
        + w.q3 * q.q3
        + w.q4 * q.q4
        + w.q5 * q.q5
    )


def _candid_multiplier(candid_score: float | None, mods: ResonanceModifiers) -> float:
    """Candid > posed. Interpolate between posed_penalty and candid_bonus by the score.

    None (no faces / unknown) => neutral 1.0. A score of 0 (posed) => posed_penalty;
    1 (candid) => candid_bonus; 0.5 => ~1.0.
    """
    if candid_score is None:
        return 1.0
    lo, hi = mods.posed_penalty, mods.candid_bonus
    return float(lo + (hi - lo) * candid_score)


def resonance(
    inp: ResonanceInputs,
    weights: ResonanceWeights,
    mods: ResonanceModifiers,
) -> float:
    """Compute the resonance score. Non-negative; unbounded above but typically ~0..3.

    Part C (tuned to the labeling pass): the quadrant affinity is a SOFT MINOR input
    (scaled by `quadrant_weight`) plus a small floor, so the MULTIPLIERS — known person,
    candid, favorite, taste — drive the ranking, matching the owner's measured
    preferences (known_person was the strongest signal at a 68% pairwise winner-rate).
    """
    # A small floor keeps every photo in play; the quadrant adds a modest, weighted lift.
    base = 0.4 + mods.quadrant_weight * _base_quadrant_affinity(inp, weights, mods)
    score = base
    score *= _candid_multiplier(inp.candid_score, mods)
    score *= mods.favorite_bonus if inp.favorite else 1.0
    # A known person implies a face; apply the stronger known-person bonus in that case,
    # else the face-present bonus for an unnamed face.
    if inp.known_person:
        score *= mods.known_person_bonus
    elif inp.face_present:
        score *= mods.face_present_bonus
    if inp.taste_score is not None:
        # Taste in [0,1] -> a multiplier centered at 1.0, its influence set by taste_weight
        # (0.5 -> 1.0; 1.0 -> 1 + 0.5*taste_weight).
        score *= 1.0 + mods.taste_weight * (inp.taste_score - 0.5)
    score *= _blur_multiplier(inp.sharpness, mods)
    return float(max(0.0, score))


def _blur_multiplier(sharpness: float | None, mods: ResonanceModifiers) -> float:
    """Gentle de-weight for very blurry photos (NOT a reject). 1.0 at/above the floor,
    ramping down to `blur_deweight` as sharpness falls toward half the floor. None
    (unknown) => 1.0 (never penalize an uncomputable signal)."""
    if sharpness is None or sharpness >= mods.blur_floor:
        return 1.0
    half = mods.blur_floor / 2.0
    if sharpness <= half:
        return mods.blur_deweight
    # Linear between half-floor (blur_deweight) and floor (1.0).
    frac = (sharpness - half) / (mods.blur_floor - half)
    return mods.blur_deweight + frac * (1.0 - mods.blur_deweight)
