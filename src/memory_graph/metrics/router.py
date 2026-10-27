"""The 5-quadrant soft scorer (spec.md §4, product.md Quadrant Scoring Model).

Turns the deterministic metrics (optics, geometry, light, faces) into a soft score
vector `[q1..q5]` in [0,1], using smooth ramps over the PRE-REGISTERED config ranges
(evaluation.md B.6 — ranges frozen before accuracy is measured against felt labels).
No winner-take-all cliffs: a photo can be "mostly Q1, a little Q3". `dominant_quadrant`
is the argmax, kept only as a convenience; the full vector is what persists and feeds
resonance.

Each quadrant's score is a blend of its cues from the intent matrix:
  Q1 Nostalgic   : normal focal, wide aperture, eye-level, warm/golden light.
  Q2 Visceral    : ultra-wide focal, fast shutter, low horizon.
  Q3 Empathetic  : tele focal, tight face crop, candid.
  Q4 Solitary    : large negative space OR long exposure.
  Q5 Provocative : Dutch tilt OR intentional shutter drag.

Cues within a quadrant are combined so that a STRONG single cue carries the quadrant
(max-like), while corroborating cues reinforce it (we average the top cues). Missing
inputs contribute 0, so a photo is scored on what we actually know.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory_graph.metrics.faces import FaceAttributes
from memory_graph.metrics.geometry import Horizon
from memory_graph.metrics.light import Light
from memory_graph.metrics.optics import Optics
from memory_graph.metrics.ramps import plateau, ramp_at_least, ramp_at_most


@dataclass(frozen=True)
class QuadrantVector:
    q1: float
    q2: float
    q3: float
    q4: float
    q5: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.q1, self.q2, self.q3, self.q4, self.q5)

    @property
    def dominant(self) -> int:
        """1-based index of the strongest quadrant (ties -> lowest index)."""
        scores = self.as_tuple()
        return int(max(range(5), key=lambda i: scores[i])) + 1


def _blend(*cues: float, core: float | None = None) -> float:
    """Combine corroborating cue memberships into a quadrant score (soft AND-ish).

    These quadrants (Q1/Q2/Q3) are defined by SEVERAL cues that together make the
    feeling — a normal focal length alone is only weakly "nostalgic"; it becomes
    strongly so with a wide aperture, eye-level framing, and warm light. So:
      - the score is the MEAN of the present cues (one lone cue -> partial membership,
        which is what we want: present, not certain),
      - lifted toward the max as more cues corroborate (so all-high -> near 1.0).
    `core` optionally names the quadrant's defining cue (e.g. a tight face crop for Q3);
    the score cannot exceed roughly the core cue plus corroboration, so a quadrant does
    not fire strongly when its essential cue is absent.
    """
    vals = [c for c in cues if c is not None]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    top = max(vals)
    # Blend mean (requires corroboration) with top (rewards a strong cue), averaged.
    score = 0.6 * mean + 0.4 * top
    if core is not None:
        # The defining cue caps the quadrant: no strong score without it.
        score = min(score, 0.4 + 0.6 * core)
    return float(min(1.0, score))


# Vision scene labels grouped by the feeling they signal (content cues, Part B).
_PEOPLE_IDS = frozenset({"people", "adult", "child", "baby", "face", "person", "crowd",
                         "group", "wedding", "portrait"})
_SCENIC_IDS = frozenset({"outdoor", "sky", "mountain", "landscape", "beach", "ocean",
                        "sea", "water", "sunset", "sunrise", "cloud", "cloudy", "snow",
                        "desert", "canyon", "valley", "hill", "field", "lake", "river",
                        "waterfall", "night_sky", "cityscape", "skyline"})
_DOMESTIC_IDS = frozenset({"indoor", "room", "home", "furniture", "food", "table",
                          "kitchen", "living_room", "bedroom", "pet", "dog", "cat"})


def _scene_has(labels, group: frozenset[str], threshold: float = 0.3) -> float:
    """Max confidence among scene labels that fall in `group` (0 if none)."""
    if not labels:
        return 0.0
    return max((c for name, c in labels if name.replace("_", " ") in
               {g.replace("_", " ") for g in group} and c >= threshold), default=0.0)


def score_quadrants(
    optics: Optics,
    horizon: Horizon,
    light: Light,
    faces: FaceAttributes,
    quad_cfg: dict[str, Any],
    *,
    negative_space: float | None = None,
    warm: bool = False,
    scene_labels: list[tuple[str, float]] | None = None,
) -> QuadrantVector:
    """Compute the soft quadrant vector from metric + CONTENT inputs and config ranges.

    `negative_space` is the geometry ratio or None; `warm` gates the warmth cue;
    `scene_labels` are Vision scene (identifier, confidence) pairs that supply CONTENT
    cues (Part B). Content matters because the owner's felt quadrants — especially Q1
    nostalgia and Q3 tenderness — are driven by WHAT/WHO is in the frame, not only by
    the technical execution EXIF captures (a finding from the labeling pass: the pure
    EXIF/geometry router agreed with felt labels only 25% of the time, missing Q1/Q3).
    """
    focal = optics.focal_35mm
    has_people = _scene_has(scene_labels, _PEOPLE_IDS) or (
        1.0 if faces.face_count and faces.face_count > 0 else 0.0
    )
    scenic = _scene_has(scene_labels, _SCENIC_IDS)
    domestic = _scene_has(scene_labels, _DOMESTIC_IDS)

    # ---- Q1 Nostalgic: a preserved human/domestic moment, often warm --------------
    # Content-first: people or a domestic scene in warm light reads as nostalgia even
    # without a textbook normal lens. The old focal-only core missed most felt-Q1.
    q1c = quad_cfg["q1_nostalgic"]
    q1_focal = plateau(focal, q1c["focal_full"][0], q1c["focal_full"][1], q1c["focal_fade"])
    q1_aperture = ramp_at_most(optics.aperture, q1c["aperture_wide_from"], q1c["aperture_fade"])
    q1_warm = 1.0 if (light.golden_hour or warm) else 0.0
    q1_content = max(has_people, domestic)
    # Core is now "a preserved moment" = people/domestic content OR a normal-lens warm
    # frame — far closer to how nostalgia actually reads.
    q1_core = max(q1_content, 0.5 * q1_focal + 0.5 * q1_warm)
    q1 = _blend(q1_content, q1_focal, q1_aperture, q1_warm, core=q1_core)

    # ---- Q2 Visceral: ultra-wide / fast shutter / low horizon / grand scenery ------
    q2c = quad_cfg["q2_visceral"]
    q2_focal = plateau(focal, q2c["focal_full"][0], q2c["focal_full"][1], q2c["focal_fade"])
    q2_shutter = ramp_at_most(optics.shutter, q2c["fast_shutter_from"], q2c["fast_shutter_from"])
    q2_low_horizon = (
        ramp_at_most(horizon.y_ratio, q2c["low_horizon_below"], 0.15)
        if horizon.y_ratio is not None
        else 0.0
    )
    q2_core = max(q2_focal, scenic)   # grand scenery also carries awe
    q2 = _blend(q2_focal, q2_shutter, q2_low_horizon, scenic, core=q2_core)

    # ---- Q3 Empathetic: faces, tenderness, candid ----------------------------------
    # A face present is the core — tightness and tele focal reinforce but are not
    # required (a candid of a person is tender even at a normal focal).
    q3c = quad_cfg["q3_empathetic"]
    q3_focal = plateau(focal, q3c["focal_full"][0], q3c["focal_full"][1], q3c["focal_fade"])
    q3_face_ratio = ramp_at_least(faces.face_ratio, q3c["face_ratio_from"], q3c["face_ratio_fade"])
    q3_candid = faces.candid_score if faces.candid_score is not None else 0.0
    q3_core = max(q3_face_ratio, 0.8 * has_people)   # any person present anchors Q3
    q3 = _blend(q3_face_ratio, q3_focal, q3_candid, has_people, core=q3_core)

    # ---- Q4 Solitary: negative space / long exposure / lone scenery, no crowd ------
    q4c = quad_cfg["q4_solitary"]
    q4_negspace = ramp_at_least(
        negative_space, q4c["negative_space_from"], q4c["negative_space_fade"]
    )
    q4_long = ramp_at_least(optics.shutter, q4c["long_exposure_from"], 1.0)
    # Scenery with few/no people reinforces solitude; a crowd suppresses it.
    lonely_scene = scenic * (1.0 if (faces.face_count or 0) <= 1 else 0.0)
    q4 = max(q4_negspace, q4_long, 0.7 * lonely_scene)

    # ---- Q5 Provocative: Dutch tilt OR shutter drag ---------------------------------
    q5c = quad_cfg["q5_provocative"]
    # Only a CONFIDENT horizon line counts as a Dutch tilt (geometry.horizon.dutch folds
    # in the line-fit confidence). This prevents tele shots with no true horizon — where
    # the edge fit is meaningless — from spuriously scoring Q5 (P6 finding).
    q5_tilt = (
        ramp_at_least(abs(horizon.angle_deg), q5c["dutch_tilt_from"], q5c["dutch_tilt_fade"])
        if (horizon.angle_deg is not None and horizon.dutch)
        else 0.0
    )
    q5_drag = plateau(optics.shutter, q5c["shutter_drag"][0], q5c["shutter_drag"][1], 0.1)
    q5 = max(q5_tilt, q5_drag)

    return QuadrantVector(q1=q1, q2=q2, q3=q3, q4=q4, q5=q5)


def _level_bonus(horizon: Horizon) -> float:
    """Eye-level membership: strongest when the horizon is level (not Dutch) and centered."""
    if horizon.angle_deg is None:
        return 0.0
    # Level = small tilt. 1.0 at 0 deg, fading out by ~6 deg.
    level = max(0.0, 1.0 - abs(horizon.angle_deg) / 6.0)
    return level
