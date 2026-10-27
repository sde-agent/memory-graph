"""Face attributes and the candidness proxy (spec.md §4, evaluation.md B.4).

Turns raw face detections into the signals resonance and the quadrant router need:
  - `face_count`     : how many faces (people-together cue, resonance factor #2).
  - `face_ratio`     : largest face area / frame (Q3 tight-crop cue, resonance #1).
  - `eyes_open`      : helps burst election pick the frame with open eyes (P3 hook).
  - `candid_score`   : 0 (clearly posed) .. 1 (clearly candid). A ROUGH v1 heuristic
                       whose job is only to nudge resonance (candid > posed), so it is
                       kept conservative and its weight modest until validated.

Faces can come from Apple Vision (`vision.detect_faces`, both sources) or from the
Photos Library's own face metadata (library_meta["faces"]). Either path produces the
same `FaceAttributes`. When no face information exists, attributes are empty (0 faces,
neutral candidness) — a photo is never penalized for an uncomputable signal.

The candidness proxy: a posed shot tends to be few faces, large and frontal, eyes open,
looking at the camera. Candid tends to be off-angle (head turned), smaller faces, or
many faces not all frontal. We estimate "frontal-ness" from |yaw| when Vision exposes
it. This is intentionally crude for v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# An eye open-ratio below this reads as "eyes likely closed" (blink).
_EYES_OPEN_MIN = 0.18
# |yaw| (radians) above this reads as a turned/averted head => more candid.
_FRONTAL_YAW = 0.35


@dataclass(frozen=True)
class FaceAttributes:
    face_count: int
    face_ratio: float | None       # largest face area / frame, 0..1
    eyes_open: bool | None         # the largest face's eyes appear open
    candid_score: float | None     # 0 posed .. 1 candid; None if no faces


def _largest(faces: list[dict[str, Any]]) -> dict[str, Any] | None:
    boxes = [f for f in faces if f.get("area") or (f.get("w") and f.get("h"))]
    if not boxes:
        return None

    def area(f: dict[str, Any]) -> float:
        return float(f.get("area") or (f.get("w", 0.0) * f.get("h", 0.0)))

    return max(boxes, key=area)


def from_vision(faces: list[dict[str, Any]] | None) -> FaceAttributes:
    """Build attributes from `vision.detect_faces` output."""
    if not faces:
        return FaceAttributes(face_count=0, face_ratio=None, eyes_open=None, candid_score=None)
    largest = _largest(faces)
    ratio = None
    eyes_open = None
    if largest is not None:
        ratio = float(largest.get("area") or (largest.get("w", 0) * largest.get("h", 0)))
        openness = largest.get("eye_openness")
        eyes_open = (openness >= _EYES_OPEN_MIN) if openness is not None else None
    return FaceAttributes(
        face_count=len(faces),
        face_ratio=ratio,
        eyes_open=eyes_open,
        candid_score=_candid_score(faces),
    )


def from_library_meta(meta: dict[str, Any] | None) -> FaceAttributes:
    """Build attributes from the Photos Library's own face metadata.

    The Library gives per-face size + `has_smile` + `yaw`, so we get real candidness:
    a posed shot is frontal (|yaw| small) and smiling; a turned head or non-smiling
    face leans candid. (Identity names are read separately via `named_people`.)
    """
    faces = (meta or {}).get("faces") or []
    boxes = []
    for f in faces:
        size = f.get("size")
        if size:
            boxes.append(
                {"area": float(size) * float(size), "yaw": f.get("yaw"),
                 "has_smile": f.get("has_smile")}
            )
    if not boxes:
        return FaceAttributes(face_count=0, face_ratio=None, eyes_open=None, candid_score=None)
    ratio = max(b["area"] for b in boxes)
    return FaceAttributes(
        face_count=len(boxes),
        face_ratio=float(ratio),
        eyes_open=None,  # Library has no reliable per-face eye-open signal
        candid_score=_candid_score_library(boxes),
    )


def named_people(meta: dict[str, Any] | None) -> list[str]:
    """Authoritative named people on a Library photo, from Photos' own face recognition.

    This is the identity source for Library photos — no embedding or matching needed.
    """
    faces = (meta or {}).get("faces") or []
    names = [f.get("name") for f in faces if f.get("name")]
    # De-dupe while preserving order.
    seen: set[str] = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _candid_score_library(boxes: list[dict[str, Any]]) -> float:
    """Candidness from the Library's per-face smile + yaw. 0 posed .. 1 candid.

    Posed: frontal (|yaw| small) AND smiling. Candid: turned head or not smiling.
    """
    n = len(boxes)
    posed_signals = 0.0
    for b in boxes:
        yaw = b.get("yaw")
        smiling = bool(b.get("has_smile"))
        frontal = (yaw is None) or (abs(float(yaw)) < _FRONTAL_YAW)
        if frontal and smiling:
            posed_signals += 1.0
        elif frontal:
            posed_signals += 0.5  # frontal but not smiling: ambiguous
    posed_frac = posed_signals / n if n else 0.5
    candid = 1.0 - posed_frac
    if n == 1 and posed_frac >= 1.0:
        candid = min(candid, 0.2)
    return float(max(0.0, min(1.0, candid)))


def _candid_score(faces: list[dict[str, Any]]) -> float:
    """Rough candidness from head-turn and face count. 0 posed .. 1 candid.

    Heuristic:
      - A single large frontal face (looking at camera) reads posed  -> low.
      - Turned heads (|yaw| high) or averted gaze read candid         -> high.
      - More faces, not all frontal, lean candid.
    """
    n = len(faces)
    # Fraction of faces that look frontal (|yaw| small). Unknown yaw counts as frontal
    # (conservative: we do not over-claim "candid").
    frontal = 0
    known = 0
    for f in faces:
        yaw = f.get("yaw")
        if yaw is None:
            frontal += 1
            continue
        known += 1
        if abs(yaw) < _FRONTAL_YAW:
            frontal += 1
    frontal_frac = frontal / n if n else 1.0

    # Posed-ness rises with frontal fraction and falls with face count variety.
    posed = frontal_frac
    candid = 1.0 - posed
    # A lone, fully frontal face is the canonical posed portrait: pin it low.
    if n == 1 and frontal_frac >= 1.0:
        candid = min(candid, 0.2)
    return float(max(0.0, min(1.0, candid)))


def face_attributes(
    *,
    vision_faces: list[dict[str, Any]] | None = None,
    library_meta: dict[str, Any] | None = None,
) -> FaceAttributes:
    """Prefer Vision detections (richer: eyes, yaw); fall back to Library metadata."""
    if vision_faces is not None:
        return from_vision(vision_faces)
    return from_library_meta(library_meta)


def crop_face(thumb: Any, face: dict[str, Any], pad: float = 0.25) -> Any:
    """Crop the face region (normalized top-left bbox) from a PIL thumbnail, with padding.

    Recognition embeds the FACE, not the whole frame, so a small face in a big scene is
    still matched on its own features. `pad` expands the box for context (hair, chin).
    """
    w, h = thumb.size
    x = face.get("x", 0.0)
    y = face.get("y", 0.0)
    bw = face.get("w", 0.0)
    bh = face.get("h", 0.0)
    px, py = bw * pad, bh * pad
    left = max(0, int((x - px) * w))
    top = max(0, int((y - py) * h))
    right = min(w, int((x + bw + px) * w))
    bottom = min(h, int((y + bh + py) * h))
    if right - left < 2 or bottom - top < 2:
        return thumb
    return thumb.crop((left, top, right, bottom))
