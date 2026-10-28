"""Load configuration from config.toml, overlaid by an optional config.local.toml.

Every threshold the engine uses comes from here — never hard-coded. The default
`config.toml` ships with the repo and holds NO personal data; `config.local.toml`
(gitignored) overlays personal values (archive paths, crop factors, eras). A deep
merge lets the local file override individual keys without restating whole tables.

Config objects are frozen dataclasses with validation in __post_init__, so a bad
threshold fails loudly at load time rather than silently mid-run.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory_graph.paths import DEFAULT_CONFIG, LOCAL_CONFIG

# --------------------------------------------------------------------------- tables


@dataclass(frozen=True)
class IngestCfg:
    thumb_long_edge: int
    hdd_extensions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 128 <= self.thumb_long_edge <= 4096:
            raise ValueError(f"ingest.thumb_long_edge out of range: {self.thumb_long_edge}")
        if not all(e.startswith(".") and e == e.lower() for e in self.hdd_extensions):
            raise ValueError("ingest.hdd_extensions must be lowercase and start with '.'")


@dataclass(frozen=True)
class DedupCfg:
    burst_window_seconds: float
    phash_max_distance: int

    def __post_init__(self) -> None:
        if self.burst_window_seconds <= 0:
            raise ValueError("dedup.burst_window_seconds must be positive")
        if not 0 <= self.phash_max_distance <= 64:
            raise ValueError("dedup.phash_max_distance must be 0..64")


@dataclass(frozen=True)
class FramingCfg:
    bounds: tuple[float, ...]
    crop: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.bounds) != 5 or list(self.bounds) != sorted(set(self.bounds)):
            raise ValueError(f"framing.bounds must be 5 strictly increasing values: {self.bounds}")


@dataclass(frozen=True)
class SunCfg:
    night_below: float
    blue_hour_below: float
    golden_hour_below: float
    high_sun_from: float


@dataclass(frozen=True)
class ColorCfg:
    warm_from: float
    cool_below: float

    def __post_init__(self) -> None:
        if self.cool_below > self.warm_from:
            raise ValueError("color.cool_below must be <= color.warm_from")


@dataclass(frozen=True)
class ResonanceWeights:
    q1: float
    q2: float
    q3: float
    q4: float
    q5: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.q1, self.q2, self.q3, self.q4, self.q5)

    def __post_init__(self) -> None:
        if any(w < 0 for w in self.as_tuple()):
            raise ValueError("resonance weights must be non-negative")


@dataclass(frozen=True)
class ResonanceModifiers:
    age_full_years: float
    age_max_boost: float
    warm_gate_cold: float
    candid_bonus: float
    posed_penalty: float
    favorite_bonus: float
    known_person_bonus: float
    face_present_bonus: float = 1.0   # a face present (any identity) is dispatch-worthy
    quadrant_weight: float = 1.0   # scales the quadrant affinity (Part C: soft minor input)
    taste_weight: float = 1.0      # scales the taste multiplier's influence
    blur_floor: float = 60.0       # Laplacian variance below which blur de-weight ramps in
    blur_deweight: float = 0.7     # multiplier at/below ~half the floor (never a hard reject)

    def __post_init__(self) -> None:
        if self.age_full_years <= 0:
            raise ValueError("resonance.modifiers.age_full_years must be positive")
        if self.age_max_boost < 1.0:
            raise ValueError("resonance.modifiers.age_max_boost must be >= 1.0")
        if not 0 < self.warm_gate_cold <= 1.0:
            raise ValueError("resonance.modifiers.warm_gate_cold must be in (0, 1]")
        if not 0 < self.quadrant_weight <= 1.0:
            raise ValueError("resonance.modifiers.quadrant_weight must be in (0, 1]")


@dataclass(frozen=True)
class DispatchCfg:
    count_min: int
    count_max: int
    repeat_window_days: int
    max_favorite_share: float
    seed: int

    def __post_init__(self) -> None:
        if not 1 <= self.count_min <= self.count_max:
            raise ValueError("dispatch: require 1 <= count_min <= count_max")
        if not 0 < self.max_favorite_share <= 1.0:
            raise ValueError("dispatch.max_favorite_share must be in (0, 1]")


@dataclass(frozen=True)
class RejectionCfg:
    vision_confidence: float
    document_ids: tuple[str, ...]
    screenshot_ids: tuple[str, ...]
    low_contrast_below: float
    low_saturation_below: float
    text_coverage_reject: float
    uniformity_reject: float

    def __post_init__(self) -> None:
        if not 0 < self.vision_confidence < 1:
            raise ValueError("rejection.vision_confidence must be in (0, 1)")
        if not 0 < self.text_coverage_reject <= 1:
            raise ValueError("rejection.text_coverage_reject must be in (0, 1]")
        if not 0 < self.uniformity_reject <= 1:
            raise ValueError("rejection.uniformity_reject must be in (0, 1]")


@dataclass(frozen=True)
class FacesCfg:
    recognition_min_similarity: float

    def __post_init__(self) -> None:
        if not 0 < self.recognition_min_similarity < 1:
            raise ValueError("faces.recognition_min_similarity must be in (0, 1)")


@dataclass(frozen=True)
class IdentityCfg:
    """Experimental face identity settings; values require precision calibration."""

    feature_model_version: str
    gallery_path: str
    embeddings_path: str
    crop_padding: float
    anonymous_similarity_threshold: float
    anonymous_min_cluster_size: int
    max_references_per_name: int = 64

    def __post_init__(self) -> None:
        if not self.feature_model_version:
            raise ValueError("identity.feature_model_version must not be empty")
        if not 0 <= self.crop_padding <= 1:
            raise ValueError("identity.crop_padding must be in [0, 1]")
        if not -1 <= self.anonymous_similarity_threshold <= 1:
            raise ValueError("identity.anonymous_similarity_threshold must be in [-1, 1]")
        if self.anonymous_min_cluster_size < 2:
            raise ValueError("identity.anonymous_min_cluster_size must be at least 2")
        if self.max_references_per_name < 1:
            raise ValueError("identity.max_references_per_name must be positive")


@dataclass(frozen=True)
class PrivacyCfg:
    hidden_albums: tuple[str, ...]


@dataclass(frozen=True)
class OwnerCfg:
    """The archive owner's OWN identity in the named-people graph.

    The engine assumes "you" = the photographer. But when the owner is IN a frame, someone
    else held the camera, so the technical choices in that photo are not the owner's. The
    owner's self-names let the insight generators treat owner-present photos correctly: the
    owner is never a 'subject you chose to frame' (no signature), and a pairing with the
    owner means 'you appear alongside X', not 'you photographed X and Y together'. Empty by
    default (no personal data in the repo); set in gitignored config.local.toml.
    """

    self_names: tuple[str, ...] = ()

    def is_self(self, name: str) -> bool:
        return name in self.self_names


@dataclass(frozen=True)
class EraRange:
    label: str
    start: int   # inclusive year
    end: int     # exclusive year


@dataclass(frozen=True)
class ErasCfg:
    ranges: tuple[EraRange, ...]

    def era_of(self, year: int | None) -> str | None:
        if year is None:
            return None
        for r in self.ranges:
            if r.start <= year < r.end:
                return r.label
        return None


@dataclass(frozen=True)
class NarrationCfg:
    backend: str
    max_regenerations: int

    def __post_init__(self) -> None:
        if self.backend not in ("foundation_models", "claude_cli"):
            raise ValueError(f"narration.backend: {self.backend!r}")
        if self.max_regenerations < 0:
            raise ValueError("narration.max_regenerations must be >= 0")


@dataclass(frozen=True)
class Config:
    ingest: IngestCfg
    dedup: DedupCfg
    framing: FramingCfg
    sun: SunCfg
    color: ColorCfg
    resonance_weights: ResonanceWeights
    resonance_modifiers: ResonanceModifiers
    dispatch: DispatchCfg
    rejection: RejectionCfg
    faces: FacesCfg
    identity: IdentityCfg
    privacy: PrivacyCfg
    owner: OwnerCfg
    eras: ErasCfg
    narration: NarrationCfg
    # The raw quadrant ranges are kept as a plain dict: the router owns their shape,
    # and we do not want to hard-type ramp parameters that are pure tuning knobs.
    quadrants: dict[str, Any]


# --------------------------------------------------------------------------- loading


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively overlay `overlay` onto `base`, returning a new dict."""
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _read_toml(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(
    default_path: Path = DEFAULT_CONFIG,
    local_path: Path = LOCAL_CONFIG,
) -> Config:
    """Load the default config, overlaid by the optional local config, into a Config."""
    raw = _read_toml(default_path)
    if local_path.is_file():
        raw = _deep_merge(raw, _read_toml(local_path))
    return _build(raw)


def save_local_resonance_profile(weights: ResonanceWeights, modifiers: ResonanceModifiers) -> None:
    """Persist a manual resonance profile in the gitignored local overlay."""
    path = LOCAL_CONFIG
    text = path.read_text() if path.is_file() else ""
    lines = text.splitlines()
    kept: list[str] = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped in ("[resonance.weights]", "[resonance.modifiers]"):
            skip = True
            continue
        if skip and stripped.startswith("[") and stripped.endswith("]"):
            skip = False
        if not skip:
            kept.append(line)
    block = f"""# Local resonance profile written by the tuning panel.
[resonance.weights]
q1 = {weights.q1}
q2 = {weights.q2}
q3 = {weights.q3}
q4 = {weights.q4}
q5 = {weights.q5}

[resonance.modifiers]
age_full_years = {modifiers.age_full_years}
age_max_boost = {modifiers.age_max_boost}
warm_gate_cold = {modifiers.warm_gate_cold}
candid_bonus = {modifiers.candid_bonus}
posed_penalty = {modifiers.posed_penalty}
favorite_bonus = {modifiers.favorite_bonus}
known_person_bonus = {modifiers.known_person_bonus}
face_present_bonus = {modifiers.face_present_bonus}
quadrant_weight = {modifiers.quadrant_weight}
taste_weight = {modifiers.taste_weight}
blur_floor = {modifiers.blur_floor}
blur_deweight = {modifiers.blur_deweight}
"""
    path.write_text("\n".join(kept).rstrip() + "\n\n" + block)


def _build(raw: dict) -> Config:
    framing = raw["framing"]
    res = raw["resonance"]
    return Config(
        ingest=IngestCfg(
            thumb_long_edge=raw["ingest"]["thumb_long_edge"],
            hdd_extensions=tuple(raw["ingest"]["hdd_extensions"]),
        ),
        dedup=DedupCfg(**raw["dedup"]),
        framing=FramingCfg(
            bounds=tuple(framing["bounds"]),
            crop=dict(framing.get("crop", {})),
        ),
        sun=SunCfg(**raw["sun"]),
        color=ColorCfg(**raw["color"]),
        resonance_weights=ResonanceWeights(**res["weights"]),
        resonance_modifiers=ResonanceModifiers(**res["modifiers"]),
        dispatch=DispatchCfg(**raw["dispatch"]),
        rejection=RejectionCfg(
            vision_confidence=raw["rejection"]["vision_confidence"],
            document_ids=tuple(raw["rejection"]["document_ids"]),
            screenshot_ids=tuple(raw["rejection"]["screenshot_ids"]),
            low_contrast_below=raw["rejection"]["low_contrast_below"],
            low_saturation_below=raw["rejection"]["low_saturation_below"],
            text_coverage_reject=raw["rejection"]["text_coverage_reject"],
            uniformity_reject=raw["rejection"]["uniformity_reject"],
        ),
        faces=FacesCfg(**raw["faces"]),
        identity=IdentityCfg(**raw["identity"]),
        privacy=PrivacyCfg(hidden_albums=tuple(raw["privacy"]["hidden_albums"])),
        owner=OwnerCfg(self_names=tuple(raw.get("owner", {}).get("self_names", []))),
        eras=ErasCfg(
            ranges=tuple(
                EraRange(label=r["label"], start=r["start"], end=r["end"])
                for r in raw["eras"]["ranges"]
            )
        ),
        narration=NarrationCfg(**raw["narration"]),
        quadrants=dict(raw["quadrants"]),
    )
