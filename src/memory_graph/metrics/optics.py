"""Optical primitives from EXIF (spec.md §4).

Turns the normalized EXIF dict into the optical values the 5-quadrant router consumes:
35mm-equivalent focal length, aperture (f-number), shutter (seconds), ISO, and a
derived exposure value (EV). Pure function of EXIF (and a per-camera crop table from
config) — no pixels, no I/O — so it is exactly unit-testable against known inputs
(evaluation.md B.5).

35mm-equivalent focal, in priority order:
  1. EXIF `focal35` (FocalLengthIn35mmFormat) when present — the camera's own answer.
  2. focal x crop-factor from the config [framing.crop] table, keyed by camera model.
  3. raw focal as a last resort (better than nothing for bucketing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from memory_graph.config import Config


@dataclass(frozen=True)
class Optics:
    focal_35mm: float | None
    aperture: float | None     # f-number
    shutter: float | None      # seconds
    iso: int | None
    ev: float | None           # exposure value at ISO 100
    focal_raw: float | None = None     # the EXIF lens/physical focal (pre-crop), persisted
    exif_focal35: float | None = None  # the camera's own 35mm-equiv, if it supplied one


def recompute_focal_35mm(
    *,
    model: str | None,
    exif_focal35: float | None,
    raw_focal: float | None,
    cfg: Config,
    stored_focal: float | None = None,
) -> float | None:
    """The 35mm-equivalent focal, derived DETERMINISTICALLY from the raw inputs.

    Priority: (1) the camera's own 35mm-equiv when it supplied one; (2) raw focal x the
    per-model crop factor from config; (3) raw focal as a last resort. Because it always
    works from the RAW focal (never from a previously-stored equivalent), a re-score is
    idempotent and immune to whatever was stored before — re-running after a crop-table
    change always lands on the same correct value. `stored_focal` is only a fallback when
    the raw focal was not persisted (pre-B2 rows).
    """
    if exif_focal35:
        return float(exif_focal35)
    focal = raw_focal if raw_focal is not None else stored_focal
    if focal and model:
        crop = cfg.framing.crop.get(model)
        if crop:
            return float(focal) * float(crop)
    return float(focal) if focal else None


def compute_ev(aperture: float | None, shutter: float | None, iso: int | None
               ) -> float | None:
    """Exposure value normalized to ISO 100: EV = log2(N^2 / t) - log2(ISO/100).

    Requires aperture and shutter; ISO defaults to 100 when unknown.
    """
    if not aperture or not shutter or shutter <= 0 or aperture <= 0:
        return None
    base = math.log2((aperture * aperture) / shutter)
    if iso and iso > 0:
        base -= math.log2(iso / 100.0)
    return base


def extract_optics(exif: dict[str, Any], cfg: Config) -> Optics:
    """Build the Optics record from normalized EXIF."""
    aperture = exif.get("aperture")
    shutter = exif.get("shutter")
    iso = exif.get("iso")
    aperture = float(aperture) if aperture else None
    shutter = float(shutter) if shutter else None
    iso = int(iso) if iso else None
    raw_focal = float(exif["focal"]) if exif.get("focal") else None
    exif_focal35 = float(exif["focal35"]) if exif.get("focal35") else None
    focal35 = recompute_focal_35mm(
        model=exif.get("model"), exif_focal35=exif_focal35,
        raw_focal=raw_focal, cfg=cfg,
    )
    return Optics(
        focal_35mm=focal35,
        aperture=aperture,
        shutter=shutter,
        iso=iso,
        ev=compute_ev(aperture, shutter, iso),
        focal_raw=raw_focal,
        exif_focal35=exif_focal35,
    )
