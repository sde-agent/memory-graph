"""Light metrics: solar elevation -> time-of-day band, plus warm/cool tint (spec.md §4).

Golden hour is the Q1 (nostalgic) / warmth signal, and gating "warm-Q2" in resonance.
Sun elevation uses the NOAA low-precision solar-position formulas (~0.01 deg) on the
photo's GPS position and UTC timestamp — fully deterministic and offline, so it is
exactly unit-testable (evaluation.md B.5). No GPS or no timestamp => elevation is None
and the band is "unknown" (we never guess the light).

Warm/cool tint comes from pixel statistics (metrics/pixels.py): mean(R) - mean(B).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from memory_graph.config import SunCfg

_J2000 = datetime(2000, 1, 1, 12, tzinfo=UTC)


def sun_elevation(lat: float, lon: float, when: datetime) -> float:
    """Sun elevation in degrees above the horizon (geometric, no refraction).

    `when` must be timezone-aware. NOAA low-precision formulas.
    """
    n = (when.astimezone(UTC) - _J2000).total_seconds() / 86400.0
    mean_lon = (280.460 + 0.9856474 * n) % 360
    anomaly = math.radians((357.528 + 0.9856003 * n) % 360)
    ecl_lon = math.radians(mean_lon + 1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly))
    obliquity = math.radians(23.439 - 0.0000004 * n)
    declination = math.asin(math.sin(obliquity) * math.sin(ecl_lon))
    right_asc = math.atan2(math.cos(obliquity) * math.sin(ecl_lon), math.cos(ecl_lon))
    gmst_deg = ((18.697374558 + 24.06570982441908 * n) % 24) * 15
    hour_angle = math.radians((gmst_deg + lon) % 360) - right_asc
    phi = math.radians(lat)
    sin_el = math.sin(phi) * math.sin(declination) + math.cos(phi) * math.cos(
        declination
    ) * math.cos(hour_angle)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))


def elevation_of(
    lat: float | None, lon: float | None, utc: datetime | None
) -> float | None:
    """Sun elevation, or None when GPS or timestamp is missing (not analyzed, not guessed)."""
    if lat is None or lon is None or utc is None:
        return None
    return sun_elevation(lat, lon, utc)


def sun_band(elevation: float | None, cfg: SunCfg) -> str:
    """night / blue hour / golden hour / daylight / high sun; 'unknown' without elevation."""
    if elevation is None:
        return "unknown"
    if elevation < cfg.night_below:
        return "night"
    if elevation < cfg.blue_hour_below:
        return "blue hour"
    if elevation < cfg.golden_hour_below:
        return "golden hour"
    if elevation >= cfg.high_sun_from:
        return "high sun"
    return "daylight"


@dataclass(frozen=True)
class Light:
    elevation: float | None
    band: str
    golden_hour: bool
    warm_cool: float | None     # from pixel stats; positive = warm


def extract_light(
    lat: float | None,
    lon: float | None,
    utc: datetime | None,
    cfg: SunCfg,
    warm_cool: float | None = None,
) -> Light:
    """Build the Light record from position, time, and (optionally) the warm-cool tint."""
    elev = elevation_of(lat, lon, utc)
    band = sun_band(elev, cfg)
    return Light(
        elevation=elev,
        band=band,
        golden_hour=(band == "golden hour"),
        warm_cool=warm_cool,
    )


def is_warm(warm_cool: float | None, band: str, cfg_warm_from: float) -> bool:
    """Whether the photo reads as warm-lit: golden hour, or a warm pixel tint.

    Used to gate "warm-Q2" in resonance (a cold ultra-wide is not 'warm aliveness').
    """
    if band == "golden hour":
        return True
    return warm_cool is not None and warm_cool >= cfg_warm_from
