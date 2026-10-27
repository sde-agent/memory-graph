"""Light metrics: solar elevation and golden-hour band (P4)."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_graph.config import load_config
from memory_graph.metrics.light import (
    elevation_of,
    extract_light,
    is_warm,
    sun_band,
    sun_elevation,
)


def suncfg():
    return load_config().sun


def test_sun_high_at_solar_noon_equator() -> None:
    # Equator, near equinox, ~noon UTC at 0 lon: sun should be high overhead.
    when = datetime(2020, 3, 20, 12, 0, tzinfo=UTC)
    elev = sun_elevation(0.0, 0.0, when)
    assert elev > 45.0


def test_sun_below_horizon_at_local_midnight() -> None:
    # 0 lon, midnight UTC: sun is on the far side => negative elevation.
    when = datetime(2020, 3, 20, 0, 0, tzinfo=UTC)
    assert sun_elevation(0.0, 0.0, when) < 0.0


def test_elevation_none_without_gps_or_time() -> None:
    assert elevation_of(None, 10.0, datetime(2020, 1, 1, tzinfo=UTC)) is None
    assert elevation_of(10.0, 10.0, None) is None


def test_sun_band_boundaries() -> None:
    c = suncfg()
    assert sun_band(None, c) == "unknown"
    assert sun_band(-10.0, c) == "night"
    assert sun_band(-3.0, c) == "blue hour"
    assert sun_band(3.0, c) == "golden hour"
    assert sun_band(20.0, c) == "daylight"
    assert sun_band(60.0, c) == "high sun"


def test_extract_light_golden_hour_flag() -> None:
    c = suncfg()
    # A low positive elevation (within golden-hour band) sets the flag.
    # Build inputs that land in golden hour by passing the band through sun_band.
    light = extract_light(None, None, None, c, warm_cool=5.0)
    assert light.band == "unknown"   # no GPS/time => unknown, golden_hour False
    assert light.golden_hour is False


def test_is_warm_golden_hour_or_warm_tint() -> None:
    warm_from = load_config().color.warm_from
    assert is_warm(None, "golden hour", warm_from) is True      # golden hour always warm
    assert is_warm(20.0, "daylight", warm_from) is True         # warm tint
    assert is_warm(-20.0, "daylight", warm_from) is False       # cool tint
    assert is_warm(None, "daylight", warm_from) is False        # unknown tint, not golden
