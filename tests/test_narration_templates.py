"""Deterministic narration templates (P8) — exact strings from DB rows."""

from __future__ import annotations

import json

from memory_graph.narration.templates import photo_facts, photo_sentence

# A fixed "now" so age is deterministic: 2024-01-01 UTC.
NOW = 1_704_067_200.0
# A capture ~12 years before NOW.
TS_12Y = NOW - 12 * 365.25 * 24 * 3600


def _row(**kw) -> dict:
    base = {
        "focal_length_35mm": 50.0,
        "aperture": 1.8,
        "shutter_speed": 1 / 125,
        "iso": 200,
        "golden_hour": 0,
        "timestamp": TS_12Y,
        "dominant_quadrant": 1,
        "person_names": None,
    }
    base.update(kw)
    return base


def test_facts_extracted() -> None:
    f = photo_facts(_row(), now=NOW)
    assert f.focal_35mm == 50.0
    assert f.aperture == 1.8
    assert f.age_years == 12
    assert f.quadrant_name == "Nostalgic & Reflective"


def test_sentence_renders_settings() -> None:
    s = photo_sentence(photo_facts(_row(golden_hour=1), now=NOW))
    assert "50mm" in s
    assert "f/1.8" in s
    assert "1/125s" in s
    assert "golden hour" in s
    assert "12 years ago" in s
    assert s.startswith("Nostalgic & Reflective")


def test_sentence_with_people() -> None:
    s = photo_sentence(photo_facts(_row(person_names=json.dumps(["Ada"])), now=NOW))
    assert "With Ada." in s


def test_long_exposure_shutter_format() -> None:
    s = photo_sentence(photo_facts(_row(shutter_speed=4.0), now=NOW))
    assert "4.0s" in s


def test_missing_fields_omitted() -> None:
    s = photo_sentence(photo_facts(
        {"focal_length_35mm": None, "aperture": None, "shutter_speed": None,
         "iso": None, "golden_hour": 0, "timestamp": None, "dominant_quadrant": None,
         "person_names": None},
        now=NOW,
    ))
    assert s == ""  # nothing known => empty (the sentence never fabricates)


def test_numbers_for_verification() -> None:
    nums = photo_facts(_row(), now=NOW).numbers()
    assert 50.0 in nums and 1.8 in nums and 200.0 in nums and 12.0 in nums
