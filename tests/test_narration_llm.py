"""Grounded LLM narration orchestration + honest fallback (P8)."""

from __future__ import annotations

from memory_graph.narration.llm import CallableBackend, narrate_photo
from memory_graph.narration.templates import PhotoFacts

FACTS = PhotoFacts(
    focal_35mm=50.0, aperture=1.8, shutter=1 / 125, iso=200, golden_hour=True,
    age_years=12, quadrant=1, quadrant_name="Nostalgic & Reflective", people=("Ada",),
)


def test_grounded_llm_output_is_used() -> None:
    good = lambda _p: (  # noqa: E731
        "You lingered close with Ada in warm light, letting the background dissolve. "
        "It reads as tenderness you wanted to keep."
    )
    n = narrate_photo(FACTS, CallableBackend(good))
    assert n.grounded
    assert not n.fell_back
    assert "Ada" in n.text


def test_invented_fact_triggers_regeneration_then_falls_back() -> None:
    # Always returns an ungrounded claim (85mm, not in facts) -> fails twice -> fallback.
    bad = lambda _p: "Shot at 85mm, a world away."  # noqa: E731
    n = narrate_photo(FACTS, CallableBackend(bad), max_regenerations=1)
    assert not n.grounded
    assert n.fell_back
    assert n.notice is not None                     # honest message present
    assert "50mm" in n.text                          # fell back to the template facts


def test_regeneration_succeeds_on_second_try() -> None:
    calls = {"n": 0}

    def flaky(_p):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Shot at 85mm."          # bad first attempt
        return "You stayed close at 50mm, warmly."  # good retry

    n = narrate_photo(FACTS, CallableBackend(flaky), max_regenerations=1)
    assert n.grounded
    assert not n.fell_back
    assert calls["n"] == 2


def test_no_backend_falls_back_quietly() -> None:
    n = narrate_photo(FACTS, None)
    assert n.fell_back
    assert n.notice is None            # no backend configured => quiet template, no alarm
    assert "50mm" in n.text


def test_backend_returning_none_falls_back() -> None:
    n = narrate_photo(FACTS, CallableBackend(lambda _p: None), max_regenerations=1)
    assert n.fell_back
    assert n.notice is not None


def test_invented_name_is_caught() -> None:
    # The LLM names someone not in the facts -> ungrounded -> fallback.
    bad = lambda _p: "You and Grace shared a quiet moment."  # noqa: E731
    n = narrate_photo(FACTS, CallableBackend(bad), max_regenerations=0)
    assert n.fell_back
