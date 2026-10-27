"""Groundedness verification — the critical fluent-but-wrong guardrail (P8)."""

from __future__ import annotations

from memory_graph.narration.verify import verify_narration


def _v(prose, numbers=(), names=(), direction=None):
    return verify_narration(
        prose, allowed_numbers=list(numbers), allowed_names=list(names),
        committed_direction=direction,
    )


def test_grounded_prose_passes() -> None:
    r = _v(
        "You stood close with a 50mm lens, letting the background soften. "
        "It feels like a quiet, preserved moment.",
        numbers=[50.0, 1.8],
    )
    assert r.ok, r.reasons


def test_invented_number_rejected() -> None:
    # The facts say 50mm; the prose claims 85mm -> reject.
    r = _v("Shot at 85mm, isolating the face.", numbers=[50.0])
    assert not r.ok
    assert any("85" in reason for reason in r.reasons)


def test_rounded_number_accepted() -> None:
    # 49.6mm fact, prose says "50mm" -> within tolerance, accepted.
    r = _v("A 50mm frame.", numbers=[49.6])
    assert r.ok, r.reasons


def test_shutter_fraction_accepted() -> None:
    # Fact shutter 0.008s, prose writes "1/125" -> accepted via the 1/shutter form.
    r = _v("Frozen at 1/125 of a second.", numbers=[0.008])
    assert r.ok, r.reasons


def test_invented_name_rejected() -> None:
    r = _v("You leaned in toward Grace, tenderly.", names=["Ada"])
    assert not r.ok
    assert any("Grace" in reason for reason in r.reasons)


def test_allowed_name_accepted() -> None:
    r = _v("You leaned in toward Ada, tenderly.", names=["Ada"])
    assert r.ok, r.reasons


def test_ordinary_capitalized_words_not_flagged() -> None:
    # Sentence-start and quadrant words are capitalized but are not names.
    r = _v("Nostalgic and reflective. You preserved a quiet moment. Notice the soft light.")
    assert r.ok, r.reasons


def test_direction_contradiction_rejected() -> None:
    # The insight committed 'higher'; the prose says 'lower' -> reject.
    r = _v("Over the decade you shot lower and lower.", direction="higher")
    assert not r.ok
    assert any("direction" in reason for reason in r.reasons)


def test_direction_consistent_accepted() -> None:
    r = _v("Over the decade this rose and grew more central to your work.", direction="higher")
    assert r.ok, r.reasons
