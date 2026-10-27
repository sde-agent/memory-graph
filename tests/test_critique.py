"""Longitudinal critique: era aggregation + committed insights (P10)."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_graph.config import load_config
from memory_graph.narration.critique import build_insights, critique, summarize_eras


def eras():
    return load_config().eras


def _ts(year: int) -> float:
    return datetime(year, 6, 1, tzinfo=UTC).timestamp()


def _row(rid, year, q, resonance=1.0) -> dict:
    return {
        "id": rid,
        "is_canonical": 1,
        "timestamp": _ts(year),
        "dominant_quadrant": q,
        "resonance": resonance,
    }


def test_era_bucketing() -> None:
    rows = [_row("a", 2012, 1), _row("b", 2013, 1), _row("c", 2021, 3)]
    summaries = summarize_eras(rows, eras())
    labels = {s.era for s in summaries}
    assert "early 2010s" in labels
    assert "2020s" in labels


def test_shot_shares_sum_to_one() -> None:
    rows = [_row(f"p{i}", 2012, (i % 3) + 1) for i in range(9)]
    s = summarize_eras(rows, eras())[0]
    assert abs(sum(s.shot_shares.values()) - 1.0) < 1e-9


def test_top_shot_is_most_common_quadrant() -> None:
    rows = [_row("a", 2012, 1), _row("b", 2012, 1), _row("c", 2012, 3)]
    s = summarize_eras(rows, eras())[0]
    assert s.top_shot == 1


def test_resonance_weighting_shifts_top() -> None:
    # Q1 is shot most, but a single Q3 photo has overwhelming resonance.
    rows = [
        _row("a", 2012, 1, resonance=0.1),
        _row("b", 2012, 1, resonance=0.1),
        _row("c", 2012, 3, resonance=10.0),
    ]
    s = summarize_eras(rows, eras())[0]
    assert s.top_shot == 1          # shot Q1 most
    assert s.top_resonant == 3      # but drawn back to Q3


def test_insight_committed_direction_and_claim() -> None:
    rows = [_row("a", 2012, 1), _row("b", 2012, 1), _row("c", 2012, 3)]
    insights = build_insights(summarize_eras(rows, eras()))
    style = [i for i in insights if i.kind == "era_style"][0]
    assert style.direction == "higher"
    assert style.magnitude is not None
    assert "early 2010s" in style.claim
    assert style.era == "early 2010s"


def test_resonance_gap_insight_created_when_top_differs() -> None:
    rows = [
        _row("a", 2012, 1, resonance=0.1),
        _row("b", 2012, 1, resonance=0.1),
        _row("c", 2012, 3, resonance=10.0),
    ]
    insights = critique(rows, eras())
    # The per-era FLIP gap (id ends ':gap'); distinct from the always-on 'keep' node.
    gaps = [i for i in insights if i.id.endswith(":gap")]
    assert len(gaps) == 1
    assert gaps[0].resonance_angle is not None


def test_no_gap_insight_when_top_matches() -> None:
    rows = [_row("a", 2012, 1), _row("b", 2012, 1)]
    insights = critique(rows, eras())
    # No per-era FLIP gap when the top does not differ. (The whole-archive 'keep' node,
    # id 'ins:resonance:keep', always fires and is not a per-era flip gap.)
    assert not [i for i in insights if i.id.endswith(":gap")]


def test_resonance_keep_node_always_fires() -> None:
    # "Which photos do I keep more" must always be answerable: the whole-archive
    # resonance-preference node fires even when no era's top flips.
    rows = [_row("a", 2012, 1), _row("b", 2012, 1), _row("c", 2021, 3)]
    insights = critique(rows, eras())
    keep = [i for i in insights if i.id == "ins:resonance:keep"]
    assert keep
    assert "keep" in keep[0].tags and keep[0].era is None


def _row_rich(rid, year, q, *, focal=None, faces=0, known=False, names=None,
              resonance=1.0) -> dict:
    import json
    return {
        "id": rid, "is_canonical": 1, "timestamp": _ts(year), "dominant_quadrant": q,
        "resonance": resonance, "focal_length_35mm": focal, "face_count": faces,
        "known_person": 1 if known else 0, "golden_hour": 0,
        "person_names": json.dumps(names) if names else "[]",
    }


def test_trend_people_centrality() -> None:
    # Known-person share rises sharply from the first era to the last.
    rows = [_row_rich(f"a{i}", 2012, 1, known=False) for i in range(10)]
    rows += [_row_rich(f"z{i}", 2022, 1, known=(i < 8)) for i in range(10)]  # 80% known
    insights = critique(rows, eras())
    trend = [i for i in insights if i.kind == "trend_people"]
    assert trend
    assert trend[0].direction == "higher"
    assert "recognized loved one" in trend[0].claim


def test_trend_dominant_person_names_only_above_bar() -> None:
    rows = [_row_rich(f"z{i}", 2022, 3, faces=1, known=True, names=["Ada"])
            for i in range(25)]
    rows += [_row_rich(f"a{i}", 2012, 1) for i in range(5)]
    insights = critique(rows, eras())
    person = [i for i in insights if i.kind == "trend_person"]
    assert person
    assert "Ada" in person[0].claim
    assert person[0].allowed_entities == ("Ada",)   # the name is a permitted entity


def test_trend_focal_drift() -> None:
    rows = [_row_rich(f"a{i}", 2012, 1, focal=28.0) for i in range(10)]
    rows += [_row_rich(f"z{i}", 2022, 1, focal=85.0) for i in range(10)]
    insights = critique(rows, eras())
    focal = [i for i in insights if i.kind == "trend_focal"]
    assert focal
    assert focal[0].direction == "higher"   # 28 -> 85 is longer


def test_trend_person_json_parse_works() -> None:
    # Regression: json must be imported (a NameError was silently swallowed, dropping
    # every people insight). A populated names column must produce a person trend.
    # Needs >=2 eras (trends compare first vs last).
    rows = [_row_rich(f"a{i}", 2012, 1) for i in range(5)]
    rows += [_row_rich(f"z{i}", 2022, 3, faces=1, known=True, names=["Ada", "Grace"])
             for i in range(30)]
    insights = critique(rows, eras())
    assert any(i.kind == "trend_person" for i in insights)


def test_undated_and_out_of_range_excluded() -> None:
    rows = [
        _row("a", 2012, 1),
        {"id": "nodate", "is_canonical": 1, "timestamp": None, "dominant_quadrant": 2,
         "resonance": 1.0},
        _row("old", 1995, 1),   # before any configured era
    ]
    summaries = summarize_eras(rows, eras())
    total = sum(s.count for s in summaries)
    assert total == 1   # only the 2012 photo counted
