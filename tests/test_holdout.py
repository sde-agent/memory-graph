"""Holdout evaluation: the pre-registered pass rule, its statistics, and its guards."""

from __future__ import annotations

import json

import pytest

from memory_graph.evaluation import holdout as H


def test_pass_rule_pass() -> None:
    # Lift at least 25 points with a positive lower bound passes.
    assert H.decide(0.30, 0.10) == "PASS"
    assert H.decide(0.25, 0.01) == "PASS"


def test_pass_rule_large_lift_but_bound_crosses_zero_is_inconclusive() -> None:
    assert H.decide(0.30, -0.01) == "INCONCLUSIVE"


def test_pass_rule_middle_band_is_inconclusive() -> None:
    assert H.decide(0.20, 0.05) == "INCONCLUSIVE"
    # Exactly at the fail boundary is not a fail: 10 points is the floor of inconclusive.
    assert H.decide(0.10, -0.05) == "INCONCLUSIVE"


def test_pass_rule_fail_below_ten_points() -> None:
    assert H.decide(0.05, 0.0) == "FAIL"
    assert H.decide(-0.10, -0.20) == "FAIL"


def test_lift_statistics_match_the_pre_registered_formula() -> None:
    st = H.lift_stats(k_top=38, n_top=75, k_ctrl=5, n_ctrl=25)
    assert st["p_top"] == pytest.approx(38 / 75)
    assert st["p_control"] == pytest.approx(0.20)
    assert st["lift"] == pytest.approx(38 / 75 - 0.20)
    expected_se = (((38 / 75) * (1 - 38 / 75)) / 75 + (0.2 * 0.8) / 25) ** 0.5
    assert st["se"] == pytest.approx(expected_se)
    assert st["lower"] == pytest.approx(st["lift"] - 1.96 * expected_se)


def test_wilson_interval_brackets_the_point_estimate() -> None:
    lo, hi = H.wilson(40, 100)
    assert lo < 0.40 < hi
    assert H.wilson(0, 0) == (0.0, 0.0)


def test_build_refuses_to_overwrite_an_existing_holdout(tmp_path, monkeypatch) -> None:
    # The holdout is one-shot: once a set exists it must not be re-selected.
    monkeypatch.setattr(H, "SET_FILE", tmp_path / "holdout_set.json")
    H.SET_FILE.write_text(json.dumps({"items": []}))
    with pytest.raises(FileExistsError):
        H.build_set(conn=None, cfg=None, now=0.0)


def test_score_reads_labels_and_applies_the_rule(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(H, "LABEL_FILE", tmp_path / "holdout.jsonl")
    items = [{"id": f"t{i}", "kind": "dispatch", "day": i} for i in range(75)]
    items += [{"id": f"c{i}", "kind": "control", "day": None} for i in range(25)]
    set_obj = {"items": items}
    labels = [{"id": f"t{i}", "worthy": i < 52, "quadrant": 1} for i in range(75)]
    labels += [{"id": f"c{i}", "worthy": i < 5, "quadrant": 1} for i in range(25)]
    H.LABEL_FILE.write_text("\n".join(json.dumps(x) for x in labels) + "\n")
    rows = {f"t{i}": {"dominant_quadrant": 1} for i in range(75)}
    rows |= {f"c{i}": {"dominant_quadrant": 2} for i in range(25)}

    rep = H.score(set_obj, rows)
    assert rep["labelled_top"] == 75 and rep["labelled_control"] == 25
    assert rep["p_top"] == pytest.approx(52 / 75)
    assert rep["p_control"] == pytest.approx(0.20)
    assert rep["decision"] == "PASS"            # lift ~ 50 points, lower bound > 0
    # Quadrant agreement counts yes answers only: 52 top-3 yes (feeling 1, engine 1) and 5 control
    # yes (feeling 1, engine 2, so no match). 52 agree out of 57 yes answers.
    assert rep["quadrant"]["agree"] == 52
    assert rep["quadrant"]["n"] == 57


def test_exclusion_covers_every_earlier_round(tmp_path, monkeypatch) -> None:
    # Round 2 must never draw a photo from round 1, from the tuning labels, or from any answer.
    monkeypatch.setattr(H, "LABELS_DIR", tmp_path)
    (tmp_path / "label_set.json").write_text(json.dumps({"ids": ["tune1"]}))
    (tmp_path / "holdout_set.json").write_text(json.dumps({"items": [{"id": "r1set"}]}))
    (tmp_path / "holdout.jsonl").write_text(json.dumps({"id": "r1ans"}) + "\n")
    (tmp_path / "holdout_set_r2.json").write_text(json.dumps({"items": [{"id": "r2set"}]}))
    (tmp_path / "holdout_r2.jsonl").write_text(json.dumps({"id": "r2ans"}) + "\n")
    used = H._excluded_ids()
    assert {"tune1", "r1set", "r1ans", "r2set", "r2ans"} <= used


def test_pooled_score_combines_rounds_and_counts_yes_quadrants(tmp_path) -> None:
    def rnd(prefix, top_yes, ctrl_yes):
        items = [{"id": f"{prefix}t{i}", "kind": "dispatch", "day": i} for i in range(75)]
        items += [{"id": f"{prefix}c{i}", "kind": "control", "day": None} for i in range(25)]
        labels = [{"id": f"{prefix}t{i}", "worthy": i < top_yes,
                   "quadrant": 1 if i < top_yes else None} for i in range(75)]
        labels += [{"id": f"{prefix}c{i}", "worthy": i < ctrl_yes,
                    "quadrant": 1 if i < ctrl_yes else None} for i in range(25)]
        path = tmp_path / f"{prefix}.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in labels) + "\n")
        return {"items": items}, path

    r1, p1 = rnd("a", 32, 6)
    r2, p2 = rnd("b", 36, 6)  # 32+36 = 68 engine yes; 6+6 = 12 control yes
    rows = {f"{p}t{i}": {"dominant_quadrant": 1} for p in "ab" for i in range(75)}
    rows |= {f"{p}c{i}": {"dominant_quadrant": 2} for p in "ab" for i in range(25)}
    rep = H.score_rounds([(r1, p1), (r2, p2)], rows)
    assert rep["labelled_top"] == 150 and rep["labelled_control"] == 50
    assert rep["p_top"] == pytest.approx(68 / 150)
    assert rep["p_control"] == pytest.approx(12 / 50)
    # Quadrant agreement only on yes answers: 68 yes on top-3 + 12 on controls = 80.
    assert rep["quadrant"]["n"] == 80


def test_score_incomplete_without_both_groups(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(H, "LABEL_FILE", tmp_path / "holdout.jsonl")
    H.LABEL_FILE.write_text(json.dumps({"id": "t0", "worthy": True, "quadrant": 1}) + "\n")
    set_obj = {"items": [{"id": "t0", "kind": "dispatch", "day": 0}]}
    rep = H.score(set_obj, {"t0": {"dominant_quadrant": 1}})
    assert rep["decision"] == "INCOMPLETE"


def _feel_rows(n_each, dom_by_q):
    rows = {}
    items = []
    for q, count in dom_by_q.items():
        for i in range(count):
            pid = f"q{q}_{i}"
            rows[pid] = {"dominant_quadrant": q}
            items.append({"id": pid, "kind": "dispatch", "day": None})
    return rows, items


def test_broad_picks_that_cover_the_engine_are_not_informative(tmp_path) -> None:
    # Engine's dominant quadrant is Quiet (4) for half the photos and Nostalgic (1) for the
    # other half. The owner picks {1, 4} everywhere: every pick hits, but a two-feeling pick
    # covering both common quadrants matches by chance, so the lift is zero.
    rows, items = _feel_rows(None, {4: 50, 1: 50})
    labels = [{"id": it["id"], "worthy": True, "quadrants": [1, 4]} for it in items]
    path = tmp_path / "feel.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in labels) + "\n")
    rep = H.score_feelings({"items": items}, path, rows)
    assert rep["observed"] == pytest.approx(1.0)
    assert rep["chance"] == pytest.approx(1.0)
    assert rep["lift"] == pytest.approx(0.0)
    assert rep["decision"] == "NOT INFORMATIVE"


def test_feelings_informative_when_hits_beat_chance(tmp_path) -> None:
    # One pick per photo, matching the engine's dominant quadrant every time. A single pick
    # has a chance rate of about one half here, so the lift is large and clear of zero.
    rows, items = _feel_rows(None, {4: 50, 1: 50})
    labels = [{"id": it["id"], "worthy": True,
               "quadrants": [rows[it["id"]]["dominant_quadrant"]]} for it in items]
    path = tmp_path / "feel.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in labels) + "\n")
    rep = H.score_feelings({"items": items}, path, rows)
    assert rep["observed"] == pytest.approx(1.0)
    assert rep["chance"] == pytest.approx(0.5)
    assert rep["lift"] == pytest.approx(0.5)
    assert rep["lower"] > 0
    assert rep["decision"] == "INFORMATIVE"


def test_padding_is_flagged_when_owner_picks_almost_everything(tmp_path) -> None:
    rows, items = _feel_rows(None, {1: 100})
    labels = [{"id": it["id"], "worthy": True, "quadrants": [1, 2, 3, 4, 5]} for it in items]
    path = tmp_path / "feel.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in labels) + "\n")
    rep = H.score_feelings({"items": items}, path, rows)
    assert rep["padding_flag"] is True
    assert rep["all_five_share"] == pytest.approx(1.0)
