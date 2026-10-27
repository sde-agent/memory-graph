"""The labelling page's logic: blinding, validation, saving, and file safety."""

from __future__ import annotations

import json

import pytest

from memory_graph.evaluation import holdout as H
from memory_graph.evaluation import holdout_app as A


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "LABEL_FILE", tmp_path / "holdout.jsonl")
    monkeypatch.setattr(H, "THUMB_DIR", tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir()
    items = [{"id": "hdd:a", "kind": "dispatch", "day": 0},
             {"id": "lib:b", "kind": "control", "day": None}]
    for it in items:
        (tmp_path / "thumbs" / A.thumb_name(it["id"])).write_bytes(b"jpg")
    return {"items": items}


def test_page_does_not_reveal_dispatch_or_control() -> None:
    # Blinding: the labeller must never see which photos came from the engine.
    low = A.PAGE.lower()
    assert "dispatch" not in low
    assert "control" not in low


def test_state_is_blind_and_points_at_the_next_photo(env) -> None:
    s = A.state(env)
    text = json.dumps(s)
    assert "dispatch" not in text and "control" not in text and "kind" not in text
    assert s["total"] == 2 and s["done"] == 0
    assert s["next"]["image"] == "/thumb/" + A.thumb_name("hdd:a")


def test_record_saves_a_valid_answer(env) -> None:
    ok, msg = A.record(env, {"id": "hdd:a", "worthy": True, "quadrant": 3})
    assert ok and msg == "saved"
    rec = json.loads(H.LABEL_FILE.read_text().splitlines()[0])
    assert rec == {"id": "hdd:a", "worthy": True, "quadrant": 3}


def test_record_rejects_bad_input(env) -> None:
    assert A.record(env, {"id": "not-in-set", "worthy": True, "quadrant": 1})[0] is False
    assert A.record(env, {"id": "hdd:a", "worthy": None, "quadrant": 1})[0] is False
    assert A.record(env, {"id": "hdd:a", "worthy": True, "quadrant": 9})[0] is False
    assert not H.LABEL_FILE.exists()


def test_no_saves_at_once_without_a_feeling(env) -> None:
    # Part V: a No has no feeling step, so it saves with quadrant null.
    ok, _ = A.record(env, {"id": "hdd:a", "worthy": False, "quadrant": None})
    assert ok
    rec = json.loads(H.LABEL_FILE.read_text().splitlines()[0])
    assert rec == {"id": "hdd:a", "worthy": False, "quadrant": None}


def test_yes_still_needs_a_feeling(env) -> None:
    ok, msg = A.record(env, {"id": "hdd:a", "worthy": True, "quadrant": None})
    assert not ok and "feeling" in msg


def test_each_photo_is_answered_once(env) -> None:
    assert A.record(env, {"id": "hdd:a", "worthy": True, "quadrant": 1})[0]
    ok, msg = A.record(env, {"id": "hdd:a", "worthy": False, "quadrant": 2})
    assert not ok and msg == "already answered"
    assert len(H.LABEL_FILE.read_text().splitlines()) == 1


def test_progress_moves_after_an_answer(env) -> None:
    A.record(env, {"id": "hdd:a", "worthy": True, "quadrant": 1})
    s = A.state(env)
    assert s["done"] == 1
    assert s["next"]["image"] == "/thumb/" + A.thumb_name("lib:b")


def test_thumb_names_are_safe_to_serve() -> None:
    assert A.thumb_name("hdd:abc") == "hdd_abc.jpg"
    assert A._SAFE_NAME.match("hdd_abc.jpg")
    assert not A._SAFE_NAME.match("../secrets.jpg")
    assert not A._SAFE_NAME.match("hdd_abc.png")


def test_multi_accepts_several_feelings_and_saves_them(env) -> None:
    ok, msg = A.record(env, {"id": "hdd:a", "worthy": True, "quadrants": [4, 1]},
                       multi=True)
    assert ok and msg == "saved"
    rec = json.loads(H.LABEL_FILE.read_text().splitlines()[0])
    assert rec == {"id": "hdd:a", "worthy": True, "quadrants": [1, 4]}


def test_multi_needs_yes_or_no_and_at_least_one_feeling(env) -> None:
    assert A.record(env, {"id": "hdd:a", "worthy": None, "quadrants": [1]},
                    multi=True)[0] is False
    assert A.record(env, {"id": "hdd:a", "worthy": False, "quadrants": []},
                    multi=True)[0] is False
    assert A.record(env, {"id": "hdd:a", "worthy": False, "quadrants": [1, 1]},
                    multi=True)[0] is False
    assert not H.LABEL_FILE.exists()


def test_multi_keeps_a_no_with_its_feelings(env) -> None:
    # Part W asks feelings for every photo, Yes or No.
    ok, _ = A.record(env, {"id": "hdd:a", "worthy": False, "quadrants": [3]}, multi=True)
    assert ok
    rec = json.loads(H.LABEL_FILE.read_text().splitlines()[0])
    assert rec == {"id": "hdd:a", "worthy": False, "quadrants": [3]}
