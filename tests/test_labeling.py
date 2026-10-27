"""Label store: resumable append, labeled-key tracking (labeling UI)."""

from __future__ import annotations

from memory_graph.labeling.store import LabelStore


def _store(tmp_path, monkeypatch, kind="base_rate") -> LabelStore:
    import memory_graph.labeling.store as store_mod

    monkeypatch.setattr(store_mod, "LABELS_DIR", tmp_path / "labels")
    return LabelStore(kind)


def test_append_and_count(tmp_path, monkeypatch) -> None:
    s = _store(tmp_path, monkeypatch)
    assert s.count() == 0
    s.append({"id": "hdd:a", "worthy": True})
    s.append({"id": "hdd:b", "worthy": False})
    assert s.count() == 2


def test_labeled_keys_dedup(tmp_path, monkeypatch) -> None:
    s = _store(tmp_path, monkeypatch)
    s.append({"id": "hdd:a", "worthy": True})
    s.append({"id": "hdd:b", "worthy": True})
    assert s.labeled_keys() == {"hdd:a", "hdd:b"}


def test_resumable_across_instances(tmp_path, monkeypatch) -> None:
    s1 = _store(tmp_path, monkeypatch)
    s1.append({"id": "hdd:a", "worthy": True})
    # A fresh store over the same dir sees prior labels (resumable session).
    s2 = _store(tmp_path, monkeypatch)
    assert s2.labeled_keys() == {"hdd:a"}


def test_corrupt_line_skipped(tmp_path, monkeypatch) -> None:
    s = _store(tmp_path, monkeypatch)
    s.append({"id": "hdd:a", "worthy": True})
    # Simulate a half-written line from an interrupted write.
    with s.path.open("a") as fh:
        fh.write('{"id": "hdd:b", "wor')
    assert s.labeled_keys() == {"hdd:a"}   # the cut line is ignored, not fatal
    assert s.count() == 1


def test_pairwise_record(tmp_path, monkeypatch) -> None:
    s = _store(tmp_path, monkeypatch, kind="pairwise")
    s.append({"id": "hdd:a|hdd:b", "a": "hdd:a", "b": "hdd:b", "winner": "a"})
    rows = s.all()
    assert rows[0]["winner"] == "a"
