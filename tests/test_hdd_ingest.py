"""HDD adapter: read-only enforcement, scandir walk, single-pass decode, skip-and-count.

These tests write real JPEGs into a temp tree and run the adapter against it. Because a
temp dir cannot be mounted read-only, the read-only guard is monkeypatched to a no-op
for the walk tests; the guard itself is tested directly in test_require_read_only.
"""

from __future__ import annotations

import os

import pytest
from PIL import Image

from memory_graph.config import load_config
from memory_graph.contracts import Source
from memory_graph.ingest import hdd


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def no_ro_guard(monkeypatch):
    """Neutralize the read-only mount check so we can test the walk on a writable temp dir."""
    monkeypatch.setattr(hdd, "require_read_only", lambda root: None)


def _jpeg(path, size=(640, 480), color=(90, 140, 190)) -> None:
    Image.new("RGB", size, color).save(path, "JPEG")


def test_require_read_only_rejects_writable_dir(tmp_path) -> None:
    # A normal temp dir is read-write, so the guard must refuse it.
    with pytest.raises(hdd.HddNotReadOnly):
        hdd.require_read_only(tmp_path)


def test_require_read_only_rejects_missing_path(tmp_path) -> None:
    with pytest.raises(hdd.HddNotReadOnly):
        hdd.require_read_only(tmp_path / "does-not-exist")


def test_scan_walks_recursively_and_filters_extensions(tmp_path, cfg, no_ro_guard) -> None:
    (tmp_path / "2012").mkdir()
    _jpeg(tmp_path / "2012" / "a.jpg")
    _jpeg(tmp_path / "2012" / "b.jpeg")
    (tmp_path / "notes.txt").write_text("not a photo")
    (tmp_path / "previews").mkdir()
    _jpeg(tmp_path / "previews" / "skip_me.jpg")  # inside a skipped derivatives dir

    cands, stats = hdd.stream_hdd_with_stats(str(tmp_path), cfg)
    refs = {os.path.basename(c.original_ref) for c in cands}
    assert refs == {"a.jpg", "b.jpeg"}          # txt ignored, previews/ skipped
    assert all(c.source == Source.HDD for c in cands)
    assert stats.yielded == 2


def test_candidate_has_thumbnail_and_phash(tmp_path, cfg, no_ro_guard) -> None:
    _jpeg(tmp_path / "x.jpg", size=(1600, 1200))
    cands, _ = hdd.stream_hdd_with_stats(str(tmp_path), cfg)
    c = cands[0]
    assert c.thumb is not None
    assert max(c.thumb.size) <= cfg.ingest.thumb_long_edge
    assert "phash" in c.exif
    assert c.id.startswith("hdd:")


def test_corrupt_file_skipped_and_counted(tmp_path, cfg, no_ro_guard) -> None:
    _jpeg(tmp_path / "good.jpg")
    # A file with a photo extension but garbage bytes: must be skipped, not fatal.
    (tmp_path / "bad.jpg").write_bytes(b"this is not a jpeg")
    cands, stats = hdd.stream_hdd_with_stats(str(tmp_path), cfg)
    assert stats.yielded == 1
    assert stats.skipped_decode == 1
    assert len(stats.errors) == 1


def test_stable_id_is_deterministic(tmp_path, cfg, no_ro_guard) -> None:
    _jpeg(tmp_path / "p.jpg")
    first, _ = hdd.stream_hdd_with_stats(str(tmp_path), cfg)
    second, _ = hdd.stream_hdd_with_stats(str(tmp_path), cfg)
    assert first[0].id == second[0].id  # re-run yields the same id (upsert, not duplicate)


def test_release_thumb_frees_memory(tmp_path, cfg, no_ro_guard) -> None:
    _jpeg(tmp_path / "p.jpg")
    cands, _ = hdd.stream_hdd_with_stats(str(tmp_path), cfg)
    c = cands[0]
    assert c.thumb is not None
    c.release_thumb()
    assert c.thumb is None
