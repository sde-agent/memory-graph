"""Library adapter: derivatives-only, the no-master guardrail, skip-and-count (P1).

Uses fake photo objects so no real Photos library (or Full Disk Access) is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from PIL import Image

from memory_graph.config import load_config
from memory_graph.contracts import Source
from memory_graph.ingest.library import (
    LibraryMasterAccessError,
    LibraryStats,
    assert_no_master_access,
    candidate_from_photo,
    select,
)


@pytest.fixture()
def cfg():
    return load_config()


def _write_jpeg(path, size=(400, 300), color=(80, 160, 200)) -> None:
    Image.new("RGB", size, color).save(path, "JPEG")


class FakeExif:
    camera_make = "FakeCam"
    camera_model = "FM-1"
    lens_model = "50mm"
    focal_length = 50.0
    fnumber = 1.8
    exposure_time = 0.004
    iso = 200


class FakePhoto:
    """Minimal osxphotos-like object. `path` raises unless explicitly allowed, mimicking
    that the adapter must never read a master."""

    def __init__(self, uuid, derivatives, *, date=None, favorite=False, ismovie=False,
                 persons=None, allow_path=False):
        self.uuid = uuid
        self.path_derivatives = derivatives
        self.date = date or datetime(2014, 6, 1, 12, 0, tzinfo=UTC)
        self.favorite = favorite
        self.ismovie = ismovie
        self.persons = persons or []
        self.exif_info = FakeExif()
        self.latitude = 40.0
        self.longitude = -74.0
        self.width = 400
        self.height = 300
        self.albums = []
        self.keywords = []
        self.screenshot = False
        self.face_info = []
        self._focal_35mm = 75.0
        self._allow_path = allow_path

    @property
    def path(self):
        if not self._allow_path:
            raise AssertionError("adapter accessed photo.path — forbidden")
        return "/Masters/should-not-be-read.cr2"


def test_candidate_built_from_local_derivative(tmp_path, cfg) -> None:
    deriv = tmp_path / "deriv.jpg"
    _write_jpeg(deriv)
    photo = FakePhoto("UUID-1", [str(deriv)], favorite=True, persons=["Ada"])
    stats = LibraryStats()
    cand = candidate_from_photo(photo, cfg, stats)
    assert cand is not None
    assert cand.id == "lib:UUID-1"
    assert cand.source == Source.LIBRARY
    assert cand.original_ref == "UUID-1"          # the uuid, NOT a path
    assert cand.library_meta["favorite"] is True
    assert cand.library_meta["persons"] == ["Ada"]
    assert cand.exif["focal35"] == 75.0
    assert cand.exif["aperture"] == 1.8
    assert "phash" in cand.exif
    assert max(cand.thumb.size) <= cfg.ingest.thumb_long_edge


def test_icloud_only_photo_skipped_and_counted(tmp_path, cfg) -> None:
    # Derivative path listed but not present on disk => iCloud-only, nothing local to read.
    photo = FakePhoto("UUID-2", ["/not/on/disk/deriv.jpg"])
    stats = LibraryStats()
    cand = candidate_from_photo(photo, cfg, stats)
    assert cand is None
    assert stats.no_local_derivative == 1


def test_largest_derivative_chosen(tmp_path, cfg) -> None:
    small = tmp_path / "s.jpg"
    large = tmp_path / "l.jpg"
    _write_jpeg(small, size=(100, 75))
    _write_jpeg(large, size=(800, 600))
    photo = FakePhoto("UUID-3", [str(small), str(large)])
    cand = candidate_from_photo(photo, cfg, LibraryStats())
    assert cand is not None  # built from the larger derivative without error


def test_movie_and_undated_are_skipped(tmp_path, cfg) -> None:
    deriv = tmp_path / "d.jpg"
    _write_jpeg(deriv)
    stats = LibraryStats()
    assert candidate_from_photo(
        FakePhoto("m", [str(deriv)], ismovie=True), cfg, stats
    ) is None
    assert stats.movies_skipped == 1

    undated = FakePhoto("u", [str(deriv)])
    undated.date = None
    assert candidate_from_photo(undated, cfg, stats) is None
    assert stats.no_date == 1


def test_adapter_never_reads_master_path(tmp_path, cfg) -> None:
    # The FakePhoto asserts if .path is read; building a candidate must not trip it.
    deriv = tmp_path / "d.jpg"
    _write_jpeg(deriv)
    photo = FakePhoto("UUID-4", [str(deriv)])
    candidate_from_photo(photo, cfg, LibraryStats())  # must not raise


def test_no_master_guard_raises_on_path_access(tmp_path) -> None:
    photo = FakePhoto("UUID-5", [], allow_path=True)
    guarded = assert_no_master_access(photo)
    # Non-path attributes pass through...
    assert guarded.uuid == "UUID-5"
    # ...but reading the master raises, proving the guard works.
    with pytest.raises(LibraryMasterAccessError):
        _ = guarded.path


def test_select_streams_and_counts(tmp_path, cfg) -> None:
    deriv = tmp_path / "d.jpg"
    _write_jpeg(deriv)
    photos = [
        FakePhoto("a", [str(deriv)]),
        FakePhoto("b", ["/missing.jpg"]),   # skipped
        FakePhoto("c", [str(deriv)]),
    ]
    stats = LibraryStats()
    out = list(select(photos, cfg, stats))
    assert len(out) == 2
    assert stats.total == 3
    assert stats.yielded == 2
    assert stats.no_local_derivative == 1
