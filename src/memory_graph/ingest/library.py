"""macOS Photos Library ingest adapter (spec.md §1) — WITH THE NO-MASTER GUARDRAIL.

Reads the Photos library through osxphotos, READ-ONLY, and — critically — NEVER accesses
`photo.path` or a master file. Touching a master whose pixels live only in iCloud would
trigger a blocking cloud download; the whole product forbids that. Pixels come solely
from the largest LOCAL derivative in `photo.path_derivatives`. Items with no local
derivative (iCloud-only, not downloaded) are skipped and counted.

The guardrail is enforced two ways:
  1. `_local_derivative()` reads ONLY `path_derivatives`, never `.path`.
  2. `assert_no_master_access()` wraps a photo so any `.path` access raises, used by the
     test suite (and optionally at runtime) to prove the rule holds.

Selection logic (`candidate_from_photo`, `select`) is pure over osxphotos-like objects,
so it is unit-testable with fakes and never needs a real library.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from PIL import Image

from memory_graph.config import Config
from memory_graph.contracts import Candidate, Source
from memory_graph.ingest.imaging import phash, to_thumb


class LibraryMasterAccessError(RuntimeError):
    """Raised if code attempts to access a Photos master path — a forbidden iCloud trigger."""


@dataclass
class LibraryStats:
    total: int = 0
    movies_skipped: int = 0
    no_date: int = 0
    no_local_derivative: int = 0   # iCloud-only; nothing local to read without a download
    decode_failed: int = 0
    yielded: int = 0
    errors: list[str] = field(default_factory=list)


def _local_derivative(photo: Any) -> str | None:
    """The largest LOCAL derivative file for a photo, or None. NEVER reads `photo.path`.

    `path_derivatives` is Photos' list of pre-rendered local JPEGs (thumbnails/previews).
    We pick the largest that actually exists on disk. The master (`photo.path`) is never
    consulted — that is the whole point of this adapter.
    """
    derivatives = getattr(photo, "path_derivatives", None) or []
    existing = [d for d in derivatives if d and os.path.isfile(d)]
    if not existing:
        return None
    return max(existing, key=os.path.getsize)


def _library_meta(photo: Any) -> dict[str, Any]:
    """Collect the Library's own knowledge: favorite, persons, albums, keywords, faces.

    These feed resonance (favorite, known people) and the identity gallery later. Only
    metadata is read here — no pixels, no master.
    """
    persons = list(getattr(photo, "persons", None) or [])
    # Per-face attributes from the Photos Library's OWN face recognition. Photos has
    # already named faces to high quality, so `name` here is the authoritative identity
    # for Library photos (no embedding needed); `has_smile` and `yaw` feed candidness.
    faces = []
    for f in getattr(photo, "face_info", None) or []:
        name = getattr(f, "name", None)
        faces.append(
            {
                "name": name if name and name != "_UNKNOWN_" else None,
                "quality": getattr(f, "quality", None),
                "center_x": getattr(f, "center_x", None),
                "center_y": getattr(f, "center_y", None),
                "size": getattr(f, "size", None),
                "has_smile": bool(getattr(f, "has_smile", False)),
                "yaw": getattr(f, "yaw", None),
                "roll": getattr(f, "roll", None),
            }
        )
    return {
        "favorite": bool(getattr(photo, "favorite", False)),
        "persons": [p for p in persons if p and p != "_UNKNOWN_"],
        "albums": list(getattr(photo, "albums", None) or []),
        "keywords": list(getattr(photo, "keywords", None) or []),
        "screenshot": bool(getattr(photo, "screenshot", False)),
        "faces": faces,
        "focal_35mm": getattr(photo, "_focal_35mm", None),  # injected from the SQL side-read
    }


def candidate_from_photo(photo: Any, cfg: Config, stats: LibraryStats) -> Candidate | None:
    """Build a Candidate from an osxphotos-like photo, or None if it must be skipped.

    Pure over the photo object: no global state, no real library needed. Decodes the
    local derivative only.
    """
    if getattr(photo, "ismovie", False):
        stats.movies_skipped += 1
        return None
    if not getattr(photo, "date", None):
        stats.no_date += 1
        return None

    deriv = _local_derivative(photo)
    if deriv is None:
        stats.no_local_derivative += 1
        return None

    try:
        with open(deriv, "rb") as fh:
            img = Image.open(io.BytesIO(fh.read()))
        thumb = to_thumb(img, cfg.ingest.thumb_long_edge)
        img.close()
    except Exception:
        stats.decode_failed += 1
        stats.errors.append(getattr(photo, "uuid", "?"))
        return None
    # Stash the resolved local derivative path so a sampled candidate can be re-decoded
    # later WITHOUT re-opening PhotosDB (and without ever touching the master).
    _deriv_path = deriv

    date = photo.date
    ts = date.astimezone(UTC).timestamp() if date else None
    exif = {
        "make": getattr(getattr(photo, "exif_info", None), "camera_make", None),
        "model": getattr(getattr(photo, "exif_info", None), "camera_model", None),
        "lens": getattr(getattr(photo, "exif_info", None), "lens_model", None),
        "focal": getattr(getattr(photo, "exif_info", None), "focal_length", None),
        "focal35": getattr(photo, "_focal_35mm", None),
        "aperture": getattr(getattr(photo, "exif_info", None), "fnumber", None),
        "shutter": getattr(getattr(photo, "exif_info", None), "exposure_time", None),
        "iso": getattr(getattr(photo, "exif_info", None), "iso", None),
        "lat": getattr(photo, "latitude", None),
        "lon": getattr(photo, "longitude", None),
        "width": getattr(photo, "width", None),
        "height": getattr(photo, "height", None),
        "phash": phash(thumb),
        "deriv_path": _deriv_path,   # local derivative, for cheap re-decode (never a master)
    }
    return Candidate(
        id=f"lib:{photo.uuid}",
        source=Source.LIBRARY,
        original_ref=photo.uuid,   # the uuid, NOT a master path
        timestamp=ts,
        thumb=thumb,
        exif=exif,
        library_meta=_library_meta(photo),
    )


def select(
    photos: list[Any], cfg: Config, stats: LibraryStats | None = None
) -> Iterator[Candidate]:
    """Yield Candidates for a list of osxphotos-like photos. Pure; testable with fakes."""
    stats = stats if stats is not None else LibraryStats()
    stats.total = len(photos)
    for p in photos:
        cand = candidate_from_photo(p, cfg, stats)
        if cand is not None:
            stats.yielded += 1
            yield cand


def build_pool_parallel(cfg: Config, keep_fn, *, workers: int = 8, dbfile: str | None = None):
    """Decode + cheap-gate every Library photo in parallel, keeping survivors.

    The Library is large (tens of thousands); serial candidate-build dominates the
    pool-building time. Decoding each local derivative is CPU-bound and releases the GIL,
    so a thread pool gives a near-linear speedup. `keep_fn(cand) -> bool` is the cheap
    rejection gate. Kept candidates KEEP their (small) derivative thumbnail for analysis.
    Returns (kept_candidates, rejected_count, total).
    """
    from concurrent.futures import ThreadPoolExecutor

    import osxphotos

    db = osxphotos.PhotosDB(dbfile=dbfile)
    focal = _focal_35mm_by_uuid(db)
    photos = db.photos()

    def work(p: Any):
        try:
            object.__setattr__(p, "_focal_35mm", focal.get(p.uuid))
        except Exception:
            pass
        cand = candidate_from_photo(assert_no_master_access(p), cfg, LibraryStats())
        if cand is None:
            return None
        keep = keep_fn(cand)
        # Release the pixel buffer regardless — dedup/sampling operate on the lightweight
        # record (id/timestamp/exif-phash/meta); the sampled few are re-decoded later.
        cand.release_thumb()
        return cand if keep else False

    kept: list[Candidate] = []
    rejected = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(work, photos, chunksize=32):
            if result is None:
                continue
            if result is False:
                rejected += 1
            else:
                kept.append(result)
    return kept, rejected, len(photos)


# --------------------------------------------------------------------------- guardrail

class _MasterBlocked:
    """A thin proxy that raises if `.path` (the master) is accessed. For tests/runtime proof."""

    def __init__(self, photo: Any) -> None:
        object.__setattr__(self, "_photo", photo)

    def __getattr__(self, name: str) -> Any:
        if name == "path":
            raise LibraryMasterAccessError(
                "attempted to access a Photos master path (would trigger an iCloud download)"
            )
        return getattr(object.__getattribute__(self, "_photo"), name)


def assert_no_master_access(photo: Any) -> Any:
    """Wrap a photo so any `.path` access raises LibraryMasterAccessError."""
    return _MasterBlocked(photo)


# --------------------------------------------------------------------------- live entry

_FOCAL_35MM_SQL = (
    "SELECT a.ZUUID, e.ZFOCALLENGTHIN35MM FROM ZASSET a "
    "JOIN ZEXTENDEDATTRIBUTES e ON e.ZASSET = a.Z_PK "
    "WHERE e.ZFOCALLENGTHIN35MM IS NOT NULL AND e.ZFOCALLENGTHIN35MM > 0"
)


def _focal_35mm_by_uuid(db: Any) -> dict[str, float]:
    """35mm-equivalent focal per uuid via one read-only SELECT (osxphotos doesn't expose it)."""
    try:
        return {u: float(v) for u, v in db._db_connection.execute(_FOCAL_35MM_SQL)}
    except Exception:
        return {}


def stream_library(cfg: Config, dbfile: str | None = None) -> Iterator[Candidate]:
    """Open the Photos library read-only and yield Candidates. Needs Full Disk Access.

    Every photo is wrapped by `assert_no_master_access`, so even an accidental `.path`
    read anywhere downstream raises rather than silently triggering an iCloud download.
    """
    import osxphotos

    db = osxphotos.PhotosDB(dbfile=dbfile)
    focal = _focal_35mm_by_uuid(db)
    photos = []
    for p in db.photos():
        # Inject the SQL-sourced 35mm focal onto the object for candidate_from_photo.
        try:
            object.__setattr__(p, "_focal_35mm", focal.get(p.uuid))
        except Exception:
            pass
        photos.append(assert_no_master_access(p))
    yield from select(photos, cfg)


def stream_library_camera_meta(
    dbfile: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield camera metadata dicts (make/model/focal/focal35) for every Library photo.

    This is a METADATA-ONLY scan for the camera crop-factor backfill: it does NOT decode
    derivatives or touch masters, so it is fast over a large library. Each photo is still
    wrapped by `assert_no_master_access`, and only osxphotos metadata is read.
    """
    import osxphotos

    db = osxphotos.PhotosDB(dbfile=dbfile)
    focal = _focal_35mm_by_uuid(db)
    for raw in db.photos():
        if getattr(raw, "ismovie", False):
            continue
        p = assert_no_master_access(raw)
        exif = getattr(p, "exif_info", None)
        yield {
            "make": getattr(exif, "camera_make", None),
            "model": getattr(exif, "camera_model", None),
            "focal": getattr(exif, "focal_length", None),
            "focal35": focal.get(getattr(p, "uuid", None)),
        }
