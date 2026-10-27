"""Fast HDD ingest from the prototype's pre-rendered cache (optimization).

The sibling prototype already walked the external drive once and produced, under its
`data/`: upright preview JPEGs (<=1024px) keyed by a stable id, a records index (date,
GPS, camera, focal, favorite/rating), and cached EXIF. Reading from those previews on
the internal SSD is ~25x faster than re-decoding RAW off the spinning disk (4.5ms vs
113ms, measured), and the EXIF is already read.

So for the 10k tuning run we source HDD photos from this cache. One gap: the prototype's
EXIF omits aperture/shutter/ISO (our optics need them), so after SAMPLING we batch-read
just those three tags from the drive for the chosen photos (a light, fast metadata read
— not a RAW decode). This keeps full analysis quality at a fraction of the time.

This module is read-only over the prototype's cache and (optionally) the read-only HDD.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from memory_graph.config import Config
from memory_graph.contracts import Candidate, Source
from memory_graph.ingest.imaging import phash, to_thumb

# Optional reuse of a sibling prototype's preview/EXIF cache, to skip re-decoding RAW off
# a spinning HDD during a tuning run. Location is CONFIG/ENV-driven (no personal path in
# the repo): set MEMORY_GRAPH_ARCHIVE_CACHE to the cache's `data/` dir. Unset => no cache,
# and the engine ingests from the HDD directly (the normal path for any new user).
_PROTO_DATA = (
    Path(os.environ["MEMORY_GRAPH_ARCHIVE_CACHE"]).expanduser()
    if os.environ.get("MEMORY_GRAPH_ARCHIVE_CACHE")
    else None
)
_PREVIEW_DIR = (_PROTO_DATA / "archive_previews") if _PROTO_DATA else None
_RECORDS = (_PROTO_DATA / "archive_records.json") if _PROTO_DATA else None


@dataclass
class ArchiveRecord:
    uuid: str
    local_date: str | None
    path: str | None           # original on the HDD (for the EXIF top-up)
    focal_35mm: float | None
    focal_mm: float | None
    camera: str | None
    lat: float | None
    lon: float | None
    width: int | None
    height: int | None
    favorite: bool
    rating: int

    @property
    def preview_path(self) -> Path:
        return _PREVIEW_DIR / f"{self.uuid}.jpg"

    @property
    def year(self) -> int | None:
        if not self.local_date:
            return None
        try:
            return int(self.local_date[:4])
        except ValueError:
            return None


def load_records(records_path: Path = _RECORDS) -> list[ArchiveRecord]:
    """Load the prototype's archive records as lightweight ArchiveRecords."""
    data = json.loads(records_path.read_text())
    out: list[ArchiveRecord] = []
    for r in data.get("records", []):
        out.append(
            ArchiveRecord(
                uuid=r["uuid"],
                local_date=r.get("local_date"),
                path=r.get("image_path"),
                focal_35mm=r.get("focal_35mm"),
                focal_mm=r.get("focal_mm"),
                camera=r.get("camera"),
                lat=r.get("latitude"),
                lon=r.get("longitude"),
                width=r.get("width"),
                height=r.get("height"),
                favorite=bool(r.get("favorite")),
                rating=int(r.get("rating") or 0),
            )
        )
    return out


def available() -> bool:
    """True only when a prototype cache is CONFIGURED (env var set) and present on disk.
    Unset env var => no cache => callers fall back to direct HDD ingest."""
    if _RECORDS is None or _PREVIEW_DIR is None:
        return False
    return _RECORDS.is_file() and _PREVIEW_DIR.is_dir()


def _epoch(local_date: str | None) -> float | None:
    if not local_date:
        return None
    try:
        return datetime.fromisoformat(local_date).timestamp()
    except ValueError:
        return None


def candidate_from_record(rec: ArchiveRecord, cfg: Config) -> Candidate | None:
    """Build a Candidate from a cached preview + record. No HDD access. None if no preview."""
    pv = rec.preview_path
    if not pv.is_file():
        return None
    try:
        with open(pv, "rb") as fh:
            img = Image.open(fh)
            thumb = to_thumb(img, cfg.ingest.thumb_long_edge)
    except Exception:
        return None
    exif: dict[str, Any] = {
        "focal35": rec.focal_35mm,
        "focal": rec.focal_mm,
        "model": rec.camera,
        "lat": rec.lat,
        "lon": rec.lon,
        "width": rec.width,
        "height": rec.height,
        "phash": phash(thumb),
        # aperture/shutter/iso filled in later (top_up_optics) for sampled photos only.
    }
    meta = {"favorite": rec.favorite, "rating": rec.rating, "faces": [], "albums": []}
    return Candidate(
        id=f"hdd:{rec.uuid}",
        source=Source.HDD,
        original_ref=rec.path or rec.uuid,
        timestamp=_epoch(rec.local_date),
        thumb=thumb,
        exif=exif,
        library_meta=meta,
    )


def stream_archive_cache(cfg: Config) -> Iterator[Candidate]:
    """Yield Candidates from every cached preview (no HDD access, fast)."""
    for rec in load_records():
        cand = candidate_from_record(rec, cfg)
        if cand is not None:
            yield cand


def build_pool_parallel(cfg: Config, keep_fn, workers: int = 8) -> tuple[list[Candidate], int]:
    """Decode + cheap-gate every cached preview in parallel, keeping survivors.

    `keep_fn(cand) -> bool` decides whether to keep a candidate (the cheap rejection
    gate). Decode + pixel-stats + pHash per preview is pure CPU (PIL/numpy release the
    GIL), so a thread pool gives a near-linear speedup without pickling candidates across
    processes. Thumbs of KEPT candidates are released here; sampled ones are re-decoded
    later. Returns (kept_candidates, rejected_count).
    """
    from concurrent.futures import ThreadPoolExecutor

    records = load_records()

    def work(rec: ArchiveRecord):
        cand = candidate_from_record(rec, cfg)
        if cand is None:
            return None
        keep = keep_fn(cand)
        cand.release_thumb()   # free pixels immediately; keep only the lightweight record
        return cand if keep else False

    kept: list[Candidate] = []
    rejected = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(work, records, chunksize=64):
            if result is None:
                continue
            if result is False:
                rejected += 1
            else:
                kept.append(result)
    return kept, rejected


# Only these three tags are missing from the prototype's EXIF; read them in a batch for
# the SAMPLED photos (fast metadata read off the read-only drive, not a RAW decode).
_OPTICS_TAGS = ["-FNumber", "-ApertureValue", "-ExposureTime", "-ISO"]


def top_up_optics(candidates: list[Candidate]) -> int:
    """Batch-read aperture/shutter/ISO from the HDD for candidates that have a real path.

    Mutates each candidate's exif in place. Returns how many were topped up. Silently
    skips candidates whose original is not reachable (drive unmounted / moved).
    """
    import subprocess

    paths = [
        c.original_ref
        for c in candidates
        if c.original_ref and os.path.isfile(c.original_ref)
    ]
    if not paths:
        return 0
    cmd = [
        "exiftool", "-json", "-n", "-fast2", "-charset", "filename=utf8",
        *_OPTICS_TAGS, "-@", "-",
    ]
    run = subprocess.run(cmd, input="\n".join(paths), capture_output=True, text=True)
    try:
        rows = {r["SourceFile"]: r for r in json.loads(run.stdout or "[]") if "SourceFile" in r}
    except json.JSONDecodeError:
        return 0
    n = 0
    for c in candidates:
        tags = rows.get(c.original_ref)
        if not tags:
            continue
        fn = tags.get("FNumber") or tags.get("ApertureValue")
        c.exif["aperture"] = float(fn) if fn else None
        et = tags.get("ExposureTime")
        c.exif["shutter"] = float(et) if et else None
        iso = tags.get("ISO")
        c.exif["iso"] = int(iso) if iso else None
        n += 1
    return n
