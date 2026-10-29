"""External-HDD ingest adapter (spec.md §1).

Walks a photo archive on a (high-latency, spinning) external drive with os.scandir
generators, reads EXIF and downsamples to an in-memory thumbnail in a SINGLE pass per
file, and yields uniform `Candidate` objects. The drive is treated as READ-ONLY: every
file is opened only for reading, and `require_read_only` refuses to proceed unless the
mount is actually read-only.

Corrupt / unreadable files are skipped and COUNTED (never fatal) so a multi-hour run
over a large archive is not killed by one bad file.
"""

from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image

from memory_graph.config import Config
from memory_graph.contracts import Candidate, Source
from memory_graph.ingest import exif as exifmod
from memory_graph.ingest.imaging import phash, to_thumb

# Directories inside Apple library packages that hold derivatives, not originals.
_SKIP_DIRS = {"previews", "thumbnails", "thumbs", "resources", ".photoslibrary-cache"}
# RAW extensions whose embedded JPEG we decode (Pillow cannot open most RAW directly).
_RAW = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef"}


class HddNotReadOnly(RuntimeError):
    """The external drive is not mounted read-only; refuse to proceed."""


@dataclass
class HddStats:
    seen: int = 0            # files matching a photo extension
    yielded: int = 0         # candidates produced
    skipped_exif: int = 0    # no usable EXIF timestamp/optics
    skipped_decode: int = 0  # the image could not be decoded
    errors: list[str] = field(default_factory=list)


def require_read_only(root: str | Path) -> None:
    """Refuse to go on unless the filesystem holding `root` is mounted read-only."""
    try:
        flags = os.statvfs(root).f_flag
    except OSError as e:
        raise HddNotReadOnly(f"cannot stat {root} ({e}); is the drive mounted?") from e
    if not flags & os.ST_RDONLY:
        raise HddNotReadOnly(
            f"{root} is mounted read-write; mount the archive drive read-only before ingest"
        )


def _scan(root: str, extensions: frozenset[str]) -> Iterator[str]:
    """Yield photo file paths under `root`, sequentially, via os.scandir generators."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    name = entry.name
                    if name.startswith("."):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if name.lower() not in _SKIP_DIRS:
                            stack.append(entry.path)
                        continue
                    ext = exifmod.extension(name)
                    if ext in extensions:
                        yield entry.path
        except OSError:
            # An unreadable directory is skipped; the walk continues.
            continue


def _batched(it: Iterator[str], size: int) -> Iterator[list[str]]:
    """Yield lists of up to `size` items from an iterator."""
    batch: list[str] = []
    for x in it:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _stable_id(path: str) -> str:
    """A source-prefixed, stable id for a file: 'hdd:<sha1 of absolute path>'.

    Keyed on the path (not content) so a re-run upserts the same row cheaply without
    re-hashing bytes off the slow drive. Content identity is handled later by pHash.
    """
    digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()
    return f"hdd:{digest}"


def _capture_epoch(capture: str | None) -> float | None:
    if not capture:
        return None
    try:
        return datetime.strptime(capture, "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def _decode(path: str, ext: str) -> Image.Image | None:
    """Open the image once. RAW files use their embedded JPEG; others open directly."""
    try:
        if ext in _RAW:
            data = exifmod.embedded_jpeg(path)
            if data is None:
                return None
            return Image.open(io.BytesIO(data))
        return Image.open(path)
    except Exception:
        return None


def stream_hdd(root: str, cfg: Config, stats: HddStats | None = None) -> Iterator[Candidate]:
    """Yield a Candidate for each decodable photo under `root`.

    Single pass per file: read EXIF, decode once, build the upright thumbnail + pHash,
    attach them to the Candidate. The caller releases the thumbnail after metrics.
    Pass a `stats` object to collect per-run counts (skips, errors).
    """
    require_read_only(root)
    extensions = frozenset(cfg.ingest.hdd_extensions)
    stats = stats if stats is not None else HddStats()
    # Collect paths, then read EXIF in BATCHES (one exiftool process per 500 files vs one
    # per file — ~12x faster on that stage, measured).
    for batch in _batched(_scan(root, extensions), 500):
        tags = exifmod.read_batch(batch)
        for path in batch:
            stats.seen += 1
            ext = exifmod.extension(path)
            exif = exifmod.normalize(tags.get(path))
            img = _decode(path, ext)
            if img is None:
                stats.skipped_decode += 1
                stats.errors.append(path)
                continue
            try:
                thumb = to_thumb(img, cfg.ingest.thumb_long_edge)
            except Exception:
                stats.skipped_decode += 1
                stats.errors.append(path)
                continue
            finally:
                img.close()
            cand = Candidate(
                id=_stable_id(path),
                source=Source.HDD,
                original_ref=path,
                timestamp=_capture_epoch(exif.get("capture")),
                thumb=thumb,
                exif=exif,
                library_meta=None,
            )
            # pHash on the exif dict so clustering has it without re-decoding.
            cand.exif["phash"] = phash(thumb)
            stats.yielded += 1
            yield cand


def stream_hdd_with_stats(root: str, cfg: Config) -> tuple[list[Candidate], HddStats]:
    """Eager variant returning collected candidates and the run stats (for tests/reports)."""
    stats = HddStats()
    out = list(stream_hdd(root, cfg, stats))
    return out, stats


def stream_hdd_camera_meta(root: str, cfg: Config) -> Iterator[dict]:
    """Yield camera metadata dicts (make/model/focal/focal35) for every HDD photo.

    Metadata-only scan for the crop-factor backfill: it reads EXIF in batches but does
    NOT decode images, so it is fast and still honors the read-only mount guard.
    """
    require_read_only(root)
    extensions = frozenset(cfg.ingest.hdd_extensions)
    for batch in _batched(_scan(root, extensions), 500):
        tags = exifmod.read_batch(batch)
        for path in batch:
            exif = exifmod.normalize(tags.get(path))
            yield {
                "make": exif.get("make"),
                "model": exif.get("model"),
                "focal": exif.get("focal"),
                "focal35": exif.get("focal35"),
            }
