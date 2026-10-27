"""Read EXIF from HDD files via exiftool (read-only).

Normalizes the tags the engine cares about into a plain dict. Compared with a generic
EXIF reader this adds the optics the 5-quadrant router needs — aperture, shutter, ISO —
alongside focal length / 35mm-equivalent, camera, orientation, timestamp, and GPS.

exiftool is invoked read-only (`-json -n -fast2`); it never writes to the source file.
Paths are passed on stdin so filenames with odd characters are safe.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

# Tags requested from exiftool. `-n` yields numeric values (degrees, EXIF codes, seconds).
TAGS = [
    "-Make", "-Model", "-LensModel",
    "-DateTimeOriginal", "-CreateDate", "-OffsetTimeOriginal",
    "-FocalLength", "-FocalLengthIn35mmFormat",
    "-FNumber", "-ApertureValue", "-ExposureTime", "-ISO",
    "-GPSLatitude", "-GPSLongitude", "-Orientation",
    "-ImageWidth", "-ImageHeight", "-ExifImageWidth", "-ExifImageHeight",
]  # fmt: skip

_DATE = re.compile(r"^(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})")


def _number(value: Any) -> float | None:
    """A float from a number or a string like '3.99 mm'; None when there is no number."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    m = re.match(r"\s*(-?[\d.]+)", str(value or ""))
    try:
        return float(m.group(1)) if m else None
    except (ValueError, AttributeError):
        return None


def _text(value: Any) -> str | None:
    return (str(value).strip() or None) if value is not None else None


def _capture(tags: dict) -> str | None:
    """'YYYY:MM:DD HH:MM:SS' from DateTimeOriginal, else CreateDate; None if absent/zeroed."""
    for key in ("DateTimeOriginal", "CreateDate"):
        m = _DATE.match(str(tags.get(key) or ""))
        if m and m.group(1) != "0000" and m.group(2) != "00" and m.group(3) != "00":
            return m.group(0)
    return None


def normalize(tags: dict | None) -> dict[str, Any]:
    """Normalize raw exiftool tags into the engine's EXIF dict (optics.py consumes this)."""
    tags = tags or {}
    width = _number(tags.get("ImageWidth")) or _number(tags.get("ExifImageWidth"))
    height = _number(tags.get("ImageHeight")) or _number(tags.get("ExifImageHeight"))
    orientation = _number(tags.get("Orientation"))
    return {
        "make": _text(tags.get("Make")),
        "model": _text(tags.get("Model")),
        "lens": _text(tags.get("LensModel")),
        "capture": _capture(tags),
        "offset": _text(tags.get("OffsetTimeOriginal")),
        "focal": _number(tags.get("FocalLength")),
        "focal35": _number(tags.get("FocalLengthIn35mmFormat")),
        "aperture": _number(tags.get("FNumber")) or _number(tags.get("ApertureValue")),
        "shutter": _number(tags.get("ExposureTime")),
        "iso": int(_number(tags.get("ISO"))) if _number(tags.get("ISO")) else None,
        "lat": _number(tags.get("GPSLatitude")),
        "lon": _number(tags.get("GPSLongitude")),
        "orientation": int(orientation) if orientation else None,
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "exif_error": not tags,
    }


def read_batch(paths: list[str]) -> dict[str, dict]:
    """Return {path: raw_tags} for a batch of files. Read-only exiftool call."""
    cmd = ["exiftool", "-json", "-n", "-fast2", "-charset", "filename=utf8", *TAGS, "-@", "-"]
    run = subprocess.run(cmd, input="\n".join(paths), capture_output=True, text=True)
    try:
        rows = json.loads(run.stdout or "[]")
    except json.JSONDecodeError:
        rows = []
    return {r["SourceFile"]: r for r in rows if "SourceFile" in r}


def read_one(path: str) -> dict[str, Any]:
    """Normalized EXIF for a single file."""
    return normalize(read_batch([path]).get(path))


# RAW files carry a full-size or preview JPEG we can decode without a RAW library.
_EMBEDDED = ("JpgFromRaw", "PreviewImage")


def embedded_jpeg(path: str) -> bytes | None:
    """The JPEG bytes embedded in a RAW file (full-size preferred), or None.

    exiftool only reads and prints bytes; the source file is not modified.
    """
    for tag in _EMBEDDED:
        run = subprocess.run(["exiftool", "-b", f"-{tag}", path], capture_output=True)
        if run.stdout[:2] == b"\xff\xd8":  # JPEG SOI marker
            return run.stdout
    return None


def extension(path: str) -> str:
    name = Path(path).name
    return ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
