"""Prepare a small, fixed labeling set and pre-render its thumbnails ONCE.

The labeling UI was slow because it decoded each photo on demand (and, for Library
photos, re-scanned the whole Photos DB per image). Instead we pick a bounded, stratified
labeling set up front, render each thumbnail to a static file once, and let the server
serve those files instantly. You label a few HUNDRED photos, not thousands.
"""

from __future__ import annotations

import json
import os
import random

from PIL import Image

from memory_graph.db import connect
from memory_graph.paths import DATA_DIR

LABEL_DIR = DATA_DIR / "labels"
THUMB_DIR = LABEL_DIR / "thumbs"
SET_FILE = LABEL_DIR / "label_set.json"

# How many distinct photos to put in the labeling set. A few hundred is plenty for
# tuning; you will NOT be asked to label all of them for every task.
DEFAULT_SET_SIZE = 400


def _render(row: dict, out_path) -> bool:
    try:
        if row["source"] == "hdd":
            from memory_graph.ingest.archive_cache import _PREVIEW_DIR

            uuid = row["id"].split(":", 1)[1]
            src = _PREVIEW_DIR / f"{uuid}.jpg"
            if not src.is_file():
                return False
            img = Image.open(src)
        else:
            path = _lib_derivative_cache.get(row["original_ref"])
            if not path or not os.path.isfile(path):
                return False
            img = Image.open(path)
        img = img.convert("RGB")
        img.thumbnail((600, 600))
        img.save(out_path, "JPEG", quality=80)
        return True
    except Exception:
        return False


# Built once in prepare_label_set so Library derivative paths are resolved without a
# per-photo PhotosDB scan.
_lib_derivative_cache: dict[str, str] = {}


def _build_library_derivative_map(uuids_needed: set[str]) -> None:
    """Resolve uuid -> local derivative path for the needed Library photos, in ONE scan."""
    needed = {u for u in uuids_needed}
    if not needed:
        return
    import osxphotos

    db = osxphotos.PhotosDB()
    for p in db.photos():
        if p.uuid not in needed:
            continue
        derivs = [d for d in (p.path_derivatives or []) if os.path.isfile(d)]
        if derivs:
            _lib_derivative_cache[p.uuid] = max(derivs, key=os.path.getsize)


def prepare_label_set(size: int = DEFAULT_SET_SIZE, seed: int = 0) -> dict:
    """Pick a stratified labeling set, pre-render thumbnails, and persist the set.

    Returns a summary dict. Idempotent: reuses an existing set + thumbnails, only
    rendering what is missing (resumable).
    """
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    conn = connect()
    rows = {
        r["id"]: dict(r)
        for r in conn.execute(
            "SELECT * FROM photos WHERE is_canonical = 1 AND private = 0"
        )
    }
    conn.close()

    if SET_FILE.is_file():
        chosen_ids = json.loads(SET_FILE.read_text())["ids"]
    else:
        # Stratify the pick across dominant quadrant so the quadrant task sees variety.
        by_q: dict[int, list[str]] = {}
        for pid, r in rows.items():
            by_q.setdefault(r.get("dominant_quadrant") or 0, []).append(pid)
        rng = random.Random(seed)
        per_q = max(1, size // max(1, len(by_q)))
        chosen_ids = []
        for _q, ids in sorted(by_q.items()):
            rng.shuffle(ids)
            chosen_ids.extend(ids[:per_q])
        rng.shuffle(chosen_ids)
        chosen_ids = chosen_ids[:size]
        SET_FILE.write_text(json.dumps({"ids": chosen_ids}))

    # Resolve Library derivative paths in one scan (avoids the per-image PhotosDB hang).
    lib_uuids = {
        rows[i]["original_ref"] for i in chosen_ids
        if i in rows and rows[i]["source"] == "library"
    }
    _build_library_derivative_map(lib_uuids)

    rendered = skipped = failed = 0
    for pid in chosen_ids:
        out = THUMB_DIR / f"{pid.replace(':', '_')}.jpg"
        if out.is_file():
            skipped += 1
            continue
        if pid in rows and _render(rows[pid], out):
            rendered += 1
        else:
            failed += 1
    return {
        "set_size": len(chosen_ids),
        "rendered": rendered,
        "reused": skipped,
        "failed": failed,
    }


def label_set_rows() -> list[dict]:
    """The labeling-set photo rows that have a rendered thumbnail, in set order."""
    if not SET_FILE.is_file():
        return []
    ids = json.loads(SET_FILE.read_text())["ids"]
    conn = connect()
    rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM photos WHERE is_canonical = 1")}
    conn.close()
    out = []
    for pid in ids:
        thumb = THUMB_DIR / f"{pid.replace(':', '_')}.jpg"
        if pid in rows and thumb.is_file():
            out.append(rows[pid])
    return out


def thumb_path(photo_id: str):
    return THUMB_DIR / f"{photo_id.replace(':', '_')}.jpg"
