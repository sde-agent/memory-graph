"""Named-person identity references and conservative matching.

The Photos Library's named faces are authoritative.  This module stores only normalized
Apple Vision feature-print vectors in a versioned compressed sidecar; it never stores
face crops or other pixels.  Generic feature prints are experimental, so callers must
keep the precision bar conservative and treat the model version as part of the data.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from memory_graph import vision
from memory_graph.contracts import Candidate
from memory_graph.metrics.faces import crop_face

GALLERY_FORMAT_VERSION = 1
FEATURE_PRINT_MODEL_VERSION = "vision-feature-print-v1"


@dataclass(frozen=True)
class Match:
    name: str | None
    similarity: float
    present: bool


@dataclass
class Gallery:
    """Per-person reference embeddings and nearest-neighbor matching."""

    min_similarity: float
    people: dict[str, list[np.ndarray]] = field(default_factory=dict)
    model_version: str = FEATURE_PRINT_MODEL_VERSION

    def add(self, name: str, embedding: list[float] | np.ndarray) -> None:
        name = str(name).strip()
        if not name or name == "_UNKNOWN_":
            return
        vec = _normalize(np.asarray(embedding, dtype=np.float32))
        if vec is None:
            return
        self.people.setdefault(name, []).append(vec)

    @property
    def names(self) -> list[str]:
        return sorted(self.people)

    @property
    def size(self) -> int:
        return sum(len(v) for v in self.people.values())

    def match(self, embedding: list[float] | np.ndarray | None) -> Match:
        """Nearest person; return a name only above the configured precision bar."""
        if embedding is None or not self.people:
            return Match(name=None, similarity=0.0, present=bool(self.people))
        q = _normalize(np.asarray(embedding, dtype=np.float32))
        if q is None:
            return Match(name=None, similarity=0.0, present=True)
        best_name, best_sim = None, -1.0
        for name, refs in self.people.items():
            for ref in refs:
                if ref.shape != q.shape:
                    continue
                sim = float(np.dot(q, ref))
                if sim > best_sim:
                    best_name, best_sim = name, sim
        named = best_name if best_sim >= self.min_similarity else None
        return Match(name=named, similarity=max(0.0, best_sim), present=True)

    def save(self, path: str | Path) -> None:
        """Write normalized references to a versioned, compressed vector sidecar."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        vectors: list[np.ndarray] = []
        records: list[dict[str, int | str]] = []
        for name in self.names:
            start = len(vectors)
            vectors.extend(self.people[name])
            records.append({"name": name, "start": start, "end": len(vectors)})
        matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0), np.float32)
        metadata = json.dumps({
            "format_version": GALLERY_FORMAT_VERSION,
            "model_version": self.model_version,
            "min_similarity": self.min_similarity,
            "records": records,
        }, separators=(",", ":"))
        with target.open("wb") as handle:
            np.savez_compressed(handle, vectors=matrix, metadata=np.array(metadata))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        min_similarity: float | None = None,
        expected_model_version: str | None = None,
    ) -> Gallery:
        """Load a sidecar and reject unknown formats/models rather than guessing."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format_version") != GALLERY_FORMAT_VERSION:
                raise ValueError("unsupported identity gallery format version")
            model_version = metadata.get("model_version")
            if not model_version or (expected_model_version is not None
                                     and model_version != expected_model_version):
                raise ValueError("unsupported or stale identity feature-print model")
            vectors = np.asarray(archive["vectors"], dtype=np.float32)
        gallery = cls(
            min_similarity=(min_similarity if min_similarity is not None
                            else float(metadata["min_similarity"])),
            model_version=model_version,
        )
        for record in metadata.get("records", []):
            start, end = int(record["start"]), int(record["end"])
            for vector in vectors[start:end]:
                gallery.add(str(record["name"]), vector)
        return gallery


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        return None
    return vec / norm


def cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Cosine similarity of two vectors (0 if either is degenerate)."""
    na = _normalize(np.asarray(a, dtype=np.float32))
    nb = _normalize(np.asarray(b, dtype=np.float32))
    if na is None or nb is None or na.shape != nb.shape:
        return 0.0
    return float(np.dot(na, nb))


def _face_box(face: dict[str, Any]) -> dict[str, float] | None:
    """Normalize Vision/Photos face metadata into a top-left normalized bbox."""
    raw = face.get("bbox") or face.get("bounding_box")
    if isinstance(raw, dict):
        face = {**face, **raw}
    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        face = {**face, "x": raw[0], "y": raw[1], "w": raw[2], "h": raw[3]}
    if all(face.get(key) is not None for key in ("x", "y", "w", "h")):
        return {key: float(face[key]) for key in ("x", "y", "w", "h")}
    size = face.get("size")
    cx, cy = face.get("center_x"), face.get("center_y")
    if size is None or cx is None or cy is None:
        return None
    size = float(size)
    return {"x": float(cx) - size / 2, "y": float(cy) - size / 2,
            "w": size, "h": size}


def add_named_references(
    gallery: Gallery,
    candidate: Candidate,
    *,
    feature_fn: Callable[[Any], list[float] | None] = vision.feature_print,
    pad: float = 0.25,
    max_references_per_name: int = 64,
) -> int:
    """Embed named faces from one derivative-backed Library candidate.

    The candidate must already have been decoded from `path_derivatives`; this helper
    never consults a Photos object or a master path.  It returns the number of vectors
    added and silently ignores missing boxes/embeddings conservatively.
    """
    if candidate.thumb is None or candidate.source.value != "library":
        return 0
    faces = (candidate.library_meta or {}).get("faces") or []
    added = 0
    for face in faces:
        name = face.get("name")
        box = _face_box(face)
        if not name or box is None or len(gallery.people.get(name, [])) >= max_references_per_name:
            continue
        embedding = feature_fn(crop_face(candidate.thumb, box, pad=pad))
        before = gallery.size
        gallery.add(name, embedding) if embedding is not None else None
        added += gallery.size - before
    return added


def build_named_gallery(
    candidates: Iterable[Candidate],
    *,
    min_similarity: float,
    feature_fn: Callable[[Any], list[float] | None] = vision.feature_print,
    pad: float = 0.25,
    model_version: str = FEATURE_PRINT_MODEL_VERSION,
) -> Gallery:
    """Build a named gallery from local-derivative Library candidates only."""
    gallery = Gallery(min_similarity=min_similarity, model_version=model_version)
    for candidate in candidates:
        add_named_references(gallery, candidate, feature_fn=feature_fn, pad=pad)
    return gallery


def bridge_by_phash(hdd_phash: str, library_index: dict[str, list[str]]) -> list[str]:
    """Return authoritative Library names for an exact HDD/Library pHash hit."""
    return list(dict.fromkeys(library_index.get(hdd_phash, [])))