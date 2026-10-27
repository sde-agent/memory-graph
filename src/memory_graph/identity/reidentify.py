"""Derivative-only face reidentification and anonymous recurrence clustering.

This is an on-demand V1.5 phase, deliberately separate from ingest. It reads only the
`deriv_path` already persisted by ingest, never `original_ref` for pixels, and stores
face metadata plus keys to compressed vectors rather than crops or vectors in SQLite.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from memory_graph import vision
from memory_graph.config import Config
from memory_graph.db import (
    get_face_observations,
    update_face_cluster,
    update_photo_anonymous_clusters,
    update_photo_fields,
    upsert_face_observation,
)
from memory_graph.identity.anonymous import FaceObservation, cluster_anonymous_faces
from memory_graph.identity.gallery import (
    FEATURE_PRINT_MODEL_VERSION,
    Gallery,
    add_named_references,
    bridge_by_phash,
)
from memory_graph.ingest.imaging import to_thumb
from memory_graph.metrics.faces import crop_face, named_people
from memory_graph.paths import DATA_DIR


class IdentityUnavailable(RuntimeError):
    """The local-only prerequisites for a reidentify run are unavailable."""


@dataclass(frozen=True)
class IdentityMatch:
    names: tuple[str, ...]
    similarity: float
    source: str | None


@dataclass
class IdentityRunStats:
    scanned: int = 0
    processed: int = 0
    skipped_no_derivative: int = 0
    failed_decode: int = 0
    detected_faces: int = 0
    named_faces: int = 0
    anonymous_clusters: int = 0
    vectors: int = 0


def match_face(
    embedding: list[float] | np.ndarray | None,
    gallery: Gallery | None,
    *,
    bridged_names: list[str] | tuple[str, ...] = (),
) -> IdentityMatch:
    """Match exact Library pHash bridges before the conservative feature gallery."""
    names = tuple(dict.fromkeys(str(name) for name in bridged_names if name))
    if names:
        return IdentityMatch(names=names, similarity=1.0, source="phash_bridge")
    if gallery is None:
        return IdentityMatch(names=(), similarity=0.0, source=None)
    match = gallery.match(embedding)
    return IdentityMatch(
        names=(match.name,) if match.name else (),
        similarity=match.similarity,
        source="feature_print" if match.name else None,
    )


def load_local_derivative(row: dict[str, Any], long_edge: int):
    """Decode only an existing persisted derivative/preview, never `original_ref`."""
    raw_path = row.get("deriv_path")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        image = Image.open(handle)
        try:
            return to_thumb(image, long_edge)
        finally:
            image.close()


def _vector_path(config: Config, setting: str) -> Path:
    raw = Path(getattr(config.identity, setting)).expanduser()
    return raw if raw.is_absolute() else DATA_DIR / raw


def _load_vectors(path: Path, expected_model: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format_version") != 1:
                raise ValueError("unsupported identity vector sidecar format")
            if metadata.get("model_version") != expected_model:
                raise ValueError("identity vector sidecar model version mismatch")
            matrix = np.asarray(archive["vectors"], dtype=np.float32)
            keys = list(metadata.get("keys", []))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise IdentityUnavailable(f"cannot load identity vectors: {exc}") from exc
    if len(keys) != len(matrix):
        raise IdentityUnavailable("identity vector sidecar keys and vectors differ")
    return {key: matrix[index] for index, key in enumerate(keys)}


def _save_vectors(path: Path, vectors: dict[str, np.ndarray], model_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(vectors)
    matrix = (np.vstack([vectors[key] for key in keys]).astype(np.float32)
              if keys else np.empty((0, 0), np.float32))
    metadata = json.dumps({
        "format_version": 1,
        "model_version": model_version,
        "keys": keys,
    }, separators=(",", ":"))
    fd, temp_name = tempfile.mkstemp(prefix=".face-embeddings-", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, vectors=matrix, metadata=np.array(metadata))
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _names_from_row(row: dict[str, Any]) -> list[str]:
    raw = row.get("person_names")
    if not raw:
        return []
    try:
        values = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return []
    return list(dict.fromkeys(str(value) for value in values if value))


def process_photo_row(
    row: dict[str, Any],
    thumb: Any,
    gallery: Gallery | None,
    library_index: dict[str, list[str]],
    *,
    vision_module=vision,
    crop_padding: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], set[str]]:
    """Pure per-photo identity work used by the resumable runner and unit tests."""
    faces = vision_module.detect_faces(thumb) or []
    vectors: dict[str, np.ndarray] = {}
    photo_names: set[str] = set()
    observations: list[dict[str, Any]] = []
    private = bool(row.get("private"))
    authoritative_names = _names_from_row(row) if row.get("source") == "library" else []
    bridge_names = (
        bridge_by_phash(row.get("phash") or "", library_index)
        if row.get("source") == "hdd" and not private else []
    )
    for ordinal, face in enumerate(faces):
        face_id = f"{row['id']}:face-{ordinal}"
        matched = IdentityMatch((), 0.0, None)
        embedding = None
        if not private:
            if authoritative_names:
                matched = IdentityMatch(
                    (authoritative_names[ordinal],), 1.0, "library_authoritative"
                ) if len(authoritative_names) == len(faces) else IdentityMatch(
                    tuple(authoritative_names), 1.0, "library_authoritative"
                )
            else:
                if bridge_names:
                    matched = match_face(None, gallery, bridged_names=bridge_names)
                else:
                    embedding = vision_module.feature_print(
                        crop_face(thumb, face, pad=crop_padding)
                    )
                    if embedding is not None:
                        vector = np.asarray(embedding, dtype=np.float32)
                        if vector.ndim == 1 and np.isfinite(vector).all():
                            vectors[face_id] = vector
                    matched = match_face(embedding, gallery, bridged_names=bridge_names)
        if matched.names:
            photo_names.update(matched.names)
        observations.append({
            "photo_id": row["id"],
            "face_id": face_id,
            "ordinal": ordinal,
            "bbox": {key: face.get(key) for key in ("x", "y", "w", "h")},
            "attributes": dict(face),
            "matched_name": matched.names[0] if matched.names else None,
            "similarity": matched.similarity if matched.source else None,
            "match_source": matched.source,
            "embedding_key": face_id if face_id in vectors else None,
            "model_version": FEATURE_PRINT_MODEL_VERSION if face_id in vectors else None,
            "private": 1 if private else 0,
        })
    return observations, vectors, photo_names


def _build_library_assets(config: Config, gallery: Gallery) -> dict[str, list[str]]:
    """Build references and an exact-pHash name index from local Library derivatives."""
    try:
        from memory_graph.ingest.library import stream_library
    except ImportError as exc:
        raise IdentityUnavailable("osxphotos is unavailable; cannot build the named gallery") \
            from exc
    index: dict[str, list[str]] = {}
    try:
        for candidate in stream_library(config):
            albums = set((candidate.library_meta or {}).get("albums") or [])
            if albums & set(config.privacy.hidden_albums):
                candidate.release_thumb()
                continue
            names = named_people(candidate.library_meta)
            phash = candidate.exif.get("phash")
            if phash and names:
                index.setdefault(phash, [])
                index[phash] = list(dict.fromkeys(index[phash] + names))
            add_named_references(
                gallery, candidate, pad=config.identity.crop_padding,
                max_references_per_name=config.identity.max_references_per_name,
            )
            candidate.release_thumb()
    except ImportError as exc:
        raise IdentityUnavailable("osxphotos is unavailable; cannot build the named gallery") \
            from exc
    except Exception as exc:  # noqa: BLE001 — provide a clear local prerequisite error
        raise IdentityUnavailable(f"could not read local Photos derivatives: {exc}") from exc
    return index


def reidentify(
    conn,
    config: Config,
    *,
    limit: int = 0,
    skip_gallery: bool = False,
    recognition_min_similarity: float | None = None,
    anonymous_threshold: float | None = None,
    anonymous_min_cluster_size: int | None = None,
) -> IdentityRunStats:
    """Run local face matching and recurrence clustering over existing DB rows."""
    if not vision.available():
        raise IdentityUnavailable("Apple Vision is unavailable; reidentify needs local Vision")
    gallery_path = _vector_path(config, "gallery_path")
    embeddings_path = _vector_path(config, "embeddings_path")
    min_similarity = (recognition_min_similarity
                      if recognition_min_similarity is not None
                      else config.faces.recognition_min_similarity)
    gallery = Gallery(min_similarity=min_similarity,
                      model_version=config.identity.feature_model_version)
    library_index: dict[str, list[str]] = {}
    if skip_gallery:
        if gallery_path.is_file():
            try:
                gallery = Gallery.load(
                    gallery_path, min_similarity=min_similarity,
                    expected_model_version=config.identity.feature_model_version,
                )
            except ValueError as exc:
                raise IdentityUnavailable(f"cannot load named gallery: {exc}") from exc
    else:
        library_index = _build_library_assets(config, gallery)
        gallery.save(gallery_path)

    for row in conn.execute(
        "SELECT phash, person_names FROM photos WHERE source = 'library' AND private = 0"
    ):
        names = _names_from_row(dict(row))
        if row["phash"] and names:
            library_index.setdefault(row["phash"], [])
            library_index[row["phash"]] = list(
                dict.fromkeys(library_index[row["phash"]] + names)
            )

    rows = [dict(row) for row in conn.execute(
        "SELECT * FROM photos WHERE is_canonical = 1 AND face_count > 0 ORDER BY id"
    )]
    if limit:
        rows = rows[:limit]
    stats = IdentityRunStats(scanned=len(rows))
    vectors = _load_vectors(embeddings_path, config.identity.feature_model_version)
    for row in rows:
        thumb = load_local_derivative(row, config.ingest.thumb_long_edge)
        if thumb is None:
            stats.skipped_no_derivative += 1
            continue
        try:
            observations, new_vectors, names = process_photo_row(
                row, thumb, gallery, library_index,
                crop_padding=config.identity.crop_padding,
            )
            for observation in observations:
                # Store the configured model version, not a future implementation's
                # hard-coded default, so sidecars and rows cannot silently mix models.
                if observation.get("embedding_key"):
                    observation["model_version"] = config.identity.feature_model_version
                upsert_face_observation(conn, observation)
            vectors.update(new_vectors)
            if row.get("source") == "hdd":
                identity_fields = {
                    "person_names": sorted(names),
                    "known_person": 1 if names else 0,
                }
                update_photo_fields(conn, row["id"], identity_fields)
                # Make the recognized-person signal immediately visible to dispatch. This
                # is DB-only; legacy rows lacking persisted router inputs remain available
                # for the existing `rescore` command rather than being guessed here.
                from memory_graph.pipeline import rescorable, rescore_row

                rescored_row = {**row, **identity_fields}
                if rescorable(rescored_row):
                    update_photo_fields(conn, row["id"], rescore_row(rescored_row, config))
            stats.processed += 1
            stats.detected_faces += len(observations)
            stats.named_faces += sum(1 for item in observations if item["matched_name"])
            stats.vectors += len(new_vectors)
            if stats.processed % 100 == 0:
                conn.commit()
                _save_vectors(embeddings_path, vectors, config.identity.feature_model_version)
            else:
                conn.commit()
        except Exception:
            conn.rollback()
            stats.failed_decode += 1
        finally:
            thumb.close()

    _save_vectors(embeddings_path, vectors, config.identity.feature_model_version)

    if stats.scanned and stats.processed == 0 and stats.skipped_no_derivative == stats.scanned:
        raise IdentityUnavailable(
            "no scanned photos have a readable local derivative/preview; "
            "identity never opens Photos masters"
        )

    persisted = get_face_observations(conn)
    observations: list[FaceObservation] = []
    for item in persisted:
        key = item.get("embedding_key")
        if item.get("private") or item.get("matched_name") or not key or key not in vectors:
            continue
        observations.append(FaceObservation(
            face_id=item["face_id"], photo_id=item["photo_id"],
            embedding=vectors[key], private=bool(item.get("private")),
        ))
    result = cluster_anonymous_faces(
        observations,
        similarity_threshold=(anonymous_threshold
                              if anonymous_threshold is not None
                              else config.identity.anonymous_similarity_threshold),
        min_cluster_size=(anonymous_min_cluster_size
                          if anonymous_min_cluster_size is not None
                          else config.identity.anonymous_min_cluster_size),
    )
    conn.execute("UPDATE face_observations SET anonymous_cluster_id = NULL")
    for item in observations:
        update_face_cluster(conn, item.photo_id, item.face_id, result.assignments[item.face_id])
    all_photo_ids = [row[0] for row in conn.execute("SELECT id FROM photos")]
    for photo_id in all_photo_ids:
        cluster_ids = [row[0] for row in conn.execute(
            "SELECT anonymous_cluster_id FROM face_observations "
            "WHERE photo_id = ? AND anonymous_cluster_id IS NOT NULL", (photo_id,)
        )]
        update_photo_anonymous_clusters(conn, photo_id, cluster_ids)
    conn.commit()
    stats.anonymous_clusters = len(result.clusters)
    return stats


def reidentify_cli(conn, config: Config, **kwargs: Any) -> IdentityRunStats:
    """Convenience wrapper for command handlers and integrations."""
    return reidentify(conn, config, **kwargs)


def cluster_persisted(
    conn,
    config: Config,
    *,
    anonymous_threshold: float | None = None,
    anonymous_min_cluster_size: int | None = None,
) -> int:
    """Cluster already-persisted unmatched face vectors without decoding any images."""
    embeddings_path = _vector_path(config, "embeddings_path")
    vectors = _load_vectors(embeddings_path, config.identity.feature_model_version)
    persisted = get_face_observations(conn)
    observations: list[FaceObservation] = []
    for item in persisted:
        key = item.get("embedding_key")
        if item.get("private") or item.get("matched_name") or not key or key not in vectors:
            continue
        observations.append(FaceObservation(
            face_id=item["face_id"], photo_id=item["photo_id"],
            embedding=vectors[key], private=bool(item.get("private")),
        ))
    result = cluster_anonymous_faces(
        observations,
        similarity_threshold=(anonymous_threshold
                              if anonymous_threshold is not None
                              else config.identity.anonymous_similarity_threshold),
        min_cluster_size=(anonymous_min_cluster_size
                          if anonymous_min_cluster_size is not None
                          else config.identity.anonymous_min_cluster_size),
    )
    conn.execute("UPDATE face_observations SET anonymous_cluster_id = NULL")
    for item in observations:
        update_face_cluster(conn, item.photo_id, item.face_id, result.assignments[item.face_id])
    for (photo_id,) in conn.execute("SELECT id FROM photos").fetchall():
        cluster_ids = [row[0] for row in conn.execute(
            "SELECT anonymous_cluster_id FROM face_observations "
            "WHERE photo_id = ? AND anonymous_cluster_id IS NOT NULL", (photo_id,)
        )]
        update_photo_anonymous_clusters(conn, photo_id, cluster_ids)
    conn.commit()
    return len(result.clusters)
