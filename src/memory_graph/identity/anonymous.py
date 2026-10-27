"""Conservative, identity-free recurrence clustering for detected faces.

This module deliberately works only with face embeddings and stable observation
identifiers.  It never assigns or returns a human name.  The clusterer uses
complete-linkage agglomeration: two provisional clusters may merge only when
every cross-cluster pair clears the cosine threshold and their photos are
different.  The latter constraint prevents two detections from one photo
from being treated as the same recurring anonymous person.

Private observations are fail-closed: they remain in the returned assignment
map with ``None`` and cannot influence a public cluster.  Invalid embeddings,
singletons, and clusters smaller than the configured recurrence floor are also
left unassigned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Embedding = Sequence[float] | np.ndarray


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One detected face that may participate in anonymous recurrence clustering.

    ``face_id`` must identify this observation uniquely and ``photo_id`` must
    identify the containing photo.  Embeddings are normalized by the clustering
    function rather than at construction time so malformed observations can be
    conservatively returned as unassigned.
    """

    face_id: str
    photo_id: str
    embedding: Embedding
    private: bool = False


@dataclass(frozen=True, slots=True)
class AnonymousCluster:
    """A recurring anonymous face cluster with no person-name metadata."""

    cluster_id: str
    face_ids: tuple[str, ...]
    photo_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnonymousClusteringResult:
    """Deterministic clusters and an assignment for every input face.

    ``assignments`` maps every supplied ``face_id`` to an anonymous cluster ID
    or ``None``.  It is intentionally a plain mapping of IDs rather than a
    person record, so this result cannot expose a human name.
    """

    clusters: tuple[AnonymousCluster, ...]
    assignments: dict[str, str | None]


@dataclass
class _WorkingCluster:
    face_ids: tuple[str, ...]
    photo_ids: frozenset[str]
    vectors: tuple[np.ndarray, ...]


def cluster_anonymous_faces(
    observations: Sequence[FaceObservation],
    *,
    similarity_threshold: float = 0.90,
    min_cluster_size: int = 2,
) -> AnonymousClusteringResult:
    """Cluster recurring anonymous faces by normalized cosine similarity.

    The algorithm is deterministic complete-linkage agglomeration.  It starts
    with one provisional cluster per eligible observation, repeatedly merges
    the pair with the strongest complete-linkage similarity, and stops when no
    eligible pair clears ``similarity_threshold``.  A merge is rejected when
    the two clusters contain the same ``photo_id``; this enforces at most one
    face from any photo per final cluster.

    Private observations are excluded regardless of their embedding, and all
    observations are present in ``assignments``.  Zero, non-finite, malformed,
    dimension-mismatched, singleton, and too-small observations are represented
    by ``None`` rather than being promoted to an anonymous identity.

    Args:
        observations: Face detections with stable face/photo IDs and embeddings.
        similarity_threshold: Inclusive cosine threshold in ``[-1.0, 1.0]``.
        min_cluster_size: Minimum number of observations in a returned cluster.

    Raises:
        ValueError: If the threshold or minimum size is invalid, or face IDs
            are duplicated.
    """
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1.0 and 1.0")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")

    by_face_id: dict[str, FaceObservation] = {}
    for observation in observations:
        if observation.face_id in by_face_id:
            raise ValueError(f"duplicate face_id: {observation.face_id!r}")
        by_face_id[observation.face_id] = observation

    ordered = sorted(by_face_id.values(), key=lambda item: (item.face_id, item.photo_id))
    assignments = {observation.face_id: None for observation in ordered}
    if len(ordered) > 2_000:
        return _cluster_bucketed(
            ordered,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            assignments=assignments,
        )

    working: list[_WorkingCluster] = []

    for observation in ordered:
        if observation.private:
            continue
        vector = _normalize(observation.embedding)
        if vector is None:
            continue
        working.append(
            _WorkingCluster(
                face_ids=(observation.face_id,),
                photo_ids=frozenset((observation.photo_id,)),
                vectors=(vector,),
            )
        )

    while True:
        best: tuple[float, int, int] | None = None
        for left_index, left in enumerate(working):
            for right_index in range(left_index + 1, len(working)):
                right = working[right_index]
                if left.photo_ids & right.photo_ids:
                    continue
                similarity = _complete_linkage_similarity(left, right)
                if similarity is None or similarity < similarity_threshold:
                    continue
                candidate = (similarity, left_index, right_index)
                if best is None or _is_better_candidate(candidate, best, working):
                    best = candidate

        if best is None:
            break

        _, left_index, right_index = best
        merged = _merge(working[left_index], working[right_index])
        del working[right_index]
        del working[left_index]
        working.append(merged)

    eligible = sorted(
        (cluster for cluster in working if len(cluster.face_ids) >= min_cluster_size),
        key=lambda cluster: cluster.face_ids,
    )
    clusters: list[AnonymousCluster] = []
    for number, cluster in enumerate(eligible, start=1):
        cluster_id = f"anonymous-{number:04d}"
        record = AnonymousCluster(
            cluster_id=cluster_id,
            face_ids=cluster.face_ids,
            photo_ids=tuple(sorted(cluster.photo_ids)),
        )
        clusters.append(record)
        for face_id in record.face_ids:
            assignments[face_id] = cluster_id

    return AnonymousClusteringResult(clusters=tuple(clusters), assignments=assignments)


def _normalize(embedding: Embedding) -> np.ndarray | None:
    """Return a finite unit vector, or ``None`` for an unusable embedding."""
    try:
        vector = np.asarray(embedding, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    normalized = vector / norm
    if not np.isfinite(normalized).all():
        return None
    return normalized


def _complete_linkage_similarity(
    left: _WorkingCluster,
    right: _WorkingCluster,
) -> float | None:
    """Return the weakest cross-cluster cosine, or ``None`` for shape mismatch."""
    minimum = 1.0
    for left_vector in left.vectors:
        for right_vector in right.vectors:
            if left_vector.shape != right_vector.shape:
                return None
            similarity = float(np.dot(left_vector, right_vector))
            minimum = min(minimum, max(-1.0, min(1.0, similarity)))
    return minimum


def _is_better_candidate(
    candidate: tuple[float, int, int],
    current: tuple[float, int, int],
    working: list[_WorkingCluster],
) -> bool:
    """Compare candidates by score, then by stable member IDs."""
    candidate_score, candidate_left, candidate_right = candidate
    current_score, current_left, current_right = current
    if candidate_score != current_score:
        return candidate_score > current_score
    candidate_key = (
        working[candidate_left].face_ids,
        working[candidate_right].face_ids,
    )
    current_key = (working[current_left].face_ids, working[current_right].face_ids)
    return candidate_key < current_key


def _merge(left: _WorkingCluster, right: _WorkingCluster) -> _WorkingCluster:
    """Merge two already-compatible clusters in stable face-ID order."""
    members = sorted(
        zip(left.face_ids, left.vectors, strict=True), key=lambda item: item[0]
    )
    members.extend(zip(right.face_ids, right.vectors, strict=True))
    members.sort(key=lambda item: item[0])
    return _WorkingCluster(
        face_ids=tuple(face_id for face_id, _ in members),
        photo_ids=left.photo_ids | right.photo_ids,
        vectors=tuple(vector for _, vector in members),
    )


@dataclass
class _BucketCluster:
    face_ids: list[str]
    photo_ids: set[str]
    centroid: np.ndarray


def _cluster_bucketed(
    ordered: list[FaceObservation],
    *,
    similarity_threshold: float,
    min_cluster_size: int,
    assignments: dict[str, str | None],
) -> AnonymousClusteringResult:
    """Scale recurrence clustering without constructing an O(n²) similarity matrix.

    A fixed random-hyperplane locality hash narrows each observation to its bucket and
    one-bit neighbors. Within those candidate buckets, a deterministic nearest-centroid
    assignment applies the same similarity and one-face-per-photo safeguards. This is an
    intentionally conservative approximate path for large archives; small inputs retain
    the exact complete-linkage implementation above for maximal testability.
    """
    eligible: list[tuple[FaceObservation, np.ndarray, int]] = []
    projections: np.ndarray | None = None
    buckets: dict[int, list[int]] = {}
    for observation in ordered:
        if observation.private:
            continue
        vector = _normalize(observation.embedding)
        if vector is None:
            continue
        if projections is None:
            rng = np.random.default_rng(20260226)
            projections = rng.normal(size=(12, vector.shape[0]))
            projections /= np.linalg.norm(projections, axis=1, keepdims=True)
        if vector.shape != (projections.shape[1],):
            continue
        key = _bucket_key(vector, projections)
        index = len(eligible)
        eligible.append((observation, vector, key))
        buckets.setdefault(key, []).append(index)

    clusters: list[_BucketCluster] = []
    face_to_cluster: dict[str, int] = {}
    for observation, vector, key in eligible:
        candidate_cluster_ids: set[int] = set()
        for neighbor_key in _neighbor_bucket_keys(key, projections.shape[0]):
            for index in buckets.get(neighbor_key, ()):
                # The bucket index includes future observations too; only an observation
                # already assigned to a cluster is a candidate for the current item.
                prior = eligible[index][0]
                cluster_id = face_to_cluster.get(prior.face_id)
                if cluster_id is not None:
                    candidate_cluster_ids.add(cluster_id)

        best_id: int | None = None
        best_similarity = similarity_threshold
        for cluster_id in sorted(candidate_cluster_ids):
            cluster = clusters[cluster_id]
            if observation.photo_id in cluster.photo_ids:
                continue
            similarity = float(np.dot(vector, cluster.centroid))
            if similarity >= best_similarity:
                if (best_id is None or similarity > best_similarity or
                        cluster.face_ids < clusters[best_id].face_ids):
                    best_id = cluster_id
                    best_similarity = similarity
        if best_id is None:
            cluster_id = len(clusters)
            clusters.append(_BucketCluster(
                face_ids=[observation.face_id],
                photo_ids={observation.photo_id},
                centroid=vector.copy(),
            ))
            face_to_cluster[observation.face_id] = cluster_id
        else:
            cluster = clusters[best_id]
            cluster.face_ids.append(observation.face_id)
            cluster.photo_ids.add(observation.photo_id)
            face_to_cluster[observation.face_id] = best_id
            updated = cluster.centroid + vector
            norm = float(np.linalg.norm(updated))
            cluster.centroid = updated / norm if norm else cluster.centroid

    retained = sorted(
        (cluster for cluster in clusters if len(cluster.face_ids) >= min_cluster_size),
        key=lambda cluster: tuple(sorted(cluster.face_ids)),
    )
    records: list[AnonymousCluster] = []
    for number, cluster in enumerate(retained, start=1):
        face_ids = tuple(sorted(cluster.face_ids))
        record = AnonymousCluster(
            cluster_id=f"anonymous-{number:04d}",
            face_ids=face_ids,
            photo_ids=tuple(sorted(cluster.photo_ids)),
        )
        records.append(record)
        for face_id in face_ids:
            assignments[face_id] = record.cluster_id
    return AnonymousClusteringResult(clusters=tuple(records), assignments=assignments)


def _bucket_key(vector: np.ndarray, projections: np.ndarray) -> int:
    bits = projections @ vector >= 0
    return sum((1 << index) for index, bit in enumerate(bits) if bit)


def _neighbor_bucket_keys(key: int, width: int) -> tuple[int, ...]:
    """Return the exact bucket and all one-bit neighbors in stable order."""
    return (key,) + tuple(key ^ (1 << bit) for bit in range(width))
