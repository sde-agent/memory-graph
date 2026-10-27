"""Focused tests for conservative anonymous face recurrence clustering."""

from __future__ import annotations

import numpy as np
import pytest

from memory_graph.identity.anonymous import (
    AnonymousCluster,
    FaceObservation,
    cluster_anonymous_faces,
)


def _observation(face_id: str, photo_id: str, vector, *, private: bool = False) -> FaceObservation:
    return FaceObservation(face_id, photo_id, vector, private=private)


def test_repeated_faces_form_deterministic_anonymous_cluster() -> None:
    observations = [
        _observation("face-c", "photo-3", [0.99, 0.01]),
        _observation("face-a", "photo-1", [1.0, 0.0]),
        _observation("face-b", "photo-2", [0.98, 0.02]),
    ]

    result = cluster_anonymous_faces(observations, similarity_threshold=0.99)

    assert result.clusters == (
        AnonymousCluster(
            cluster_id="anonymous-0001",
            face_ids=("face-a", "face-b", "face-c"),
            photo_ids=("photo-1", "photo-2", "photo-3"),
        ),
    )
    assert result.assignments == {
        "face-a": "anonymous-0001",
        "face-b": "anonymous-0001",
        "face-c": "anonymous-0001",
    }


def test_input_order_does_not_change_ids_or_assignments() -> None:
    observations = [
        _observation("z", "p-z", [1.0, 0.0, 0.0]),
        _observation("a", "p-a", [0.0, 1.0, 0.0]),
        _observation("y", "p-y", [0.99, 0.01, 0.0]),
        _observation("b", "p-b", [0.01, 0.99, 0.0]),
    ]

    forward = cluster_anonymous_faces(observations, similarity_threshold=0.98)
    reverse = cluster_anonymous_faces(list(reversed(observations)), similarity_threshold=0.98)

    assert reverse == forward


def test_same_photo_faces_never_share_a_cluster() -> None:
    observations = [
        _observation("face-a", "same-photo", [1.0, 0.0]),
        _observation("face-b", "same-photo", [0.999, 0.001]),
        _observation("face-c", "other-photo", [1.0, 0.0]),
    ]

    result = cluster_anonymous_faces(observations, similarity_threshold=0.99)

    assert result.assignments["face-a"] == "anonymous-0001"
    assert result.assignments["face-b"] is None
    assert result.assignments["face-c"] == "anonymous-0001"
    assert result.clusters[0].photo_ids == ("other-photo", "same-photo")


def test_singletons_small_clusters_and_private_faces_remain_unassigned() -> None:
    observations = [
        _observation("singleton", "photo-1", [1.0, 0.0]),
        _observation("private-a", "photo-2", [1.0, 0.0], private=True),
        _observation("private-b", "photo-3", [1.0, 0.0], private=True),
        _observation("small-a", "photo-4", [0.0, 1.0]),
        _observation("small-b", "photo-5", [0.0, 1.0]),
    ]

    result = cluster_anonymous_faces(observations, min_cluster_size=3)

    assert result.clusters == ()
    assert all(value is None for value in result.assignments.values())


def test_vectors_are_normalized_and_invalid_vectors_are_conservative() -> None:
    observations = [
        _observation("scaled-a", "photo-a", [10.0, 0.0]),
        _observation("scaled-b", "photo-b", [0.0, -3.0]),
        _observation("zero", "photo-zero", [0.0, 0.0]),
        _observation("nan", "photo-nan", np.array([np.nan, 1.0])),
        _observation("empty", "photo-empty", []),
        _observation("scalar", "photo-scalar", 1.0),
    ]

    result = cluster_anonymous_faces(observations, similarity_threshold=0.99)

    assert result.clusters == ()
    assert result.assignments["scaled-a"] is None
    assert result.assignments["scaled-b"] is None
    assert result.assignments["zero"] is None
    assert result.assignments["nan"] is None
    assert result.assignments["empty"] is None
    assert result.assignments["scalar"] is None


def test_minimum_cluster_size_and_validation() -> None:
    observations = [
        _observation("a", "photo-a", [1.0, 0.0]),
        _observation("b", "photo-b", [1.0, 0.0]),
    ]

    assert cluster_anonymous_faces(observations, min_cluster_size=2).clusters
    with pytest.raises(ValueError, match="at least 2"):
        cluster_anonymous_faces(observations, min_cluster_size=1)
    with pytest.raises(ValueError, match="between"):
        cluster_anonymous_faces(observations, similarity_threshold=1.1)
    with pytest.raises(ValueError, match="duplicate"):
        cluster_anonymous_faces(observations + [observations[0]])
