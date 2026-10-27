"""V1.5 identity sidecars, persistence, and derivative-only pure helpers."""

from __future__ import annotations

import json

from PIL import Image

from memory_graph.config import load_config
from memory_graph.db import (
    connect,
    get_face_observations,
    init_db,
    upsert_face_observation,
    upsert_photo,
)
from memory_graph.identity.gallery import Gallery
from memory_graph.identity.reidentify import match_face, process_photo_row
from memory_graph.narration.generate import GRAPH_KINDS, generate_graph


def _photo(pid: str, *, source: str = "hdd", private: int = 0) -> dict:
    return {
        "id": pid, "source": source, "original_ref": "/never/open/this-master.jpg",
        "phash": "abc", "sharpness": 1.0, "is_canonical": 1,
        "private": private,
    }


def test_gallery_versioned_compressed_round_trip(tmp_path) -> None:
    path = tmp_path / "gallery.npz"
    gallery = Gallery(min_similarity=0.91, model_version="test-feature-v1")
    gallery.add("Ada", [3.0, 4.0])
    gallery.save(path)

    loaded = Gallery.load(path, expected_model_version="test-feature-v1")
    assert loaded.model_version == "test-feature-v1"
    assert loaded.names == ["Ada"]
    assert loaded.match([0.6, 0.8]).name == "Ada"
    assert not list(tmp_path.glob("*.jpg"))  # serialization never writes crops


def test_face_observation_migration_and_upsert() -> None:
    conn = connect(":memory:")
    init_db(conn)
    upsert_photo(conn, _photo("hdd:a"))
    upsert_face_observation(conn, {
        "photo_id": "hdd:a", "face_id": "hdd:a:face-0", "ordinal": 0,
        "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        "attributes": {"yaw": 0.1}, "matched_name": "Ada", "similarity": 0.99,
        "match_source": "phash_bridge", "private": 0,
    })
    upsert_face_observation(conn, {
        "photo_id": "hdd:a", "face_id": "hdd:a:face-0", "ordinal": 0,
        "matched_name": "Grace", "similarity": 0.98,
    })
    rows = get_face_observations(conn, "hdd:a")
    assert len(rows) == 1
    assert rows[0]["matched_name"] == "Grace"
    assert json.loads(rows[0]["bbox"])["w"] == 0.3
    conn.close()


def test_named_match_prefers_exact_bridge_and_threshold() -> None:
    gallery = Gallery(min_similarity=0.99)
    gallery.add("Ada", [1.0, 0.0])
    bridged = match_face([0.0, 1.0], gallery, bridged_names=["Ada"])
    assert bridged.names == ("Ada",)
    assert bridged.source == "phash_bridge"
    below = match_face([0.0, 1.0], gallery)
    assert below.names == ()
    assert below.source is None


class _FakeVision:
    @staticmethod
    def detect_faces(_thumb):
        return [{"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4, "yaw": 0.2}]

    @staticmethod
    def feature_print(_crop):
        return [1.0, 0.0]


def test_reidentify_pure_helper_bridges_hdd_and_excludes_private() -> None:
    thumb = Image.new("RGB", (100, 100))
    gallery = Gallery(min_similarity=0.9)
    gallery.add("Ada", [1.0, 0.0])
    observed, vectors, names = process_photo_row(
        _photo("hdd:a"), thumb, gallery, {"abc": ["Ada"]}, vision_module=_FakeVision
    )
    assert observed[0]["matched_name"] == "Ada"
    assert observed[0]["match_source"] == "phash_bridge"
    assert not vectors and names == {"Ada"}

    def fail_feature(_crop):
        raise AssertionError("private face was embedded")

    _FakeVision.feature_print = staticmethod(fail_feature)
    observed, vectors, names = process_photo_row(
        _photo("lib:private", source="library", private=1),
        thumb, gallery, {}, vision_module=_FakeVision,
    )
    assert observed[0]["private"] == 1
    assert observed[0]["matched_name"] is None
    assert vectors == {} and names == set()


def test_anonymous_people_graph_has_no_name_entities() -> None:
    cfg = load_config()
    rows = [
        {**_photo("hdd:a"), "timestamp": 1_420_000_000.0,
         "face_count": 1, "anonymous_cluster_ids": '["anonymous-0001"]'},
        {**_photo("hdd:b"), "timestamp": 1_420_000_001.0,
         "face_count": 1, "anonymous_cluster_ids": '["anonymous-0001"]'},
    ]
    nodes = generate_graph(rows, cfg.eras, now=1.0)
    anonymous = [node for node in nodes if node.kind == "anonymous_people"]
    assert "anonymous_people" in GRAPH_KINDS
    assert anonymous
    assert anonymous[0].allowed_entities == ()
    assert "Ada" not in anonymous[0].claim
