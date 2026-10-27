"""DB schema, upsert/read round-trips, resumable-ingest id tracking (P0)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from memory_graph.contracts import Insight
from memory_graph.db import (
    connect,
    get_insight,
    get_photo,
    init_db,
    photo_ids,
    upsert_insight,
    upsert_photo,
)


@pytest.fixture()
def conn(tmp_path) -> sqlite3.Connection:
    c = connect(tmp_path / "test.db")
    init_db(c)
    yield c
    c.close()


def _minimal_photo(pid: str = "hdd:abc") -> dict:
    # Only the NOT-NULL-without-default columns are strictly required.
    return {
        "id": pid,
        "source": "hdd",
        "original_ref": "/Volumes/Archive/2012/img_0001.cr2",
        "phash": "ff00ff00",
        "sharpness": 123.4,
        "is_canonical": 1,
    }


def test_init_is_idempotent(conn: sqlite3.Connection) -> None:
    init_db(conn)  # second call must not raise
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"photos", "insights"} <= tables


def test_photo_round_trip(conn: sqlite3.Connection) -> None:
    upsert_photo(conn, _minimal_photo())
    row = get_photo(conn, "hdd:abc")
    assert row is not None
    assert row["source"] == "hdd"
    assert row["is_canonical"] == 1
    # Columns with SQL defaults come back populated.
    assert row["face_count"] == 0
    assert row["private"] == 0


def test_photo_upsert_updates_in_place(conn: sqlite3.Connection) -> None:
    upsert_photo(conn, _minimal_photo())
    upsert_photo(conn, {**_minimal_photo(), "resonance": 0.87, "dominant_quadrant": 3})
    row = get_photo(conn, "hdd:abc")
    assert row["resonance"] == 0.87
    assert row["dominant_quadrant"] == 3
    # Still exactly one row — upsert, not duplicate (resumable ingest).
    assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 1


def test_person_names_json_encoded(conn: sqlite3.Connection) -> None:
    upsert_photo(conn, {**_minimal_photo(), "person_names": ["Ada", "Grace"], "known_person": 1})
    row = get_photo(conn, "hdd:abc")
    assert json.loads(row["person_names"]) == ["Ada", "Grace"]


def test_photo_ids_for_resumability(conn: sqlite3.Connection) -> None:
    upsert_photo(conn, _minimal_photo("hdd:a"))
    upsert_photo(conn, _minimal_photo("hdd:b"))
    assert photo_ids(conn) == {"hdd:a", "hdd:b"}


def test_insight_round_trip_preserves_committed_direction(conn: sqlite3.Connection) -> None:
    ins = Insight(
        id="ins:1",
        kind="resonance_gap",
        claim="most-shot quadrant Q2; most-resonant Q3",
        direction="higher",
        magnitude=0.18,
        evidence_ids=("hdd:a", "lib:xyz"),
        resonance_angle="resonance share of Q3 exceeds its shooting share",
        allowed_entities=("Ada",),
        era="2010s",
        created_at=1_700_000_000.0,
    )
    upsert_insight(conn, ins)
    got = get_insight(conn, "ins:1")
    assert got == ins  # direction/magnitude/evidence all survive the round trip


def test_get_missing_returns_none(conn: sqlite3.Connection) -> None:
    assert get_photo(conn, "nope") is None
    assert get_insight(conn, "nope") is None


def test_insight_tags_round_trip(conn: sqlite3.Connection) -> None:
    ins = Insight(
        id="color:overall:warm", kind="color", claim="Your palette leans warm.",
        direction="higher", magnitude=0.6,
        tags=("color", "warm", "cool", "palette"), created_at=1.0,
    )
    upsert_insight(conn, ins)
    got = get_insight(conn, "color:overall:warm")
    assert got == ins
    assert got.tags == ("color", "warm", "cool", "palette")


def test_get_insights_returns_all_and_by_kind(conn: sqlite3.Connection) -> None:
    from memory_graph.db import get_insights, insert_insights

    insert_insights(conn, [
        Insight(id="a", kind="color", claim="c", tags=("color",), created_at=1.0),
        Insight(id="b", kind="framing", claim="f", tags=("framing",), created_at=1.0),
        Insight(id="c", kind="color", claim="c2", tags=("color",), created_at=1.0),
    ])
    assert len(get_insights(conn)) == 3
    assert {i.id for i in get_insights(conn, kind="color")} == {"a", "c"}


def test_replace_insights_of_kinds_clears_stale(conn: sqlite3.Connection) -> None:
    from memory_graph.db import get_insights, insert_insights, replace_insights_of_kinds

    # An old graph run left a 'warm' colour node and a critique node of another kind.
    insert_insights(conn, [
        Insight(id="color:overall:warm", kind="color", claim="warm", created_at=1.0),
        Insight(id="ins:2010s:style", kind="era_style", claim="keep me", created_at=1.0),
    ])
    # A fresh graph run: the palette flipped to 'cool'. Replacing the 'color' kind must
    # drop the stale warm node but leave the critique's era_style node untouched.
    replace_insights_of_kinds(
        conn,
        [Insight(id="color:overall:cool", kind="color", claim="cool", created_at=2.0)],
        kinds=("color", "framing"),
    )
    ids = {i.id for i in get_insights(conn)}
    assert "color:overall:warm" not in ids       # stale color node gone
    assert "color:overall:cool" in ids            # fresh color node present
    assert "ins:2010s:style" in ids               # untouched critique node survives


def test_insights_migration_adds_tags_column(tmp_path) -> None:
    # Simulate a DB created before the tags column existed, then migrate it.
    raw = sqlite3.connect(tmp_path / "old.db")
    raw.executescript(
        "CREATE TABLE insights (id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
        "claim TEXT NOT NULL, direction TEXT, magnitude REAL, evidence_ids TEXT, "
        "resonance_angle TEXT, allowed_entities TEXT, era TEXT, created_at REAL NOT NULL)"
    )
    raw.execute(
        "INSERT INTO insights (id, kind, claim, evidence_ids, allowed_entities, "
        "created_at) VALUES ('old', 'color', 'legacy', '[]', '[]', 1.0)"
    )
    raw.commit()
    raw.close()

    c = connect(tmp_path / "old.db")
    init_db(c)   # must ALTER TABLE to add tags without losing the legacy row
    cols = {r["name"] for r in c.execute("PRAGMA table_info(insights)")}
    assert "tags" in cols
    legacy = get_insight(c, "old")
    assert legacy is not None
    assert legacy.tags == ()          # legacy row has no tags, tolerated as empty
    # And a new tagged insight writes fine on the migrated table.
    upsert_insight(c, Insight(id="new", kind="color", claim="x",
                              tags=("warm",), created_at=2.0))
    assert get_insight(c, "new").tags == ("warm",)
    c.close()
