"""SQLite persistence (spec.md §7).

Stores the full soft quadrant vector and all resonance inputs per photo, plus the
insight graph. Target footprint < 100 MB for ~100k canonical photos. Writes are
upserts keyed by stable id so an interrupted ingest resumes rather than duplicates.

No full-resolution image bytes are ever stored here — only derived attributes and a
reference (uuid or path) back to the source.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memory_graph.contracts import Insight
from memory_graph.paths import DB_PATH, ensure_data_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id                 TEXT PRIMARY KEY,
    source             TEXT NOT NULL,
    original_ref       TEXT NOT NULL,
    timestamp          REAL,
    phash              TEXT NOT NULL,
    sharpness          REAL NOT NULL,
    is_canonical       INTEGER NOT NULL,
    burst_id           TEXT,
    camera_make        TEXT,
    camera_model       TEXT,
    lens_model         TEXT,
    latitude           REAL,
    longitude          REAL,
    city               TEXT,
    country            TEXT,
    state              TEXT,
    focal_raw          REAL,
    exif_focal35       REAL,
    focal_length_35mm  REAL,
    -- 1 if the photo carries GPS. Golden hour is computed from sun position, which needs
    -- GPS: without it golden_hour=0 means UNKNOWN, not 'not golden'. Gate on this flag.
    has_gps            INTEGER,
    aperture           REAL,
    shutter_speed      REAL,
    iso                INTEGER,
    ev                 REAL,
    horizon_tilt_deg   REAL,
    horizon_y_ratio    REAL,
    horizon_dutch      INTEGER,
    negative_space     REAL,
    golden_hour        INTEGER,
    warm_cool          REAL,
    face_count         INTEGER NOT NULL DEFAULT 0,
    face_ratio         REAL,
    candid_score       REAL,
    known_person       INTEGER NOT NULL DEFAULT 0,
    person_names       TEXT,
    q1_score           REAL NOT NULL DEFAULT 0,
    q2_score           REAL NOT NULL DEFAULT 0,
    q3_score           REAL NOT NULL DEFAULT 0,
    q4_score           REAL NOT NULL DEFAULT 0,
    q5_score           REAL NOT NULL DEFAULT 0,
    dominant_quadrant  INTEGER,
    favorite           INTEGER NOT NULL DEFAULT 0,
    taste_score        REAL,
    resonance          REAL,
    era                TEXT,
    last_dispatched    REAL,
    private            INTEGER NOT NULL DEFAULT 0,
    -- local image path for fast re-decode (HDD preview or Library derivative); NEVER a
    -- master. Lets the MCP server serve images without re-scanning PhotosDB (perf fix).
    deriv_path         TEXT,
    -- Vision scene classification labels (JSON [[name, confidence], ...]). Persisted so
    -- the quadrant router's CONTENT cues (people/scenic/domestic) can be reproduced in a
    -- DB-only re-score after an optics/config fix — WITHOUT a costly Vision re-decode.
    scene_labels       TEXT,
    -- JSON list of stable anonymous cluster ids seen in this photo. No names.
    anonymous_cluster_ids TEXT
);

CREATE TABLE IF NOT EXISTS face_observations (
    photo_id             TEXT NOT NULL,
    face_id              TEXT NOT NULL,
    ordinal              INTEGER NOT NULL,
    bbox                 TEXT,
    attributes           TEXT,
    matched_name         TEXT,
    similarity           REAL,
    match_source         TEXT,
    anonymous_cluster_id TEXT,
    embedding_key        TEXT,
    model_version        TEXT,
    private              INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (photo_id, face_id),
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_face_photo ON face_observations(photo_id);
CREATE INDEX IF NOT EXISTS idx_face_cluster ON face_observations(anonymous_cluster_id);
CREATE INDEX IF NOT EXISTS idx_canonical_ts  ON photos(is_canonical, timestamp);
CREATE INDEX IF NOT EXISTS idx_burst         ON photos(burst_id);
CREATE INDEX IF NOT EXISTS idx_resonance     ON photos(is_canonical, resonance);
CREATE INDEX IF NOT EXISTS idx_era           ON photos(era);

CREATE TABLE IF NOT EXISTS insights (
    id                TEXT PRIMARY KEY,
    kind              TEXT NOT NULL,
    claim             TEXT NOT NULL,
    direction         TEXT,
    magnitude         REAL,
    evidence_ids      TEXT NOT NULL,
    resonance_angle   TEXT,
    allowed_entities  TEXT,
    era               TEXT,
    created_at        REAL NOT NULL,
    tags              TEXT
);
CREATE INDEX IF NOT EXISTS idx_insight_era  ON insights(era);
CREATE INDEX IF NOT EXISTS idx_insight_kind ON insights(kind);
"""

# Columns of the photos table, in declaration order (minus the auto defaults handling).
_PHOTO_COLUMNS = [
    "id", "source", "original_ref", "timestamp", "phash", "sharpness",
    "is_canonical", "burst_id", "camera_make", "camera_model", "lens_model",
    "latitude", "longitude", "city", "country", "state", "focal_raw",
    "exif_focal35", "focal_length_35mm", "has_gps",
    "aperture", "shutter_speed", "iso", "ev", "horizon_tilt_deg", "horizon_y_ratio",
    "horizon_dutch", "negative_space", "golden_hour", "warm_cool", "face_count",
    "face_ratio", "candid_score", "known_person", "person_names", "q1_score", "q2_score",
    "q3_score", "q4_score", "q5_score", "dominant_quadrant", "favorite",
    "taste_score", "resonance", "era", "last_dispatched", "private", "deriv_path",
    "scene_labels", "anonymous_cluster_ids",
]


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (creating dirs as needed) a connection with sane pragmas and row access."""
    ensure_data_dirs()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not exist, and apply light migrations.

    Idempotent: safe to call on a fresh or an existing database.
    """
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first schema shipped (additive only)."""
    insight_cols = {r["name"] for r in conn.execute("PRAGMA table_info(insights)")}
    if "tags" not in insight_cols:
        conn.execute("ALTER TABLE insights ADD COLUMN tags TEXT")
    photo_cols = {r["name"] for r in conn.execute("PRAGMA table_info(photos)")}
    # Persisted signals that make the quadrant vector re-scorable without a Vision
    # re-decode (B2: an optics/config fix becomes a cheap DB-only re-score).
    for col, decl in (
        ("horizon_dutch", "INTEGER"),
        ("scene_labels", "TEXT"),
        ("focal_raw", "REAL"),
        ("exif_focal35", "REAL"),
        ("has_gps", "INTEGER"),
        ("anonymous_cluster_ids", "TEXT"),
        ("lens_model", "TEXT"),
        ("latitude", "REAL"),
        ("longitude", "REAL"),
        ("city", "TEXT"),
        ("country", "TEXT"),
        ("state", "TEXT"),
    ):
        if col not in photo_cols:
            conn.execute(f"ALTER TABLE photos ADD COLUMN {col} {decl}")


def upsert_photo(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Insert or replace a photo row by stable id. Missing columns take SQL defaults.

    `person_names` may be passed as a list; it is JSON-encoded here.
    """
    data = dict(row)
    if isinstance(data.get("person_names"), (list, tuple)):
        data["person_names"] = json.dumps(list(data["person_names"]))
    if isinstance(data.get("scene_labels"), (list, tuple)):
        data["scene_labels"] = json.dumps(list(data["scene_labels"]))
    cols = [c for c in _PHOTO_COLUMNS if c in data]
    placeholders = ", ".join(f":{c}" for c in cols)
    collist = ", ".join(cols)
    conn.execute(
        f"INSERT INTO photos ({collist}) VALUES ({placeholders}) "
        f"ON CONFLICT(id) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in cols if c != "id"),
        data,
    )


def get_photo(conn: sqlite3.Connection, photo_id: str) -> dict[str, Any] | None:
    cur = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
    r = cur.fetchone()
    return dict(r) if r else None


def photo_ids(conn: sqlite3.Connection) -> set[str]:
    """All known photo ids — used for resumable ingest (skip what's already done)."""
    return {r[0] for r in conn.execute("SELECT id FROM photos")}


def upsert_insight(conn: sqlite3.Connection, insight: Insight) -> None:
    d = asdict(insight)
    d["evidence_ids"] = json.dumps(list(insight.evidence_ids))
    d["allowed_entities"] = json.dumps(list(insight.allowed_entities))
    d["tags"] = json.dumps(list(insight.tags))
    conn.execute(
        """
        INSERT INTO insights
            (id, kind, claim, direction, magnitude, evidence_ids, resonance_angle,
             allowed_entities, era, created_at, tags)
        VALUES
            (:id, :kind, :claim, :direction, :magnitude, :evidence_ids,
             :resonance_angle, :allowed_entities, :era, :created_at, :tags)
        ON CONFLICT(id) DO UPDATE SET
            kind=excluded.kind, claim=excluded.claim, direction=excluded.direction,
            magnitude=excluded.magnitude, evidence_ids=excluded.evidence_ids,
            resonance_angle=excluded.resonance_angle,
            allowed_entities=excluded.allowed_entities, era=excluded.era,
            created_at=excluded.created_at, tags=excluded.tags
        """,
        d,
    )


def get_insight(conn: sqlite3.Connection, insight_id: str) -> Insight | None:
    cur = conn.execute("SELECT * FROM insights WHERE id = ?", (insight_id,))
    r = cur.fetchone()
    if not r:
        return None
    return _row_to_insight(r)


def _row_to_insight(r: sqlite3.Row) -> Insight:
    # `tags` may be absent on rows written before the column existed; tolerate it.
    tags_raw = r["tags"] if "tags" in r.keys() else None
    return Insight(
        id=r["id"],
        kind=r["kind"],
        claim=r["claim"],
        direction=r["direction"],
        magnitude=r["magnitude"],
        evidence_ids=tuple(json.loads(r["evidence_ids"])),
        resonance_angle=r["resonance_angle"],
        allowed_entities=tuple(json.loads(r["allowed_entities"] or "[]")),
        era=r["era"],
        created_at=r["created_at"],
        tags=tuple(json.loads(tags_raw or "[]")),
    )


def get_insights(conn: sqlite3.Connection, kind: str | None = None) -> list[Insight]:
    """All insight nodes (optionally of one kind) — the retrievable insight graph."""
    if kind is None:
        cur = conn.execute("SELECT * FROM insights")
    else:
        cur = conn.execute("SELECT * FROM insights WHERE kind = ?", (kind,))
    return [_row_to_insight(r) for r in cur]


def db_size_bytes(db_path: Path = DB_PATH) -> int:
    return db_path.stat().st_size if db_path.exists() else 0


def dispatch_candidates(conn: sqlite3.Connection, include_private: bool = False) -> list[dict]:
    """Canonical rows eligible for dispatch ranking (non-private unless asked)."""
    sql = "SELECT * FROM photos WHERE is_canonical = 1 AND resonance IS NOT NULL"
    if not include_private:
        sql += " AND private = 0"
    return [dict(r) for r in conn.execute(sql)]


def mark_dispatched(conn: sqlite3.Connection, photo_id: str, when: float) -> None:
    conn.execute("UPDATE photos SET last_dispatched = ? WHERE id = ?", (when, photo_id))


def count_photos(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0])


def insert_insights(conn: sqlite3.Connection, insights: Iterable[Insight]) -> None:
    for ins in insights:
        upsert_insight(conn, ins)


def update_photo_fields(conn: sqlite3.Connection, photo_id: str, fields: dict[str, Any]
                        ) -> None:
    """Update a subset of columns on one photo by id (used by the re-score path)."""
    cols = [c for c in fields if c != "id" and c in _PHOTO_COLUMNS]
    if not cols:
        return
    assignments = ", ".join(f"{c} = :{c}" for c in cols)
    params = {c: fields[c] for c in cols}
    for key in ("person_names", "scene_labels", "anonymous_cluster_ids"):
        if isinstance(params.get(key), (list, tuple)):
            params[key] = json.dumps(list(params[key]))
    params["id"] = photo_id
    conn.execute(f"UPDATE photos SET {assignments} WHERE id = :id", params)


def all_canonical_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Every canonical photo row as a dict (for a full-archive re-score)."""
    return [dict(r) for r in conn.execute("SELECT * FROM photos WHERE is_canonical = 1")]


def replace_insights_of_kinds(
    conn: sqlite3.Connection, insights: Iterable[Insight], kinds: Iterable[str]
) -> None:
    """Rebuild the insight graph for a set of kinds: delete the old nodes of those kinds,
    then insert the fresh ones. Keeps the graph from accumulating stale nodes across runs
    (e.g. a 'warm' node from an old run lingering after the palette flips to 'cool')."""
    kinds = list(kinds)
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        conn.execute(f"DELETE FROM insights WHERE kind IN ({placeholders})", kinds)
    insert_insights(conn, insights)


_FACE_COLUMNS = (
    "photo_id", "face_id", "ordinal", "bbox", "attributes", "matched_name",
    "similarity", "match_source", "anonymous_cluster_id", "embedding_key",
    "model_version", "private",
)


def upsert_face_observation(conn: sqlite3.Connection, observation: dict[str, Any]) -> None:
    """Atomically upsert one face observation; vectors live outside SQLite."""
    data = dict(observation)
    for key in ("bbox", "attributes"):
        if isinstance(data.get(key), (dict, list, tuple)):
            data[key] = json.dumps(data[key], separators=(",", ":"))
    cols = [column for column in _FACE_COLUMNS if column in data]
    placeholders = ", ".join(f":{column}" for column in cols)
    assignments = ", ".join(
        f"{column}=excluded.{column}" for column in cols if column not in {"photo_id", "face_id"}
    )
    conflict = ("DO NOTHING" if not assignments else f"DO UPDATE SET {assignments}")
    conn.execute(
        f"INSERT INTO face_observations ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(photo_id, face_id) {conflict}",
        data,
    )


def get_face_observations(
    conn: sqlite3.Connection, photo_id: str | None = None
) -> list[dict[str, Any]]:
    """Read face observations, optionally scoped to one photo."""
    if photo_id is None:
        rows = conn.execute("SELECT * FROM face_observations ORDER BY photo_id, ordinal")
    else:
        rows = conn.execute(
            "SELECT * FROM face_observations WHERE photo_id = ? ORDER BY ordinal", (photo_id,)
        )
    return [dict(row) for row in rows]


def update_face_cluster(
    conn: sqlite3.Connection, photo_id: str, face_id: str, cluster_id: str | None
) -> None:
    conn.execute(
        "UPDATE face_observations SET anonymous_cluster_id = ? "
        "WHERE photo_id = ? AND face_id = ?",
        (cluster_id, photo_id, face_id),
    )


def update_photo_anonymous_clusters(
    conn: sqlite3.Connection, photo_id: str, cluster_ids: Iterable[str]
) -> None:
    """Cache stable anonymous cluster ids on the photo, never names."""
    values = sorted(set(cluster_ids))
    conn.execute(
        "UPDATE photos SET anonymous_cluster_ids = ? WHERE id = ?",
        (json.dumps(values), photo_id),
    )
