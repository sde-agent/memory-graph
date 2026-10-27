"""The analysis pipeline: a Candidate -> a persisted photo row (spec.md build plan).

Wires the phases together for one candidate: extract metrics (optics, geometry, light,
faces), score the quadrant vector, compute resonance, and assemble the DB row. Burst
clustering and rejection happen at the batch level (they need the whole set / a verdict
the owner reviews), so the pipeline here covers the per-photo analysis that produces a
canonical shot's persisted attributes.

Vision calls (faces, saliency) are the expensive part; they run once per canonical
representative. The thumbnail is released after analysis.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from memory_graph import vision
from memory_graph.cluster.bursts import Shot
from memory_graph.config import Config
from memory_graph.contracts import Candidate, Source
from memory_graph.metrics.faces import face_attributes, named_people
from memory_graph.metrics.geometry import horizon, negative_space
from memory_graph.metrics.light import extract_light, is_warm
from memory_graph.metrics.optics import extract_optics
from memory_graph.metrics.pixels import pixel_stats
from memory_graph.metrics.router import score_quadrants
from memory_graph.metrics.sharpness import sharpness
from memory_graph.scoring.resonance import ResonanceInputs, resonance
from memory_graph.scoring.taste import TasteModel

_NOW_YEAR_SECONDS = 365.25 * 24 * 3600


def analyze_candidate(
    cand: Candidate,
    cfg: Config,
    *,
    now: float | None = None,
    use_vision: bool = True,
    taste_model: TasteModel | None = None,
) -> dict[str, Any]:
    """Run metrics + scoring for one candidate and return a DB-ready photo row.

    `use_vision=False` skips the Apple Vision face/saliency calls (faster; used in
    tests and when Vision is unavailable) — the row is still complete, just without
    face/negative-space signals. `taste_model`, if given, scores the photo's embedding
    into the resonance `taste_score` multiplier (V1.5).
    """
    thumb = cand.thumb
    exif = cand.exif
    optics = extract_optics(exif, cfg)

    # Geometry + light need pixels / Vision.
    hz = horizon(thumb) if thumb is not None else _no_horizon()
    stats = pixel_stats(thumb) if thumb is not None else None
    lat, lon = exif.get("lat"), exif.get("lon")
    utc = datetime.fromtimestamp(cand.timestamp, UTC) if cand.timestamp else None
    light = extract_light(lat, lon, utc, cfg.sun, stats.warm_cool if stats else None)
    warm = is_warm(stats.warm_cool if stats else None, light.band, cfg.color.warm_from)

    # One Vision pass (shared handler) for classify + saliency + faces + optional feature
    # print — ~2.4x faster than separate calls. `classify` supplies the CONTENT cues the
    # router now uses (Part B: faces/scenery drive Q1/Q3, not just EXIF geometry).
    vres = None
    if use_vision and thumb is not None:
        want = ["classify", "saliency", "faces"]
        if taste_model is not None:
            want.append("featureprint")
        vres = vision.analyze_image(thumb, tuple(want))

    vfaces = vres.get("faces") if vres else None
    if vfaces is not None:
        fa = face_attributes(vision_faces=vfaces)
    else:
        fa = face_attributes(library_meta=cand.library_meta)
    names = named_people(cand.library_meta) if cand.source == Source.LIBRARY else []

    ns = negative_space(vres.get("saliency")) if vres else None
    scene_labels = vres.get("classify") if vres else None

    quad = score_quadrants(
        optics, hz, light, fa, cfg.quadrants,
        negative_space=ns, warm=warm, scene_labels=scene_labels,
    )

    favorite = bool((cand.library_meta or {}).get("favorite"))
    known_person = len(names) > 0
    age_years = ((now or _now()) - cand.timestamp) / _NOW_YEAR_SECONDS if cand.timestamp else None

    taste_score = None
    if taste_model is not None and vres is not None:
        taste_score = taste_model.score(vres.get("featureprint"))

    sharp = sharpness(thumb) if thumb is not None else float(exif.get("sharpness") or 0.0)

    res = resonance(
        ResonanceInputs(
            quadrants=quad,
            age_years=age_years,
            warm=warm,
            candid_score=fa.candid_score,
            favorite=favorite,
            known_person=known_person,
            face_present=bool(fa.face_count and fa.face_count > 0),
            taste_score=taste_score,
            sharpness=sharp,
        ),
        cfg.resonance_weights,
        cfg.resonance_modifiers,
    )

    return {
        "id": cand.id,
        "source": cand.source.value,
        "original_ref": cand.original_ref,
        "timestamp": cand.timestamp,
        "phash": exif.get("phash") or "",
        "sharpness": sharp,
        "is_canonical": 1,
        "camera_make": exif.get("make"),
        "camera_model": exif.get("model"),
        "focal_raw": optics.focal_raw,
        "exif_focal35": optics.exif_focal35,
        "focal_length_35mm": optics.focal_35mm,
        "aperture": optics.aperture,
        "shutter_speed": optics.shutter,
        "iso": optics.iso,
        "ev": optics.ev,
        "horizon_tilt_deg": hz.angle_deg,
        "horizon_y_ratio": hz.y_ratio,
        "horizon_dutch": 1 if hz.dutch else 0,
        "negative_space": ns,
        # Golden hour needs GPS (sun position). Record whether GPS existed so an absent
        # location is never read as "not golden hour".
        "has_gps": 1 if (lat is not None and lon is not None) else 0,
        "golden_hour": 1 if light.golden_hour else 0,
        "warm_cool": stats.warm_cool if stats else None,
        "face_count": fa.face_count,
        "face_ratio": fa.face_ratio,
        "candid_score": fa.candid_score,
        "known_person": 1 if known_person else 0,
        "person_names": names,
        "q1_score": quad.q1,
        "q2_score": quad.q2,
        "q3_score": quad.q3,
        "q4_score": quad.q4,
        "q5_score": quad.q5,
        "dominant_quadrant": quad.dominant,
        "favorite": 1 if favorite else 0,
        "taste_score": taste_score,
        "resonance": res,
        "private": 1 if _is_private(cand, cfg) else 0,
        "deriv_path": _local_image_path(cand),
        # Persist the Vision content cues so a later optics/config fix can re-score the
        # quadrant vector from the DB alone (B2), with no Vision re-decode.
        "scene_labels": scene_labels if scene_labels else None,
    }


def analyze_shot(shot: Shot, cfg: Config, **kw: Any) -> dict[str, Any]:
    """Analyze a burst's canonical representative, tagging the burst id."""
    row = analyze_candidate(shot.representative, cfg, **kw)
    row["burst_id"] = shot.burst_id
    return row


def rescore_row(row: dict[str, Any], cfg: Config, *, now: float | None = None
                ) -> dict[str, Any]:
    """Recompute the quadrant vector + resonance for a STORED photo row — no pixels.

    This is the B2 fast path: after an optics/config fix (e.g. a camera crop factor),
    the only metric that changed is the 35mm-equivalent focal, which is recomputed here
    from the stored raw inputs. Every OTHER router input — horizon (incl. the dutch
    gate), negative space, warm/golden, faces, and the Vision scene labels — was
    persisted at analysis time, so the router produces exactly what a full re-decode
    would, WITHOUT re-running Vision.

    Returns a dict of ONLY the fields that may change (focal_length_35mm, ev, the five
    quadrant scores, dominant_quadrant, resonance), ready to upsert by id.

    REQUIRES the row to carry the persisted re-score signals (scene_labels,
    horizon_dutch). Rows analyzed before those columns existed must be re-analyzed from
    pixels instead — the caller checks `rescorable(row)`.
    """
    from memory_graph.metrics.geometry import Horizon
    from memory_graph.metrics.light import Light

    # 1. Corrected optics. focal is re-derived from the camera's RAW focal via the crop
    #    table; aperture/shutter/iso are unchanged (stored), and EV follows.
    optics = _optics_from_row(row, cfg)

    # 2. Rebuild the other router inputs from stored columns (no pixels).
    hz = Horizon(
        angle_deg=row.get("horizon_tilt_deg"),
        y_ratio=row.get("horizon_y_ratio"),
        dutch=bool(row.get("horizon_dutch")),
        confidence=1.0 if row.get("horizon_dutch") else 0.0,
    )
    golden = bool(row.get("golden_hour"))
    light = Light(
        elevation=None,
        band="golden hour" if golden else "daylight",
        golden_hour=golden,
        warm_cool=row.get("warm_cool"),
    )
    warm = _warm_from_row(row, cfg)
    fa = _faces_from_row(row)
    scene_labels = _scene_labels_from_row(row)
    ns = row.get("negative_space")

    quad = score_quadrants(
        optics, hz, light, fa, cfg.quadrants,
        negative_space=ns, warm=warm, scene_labels=scene_labels,
    )

    age_years = _age_years(row.get("timestamp"), now)
    res = resonance(
        ResonanceInputs(
            quadrants=quad,
            age_years=age_years,
            warm=warm,
            candid_score=row.get("candid_score"),
            favorite=bool(row.get("favorite")),
            known_person=bool(row.get("known_person")),
            face_present=bool((row.get("face_count") or 0) > 0),
            taste_score=row.get("taste_score"),
            sharpness=row.get("sharpness"),
        ),
        cfg.resonance_weights,
        cfg.resonance_modifiers,
    )
    return {
        "id": row["id"],
        "focal_length_35mm": optics.focal_35mm,
        "ev": optics.ev,
        "q1_score": quad.q1, "q2_score": quad.q2, "q3_score": quad.q3,
        "q4_score": quad.q4, "q5_score": quad.q5,
        "dominant_quadrant": quad.dominant,
        "resonance": res,
    }


def rescorable(row: dict[str, Any]) -> bool:
    """True if the row carries the persisted signals a DB-only re-score needs. Rows
    written before the scene_labels/horizon_dutch columns existed are NOT re-scorable
    (their Vision content cues were never saved) and must be re-analyzed from pixels."""
    return "scene_labels" in row and "horizon_dutch" in row


def _optics_from_row(row, cfg):
    from memory_graph.metrics.optics import Optics, compute_ev, recompute_focal_35mm
    raw_focal = row.get("focal_raw")
    focal = recompute_focal_35mm(
        model=row.get("camera_model"),
        exif_focal35=row.get("exif_focal35"),
        raw_focal=raw_focal,
        cfg=cfg,
        # Pre-B2 rows have no focal_raw; fall back to the stored equivalent. For the
        # cameras we are correcting, that stored value IS the raw focal (the old code
        # stored raw when no crop existed), so the crop still applies correctly.
        stored_focal=row.get("focal_length_35mm"),
    )
    ap, sh, iso = row.get("aperture"), row.get("shutter_speed"), row.get("iso")
    return Optics(focal_35mm=focal, aperture=ap, shutter=sh, iso=iso,
                  ev=compute_ev(ap, sh, iso), focal_raw=raw_focal,
                  exif_focal35=row.get("exif_focal35"))


def _warm_from_row(row, cfg) -> bool:
    from memory_graph.metrics.light import is_warm
    band = "golden" if row.get("golden_hour") else "day"
    return is_warm(row.get("warm_cool"), band, cfg.color.warm_from)


def _faces_from_row(row):
    from memory_graph.metrics.faces import FaceAttributes
    return FaceAttributes(
        face_count=int(row.get("face_count") or 0),
        face_ratio=row.get("face_ratio"),
        eyes_open=None,          # not used by the router; irrelevant to a re-score
        candid_score=row.get("candid_score"),
    )


def _scene_labels_from_row(row):
    import json
    raw = row.get("scene_labels")
    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    # Stored as [[name, conf], ...]; the router expects (name, conf) tuples.
    return [(name, float(conf)) for name, conf in data]


def _age_years(timestamp, now):
    if not timestamp:
        return None
    return ((now or _now()) - timestamp) / _NOW_YEAR_SECONDS


def _is_private(cand: Candidate, cfg: Config) -> bool:
    """A photo in a configured hidden album is off-limit (never dispatched off-device)."""
    albums = set((cand.library_meta or {}).get("albums") or [])
    return bool(albums & set(cfg.privacy.hidden_albums))


def _local_image_path(cand: Candidate) -> str | None:
    """A local image path to re-decode for display later — NEVER a master.

    HDD: the cached preview (derivable from the id). Library: the derivative path
    resolved at ingest and stashed on exif. Persisting this lets the MCP server serve
    images without re-scanning the Photos DB per image (the dispatch-latency fix).
    """
    if cand.source == Source.HDD:
        from memory_graph.ingest.archive_cache import _PREVIEW_DIR

        if _PREVIEW_DIR is None:
            return None   # no prototype cache configured; HDD previews unavailable
        uuid = cand.id.split(":", 1)[1]
        pv = _PREVIEW_DIR / f"{uuid}.jpg"
        return str(pv) if pv.is_file() else None
    return cand.exif.get("deriv_path")


def _no_horizon():
    from memory_graph.metrics.geometry import Horizon

    return Horizon(angle_deg=None, y_ratio=None, dutch=False, confidence=0.0)


def _now() -> float:
    return datetime.now(UTC).timestamp()
