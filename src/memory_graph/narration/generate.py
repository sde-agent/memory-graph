"""Generate the INSIGHT GRAPH: a large population of committed insight nodes.

The insights ARE the graph (intent doc). The magic is not a stats tool the LLM queries;
it is many pre-computed, grounded observation nodes — each a fact + its lived meaning —
that the LLM SELECTS from and weaves to answer a question. The engine derives and commits
the nodes (direction/magnitude fixed by code); the LLM only picks and narrates.

Each generator turns the whole-archive / per-era aggregates into one or more `Insight`
nodes across a dimension: light & color, framing distance, people & relationships,
time-of-day, solitude vs togetherness, era style, change-over-time. As more dimensions
are added, the graph grows (hundreds of nodes across the eras), and broad questions
naturally pull more nodes (a paragraph) while narrow ones pull a few (a sentence).

Every node carries `tags` (retrieval keywords, on the Insight itself) and a
feeling-framed `claim`; the number is incidental evidence, never the headline.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import UTC, datetime

from memory_graph.config import ErasCfg, OwnerCfg
from memory_graph.contracts import Insight
from memory_graph.narration.critique import _median, _top_people, _year

_QUADRANT_NAMES = {
    1: "Nostalgic & Reflective", 2: "Visceral & High-Arousal",
    3: "Empathetic & Connected", 4: "Solitary & Introspective",
    5: "Provocative & Dynamic",
}

# The kinds `generate_graph` emits — persistence replaces exactly these on a rebuild, so
# the graph never accumulates stale nodes and never clobbers the critique's own kinds.
GRAPH_KINDS: tuple[str, ...] = (
    "color", "framing", "people", "light", "solitude", "groups", "style", "change",
    "landscape", "relationship", "signature", "unknown_people", "anonymous_people",
    "camera", "depth", "exposure", "geometry", "quality", "season", "lens", "place",
    "travel", "resonance", "taste", "favorites", "subject", "conjunction",
)

# Curated SUBJECT groups: Vision scene labels (conf >= _SUBJECT_CONF) grouped into the
# things a person actually photographs. Curated, not raw, so Vision junk ("structure",
# "material", "textile" — background stuff, not a subject) never becomes an insight. Each
# group: (slug, retrieval tags, label set, feeling-framed phrasing).
_SUBJECT_CONF = 0.3
_SUBJECT_GROUPS: tuple[tuple[str, tuple[str, ...], frozenset[str], str], ...] = (
    ("children",
     ("children", "child", "baby", "kids", "kid", "babies", "son", "daughter", "little"),
     frozenset({"child", "baby", "teen", "toddler"}),
     "children — the young faces of your family growing up in front of the lens"),
    ("nature",
     ("nature", "outdoors", "outdoor", "landscape", "sky", "plants", "trees", "grass",
      "garden", "flowers", "greenery", "outside"),
     frozenset({"outdoor", "sky", "blue_sky", "cloudy", "foliage", "plant", "grass",
                "tree", "mountain", "hill", "field", "garden", "flower", "forest"}),
     "the outdoors — open sky, greenery, the natural world"),
    ("water",
     ("water", "beach", "ocean", "sea", "lake", "river", "waterfall", "coast", "shore",
      "waves"),
     frozenset({"water", "beach", "ocean", "sea", "lake", "river", "waterfall", "sand",
                "shore", "coast", "wave"}),
     "water and shorelines — beaches, lakes, the sea"),
    ("animals",
     ("animals", "animal", "dog", "cat", "pet", "pets", "bird", "dogs", "cats"),
     frozenset({"animal", "dog", "cat", "bird", "pet", "horse"}),
     "animals — the pets and creatures that share your frames"),
    ("food",
     ("food", "meal", "meals", "eating", "cooking", "dessert", "cake", "drinks"),
     frozenset({"food", "meal", "dessert", "cake", "fruit", "drink"}),
     "food and shared meals — the table as a gathering place"),
    ("travel",
     ("travel", "trips", "trip", "vacation", "journey", "car", "train", "boat",
      "flight", "road", "luggage"),
     frozenset({"vehicle", "car", "train", "boat", "aircraft", "luggage", "road"}),
     "travel and the road — vehicles, luggage, places away from home"),
    ("celebration",
     ("celebration", "celebrations", "party", "wedding", "birthday", "festival",
      "occasion", "event"),
     frozenset({"wedding", "balloon", "cake", "candle", "decoration"}),
     "celebrations — weddings, parties, the days you gathered to mark something"),
)
_SUBJECT_TAGS = ("subject", "what", "photograph", "shoot", "things", "about", "themes")


def generate_graph(
    rows: list[dict], eras: ErasCfg, *, owner: OwnerCfg | None = None,
    now: float | None = None, progress=None,
) -> list[Insight]:
    """Return insight nodes spanning every dimension we can compute.

    Each node's `tags` are retrieval keywords used by `ask` to select nodes relevant to
    a question; `allowed_entities` fences which names narration may use. `owner` names the
    archive owner so owner-present photos (taken by others) are treated correctly.
    """
    owner = owner or OwnerCfg()
    created = now if now is not None else time.time()
    canon = [r for r in rows if r.get("is_canonical") and not r.get("private")]
    out: list[Insight] = []

    def add(iid, kind, claim, tags, *, direction=None, magnitude=None,
            entities=(), era=None):
        out.append(Insight(
            id=iid, kind=kind, claim=claim, direction=direction, magnitude=magnitude,
            allowed_entities=tuple(entities), era=era, created_at=created,
            tags=tuple(tags),
        ))

    # Whole-archive and per-era buckets.
    by_era: dict[str, list[dict]] = {}
    for r in canon:
        e = eras.era_of(_year(r.get("timestamp")))
        if e:
            by_era.setdefault(e, []).append(r)
    buckets = [("overall", canon)] + [(er.label, by_era[er.label])
                                      for er in eras.ranges if er.label in by_era]
    buckets = [(scope, rs) for scope, rs in buckets if rs]
    total_steps = len(buckets) + 8
    step = 0

    def tick(label: str) -> None:
        nonlocal step
        step += 1
        if progress is not None:
            progress(label, step, total_steps)

    for scope, rs in buckets:
        n = len(rs)
        era = None if scope == "overall" else scope
        where = "across your photographs" if scope == "overall" else f"in the {scope}"

        _light_color_nodes(add, scope, era, where, rs, n)
        _framing_nodes(add, scope, era, where, rs, n)
        _depth_nodes(add, scope, era, where, rs, n)
        _promotion_technical_nodes(add, scope, era, where, rs)
        _people_nodes(add, scope, era, where, rs, n, owner)
        _timeofday_nodes(add, scope, era, where, rs, n)
        _solitude_nodes(add, scope, era, where, rs, n)
        _landscape_nodes(add, scope, era, where, rs, n)
        _unknown_presence_nodes(add, scope, era, where, rs, n)
        _anonymous_people_nodes(add, scope, era, where, rs, n)
        _subject_nodes(add, scope, era, where, rs, n)
        _quadrant_nodes(add, scope, era, where, rs, n)
        tick(f"graph: {scope}")

    _change_nodes(add, by_era, eras)
    tick("graph: changes")
    _relationship_nodes(add, canon, by_era, eras, created, owner)
    tick("graph: relationships")
    _signature_nodes(add, canon, created, owner)
    tick("graph: signatures")
    _golden_conjunction_nodes(add, canon, created, owner)
    tick("graph: golden-hour relationships")
    _self_presence_nodes(add, canon, by_era, eras, owner)
    tick("graph: self-presence")
    _unknown_trend_nodes(add, by_era, eras)
    tick("graph: identity coverage")
    _camera_nodes(add, canon)
    tick("graph: camera patterns")
    _promotion_aggregate_nodes(add, canon, by_era, owner)
    tick("graph: resonance and context")
    return out


def _frac(rs, pred) -> float:
    return sum(1 for r in rs if pred(r)) / len(rs) if rs else 0.0


def _names(r) -> list[str]:
    """The named people on one photo (Library face names above the recognition bar)."""
    pn = r.get("person_names")
    if not pn:
        return []
    try:
        return list(json.loads(pn) if isinstance(pn, str) else pn)
    except (json.JSONDecodeError, TypeError):
        return []


def _light_color_nodes(add, scope, era, where, rs, n) -> None:
    warms = [r.get("warm_cool") for r in rs if r.get("warm_cool") is not None]
    if not warms:
        return
    tags = ("color", "warm", "cool", "tone", "palette", "temperature", "colour")
    warm_frac = sum(1 for w in warms if w > 0.08) / len(warms)
    cool_frac = sum(1 for w in warms if w < -0.08) / len(warms)
    # Full distribution: name the lean with its magnitude even when it is modest, so a
    # colour question ALWAYS gets a specific answer. A clear lean (>=0.45) is stated as
    # a pull; a near-even split is stated as balance — but still with the numbers.
    lean = warm_frac - cool_frac
    if warm_frac >= 0.45 and lean >= 0.1:
        add(f"color:{scope}:warm", "color",
            f"Your palette {where} leans warm — golden, amber, sun-touched tones show up "
            f"more than cool ones.",
            tags, direction="higher", magnitude=round(warm_frac, 2), era=era)
    elif cool_frac >= 0.45 and lean <= -0.1:
        add(f"color:{scope}:cool", "color",
            f"Your palette {where} leans cool — blues, shade, and quiet light over warm "
            f"tones.",
            tags, direction="higher", magnitude=round(cool_frac, 2), era=era)
    else:
        # Even a balanced palette is a finding — but say WHICH WAY it tips, if at all.
        tip = ("with a faint warm tilt" if lean > 0.03 else
               "with a faint cool tilt" if lean < -0.03 else
               "with no real tilt either way")
        add(f"color:{scope}:balanced", "color",
            f"Your palette {where} is fairly balanced between warm and cool light, {tip}.",
            tags, direction="higher" if lean > 0 else "lower" if lean < 0 else None,
            magnitude=round(lean, 2), era=era)


_FRAMING_TAGS = ("framing", "focal", "lens", "wide", "normal", "telephoto", "zoom",
                 "distance", "close", "angle")


def _framing_nodes(add, scope, era, where, rs, n) -> None:
    focals = [r["focal_length_35mm"] for r in rs if r.get("focal_length_35mm")]
    med = _median(focals)
    if med is None:
        return
    fn = len(focals)
    if med <= 30:
        desc = ("wide — you tend to take in the whole scene, placing people in their "
                "surroundings")
    elif med <= 60:
        desc = ("a normal, eye-level view — the world roughly as you actually saw it, "
                "neither grand nor tight")
    else:
        desc = ("tighter and more telephoto — you single things out and compress space, "
                "leaning in on a subject")
    add(f"framing:{scope}", "framing",
        f"Your typical framing {where} is {desc}.",
        _FRAMING_TAGS, magnitude=round(med, 0), era=era)

    # Full distribution: the SPREAD across wide / normal / tele, so a question about how
    # varied the framing is (or whether you ever go long) gets a real answer, not just
    # the median. Each band is a node when it is a meaningful share.
    wide = sum(1 for f in focals if f <= 30) / fn
    normal = sum(1 for f in focals if 30 < f <= 70) / fn
    tele = sum(1 for f in focals if f > 70) / fn
    for share, slug, phrase in (
        (wide, "wide", "take in the whole scene at a wide angle"),
        (normal, "normal", "frame at a natural, eye-level distance"),
        (tele, "tele", "reach in tight with a telephoto, isolating a subject"),
    ):
        if share >= 0.15:
            add(f"framing:{scope}:{slug}", "framing",
                f"A notable share of your photographs {where} {phrase}.",
                _FRAMING_TAGS, direction="higher", magnitude=round(share, 2), era=era)


def _depth_nodes(add, scope, era, where, rs, n) -> None:
    """Describe depth-of-field preference from an explicitly labeled optical proxy.

    EXIF gives aperture and focal length, not the rendered bokeh shape. We therefore say
    "bokeh-friendly" or "shallow-depth conditions" and include the data-coverage caveat
    rather than pretending this is a direct blur measurement.
    """
    known = [r for r in rs if r.get("aperture") is not None]
    if len(known) < 30:
        return
    wide = [r for r in known if r["aperture"] <= 2.8]
    deep = [r for r in known if r["aperture"] >= 8.0]
    bokeh_likely = [
        r for r in wide
        if (r.get("focal_length_35mm") or 0) >= 50 or (r.get("face_ratio") or 0) >= 0.12
    ]
    coverage = len(known) / len(rs) if rs else 0.0
    loc = "across the archive" if scope == "overall" else where
    wide_frac = len(wide) / len(known)
    deep_frac = len(deep) / len(known)
    bokeh_frac = len(bokeh_likely) / len(known)
    tags = (
        "depth", "depth of field", "dof", "bokeh", "blur", "background", "focus",
        "aperture", "shallow", "deep", "sharp", "lens",
    )
    if bokeh_frac >= deep_frac + 0.08:
        claim = (
            f"Your photographs {loc} often use bokeh-friendly shallow-depth conditions — "
            f"wide apertures combined with a normal or longer focal length, so a subject "
            f"can separate from its background. This is an optical proxy from the "
            f"{coverage:.0%} of frames with aperture data, not a direct measurement of "
            f"the rendered blur."
        )
        direction = "higher"
    elif deep_frac >= bokeh_frac + 0.08:
        claim = (
            f"Your photographs {loc} more often preserve deep focus than seek bokeh — "
            f"a larger share of the frames with aperture data uses settings that keep more "
            f"of the scene legible. This is an optical proxy, not a direct measurement of "
            f"the rendered blur."
        )
        direction = "lower"
    else:
        claim = (
            f"Your depth-of-field choices {loc} are mixed: wide-aperture, bokeh-friendly "
            f"conditions appear alongside deeper-focus settings. The evidence covers "
            f"{coverage:.0%} of frames and is an optical proxy rather than a direct "
            f"measurement of rendered background blur."
        )
        direction = None
    add(
        f"depth:{scope}:bokeh", "depth", claim, tags,
        direction=direction, magnitude=round(bokeh_frac, 2), era=era,
    )
    if wide_frac >= 0.25:
        add(
            f"depth:{scope}:wide-aperture", "depth",
            f"A recurring visual choice {loc} is a wide aperture — a technical setting "
            f"that can create shallow focus and a softer background when the focal length "
            f"and subject distance support it.",
            tags + ("wide aperture",), direction="higher", magnitude=round(wide_frac, 2),
            era=era,
        )


def _camera_family(row: dict) -> str | None:
    """Classify only camera families needed for grounded gear comparisons."""
    text = " ".join(str(row.get(key) or "") for key in ("camera_make", "camera_model")).lower()
    if "iphone" in text or "ipad" in text:
        return "iphone"
    # Interchangeable-lens bodies present in this archive. Keep this conservative: a
    # compact camera should not silently become a DSLR comparison point.
    if any(token in text for token in (
        "nikon d", "canon eos", "rebel xti", "canon 5d", "canon 6d",
    )):
        return "dslr"
    return None


def _camera_framing_phrase(focal: float | None) -> str:
    if focal is None:
        return "a distinct visual approach"
    if focal <= 30:
        return "a wider view that keeps more of the surrounding world in frame"
    if focal <= 60:
        return "a more natural, eye-level view"
    return "a tighter view that singles a subject out from farther away"


def _camera_nodes(add, canon) -> None:
    """Compare named camera families using persisted make/model and photo metrics."""
    groups: dict[str, list[dict]] = {"iphone": [], "dslr": []}
    for row in canon:
        family = _camera_family(row)
        if family:
            groups[family].append(row)

    tags = ("camera", "cameras", "gear", "body", "iphone", "dslr", "lens", "framing")
    stats: dict[str, tuple[int, float | None, float, int | None]] = {}
    labels = {"iphone": "iPhone", "dslr": "DSLR"}
    for family, rows in groups.items():
        if len(rows) < 20:
            continue
        focals = [r.get("focal_length_35mm") for r in rows if r.get("focal_length_35mm")]
        median_focal = _median(focals)
        people_share = _frac(rows, lambda r: (r.get("face_count") or 0) > 0)
        quadrants = [r.get("dominant_quadrant") for r in rows if r.get("dominant_quadrant")]
        dominant = max(set(quadrants), key=quadrants.count) if quadrants else None
        stats[family] = (len(rows), median_focal, people_share, dominant)
        add(
            f"camera:{family}", "camera",
            f"Your {labels[family]} photographs tend toward "
            f"{_camera_framing_phrase(median_focal)}.",
            tags + (family,), direction=None,
            magnitude=round(median_focal, 0) if median_focal is not None else None,
        )

    if "iphone" not in stats or "dslr" not in stats:
        return
    _, iphone_focal, iphone_people, _ = stats["iphone"]
    _, dslr_focal, dslr_people, _ = stats["dslr"]
    clauses: list[str] = []
    if iphone_focal is not None and dslr_focal is not None and abs(iphone_focal - dslr_focal) >= 6:
        if iphone_focal < dslr_focal:
            clauses.append(
                "the iPhone gives you a wider view, while the DSLR lets you "
                "isolate subjects more"
            )
        else:
            clauses.append(
                "the DSLR gives you a wider view, while the iPhone lets you "
                "isolate subjects more"
            )
    if abs(iphone_people - dslr_people) >= 0.08:
        if iphone_people > dslr_people:
            clauses.append("the iPhone is more often pointed at people")
        else:
            clauses.append("the DSLR is more often pointed at people")
    if not clauses:
        clauses.append(
            "the two cameras do not show a strong difference in the recorded framing "
            "and people signals"
        )
    add(
        "camera:compare:iphone-dslr", "camera",
        "The camera changes something real in your practice: " + "; ".join(clauses) + ".",
        tags + ("compare", "comparison", "between", "changed", "change"),
        direction=None,
    )


_PEOPLE_TAGS = ("people", "faces", "person", "portrait", "family", "who", "subject")


def _people_nodes(add, scope, era, where, rs, n, owner) -> None:
    people_frac = _frac(rs, lambda r: (r.get("face_count") or 0) > 0)
    if people_frac >= 0.35:
        add(f"people:{scope}:present", "people",
            f"People are central {where} — a large share of these photographs hold a "
            f"human presence rather than empty scenes.",
            _PEOPLE_TAGS, direction="higher", magnitude=round(people_frac, 2), era=era)
    # "People you photograph most" must EXCLUDE the owner: the owner appears because
    # someone else shot them, not because the owner chose them as a subject.
    top = [(name, cnt) for name, cnt in _top_people(rs, k=5)
           if not owner.is_self(name)]
    named = [name for name, cnt in top[:3] if cnt >= max(5, n * 0.02)]
    if named:
        add(f"people:{scope}:top", "people",
            f"The people you return to most {where} are {', '.join(named)}.",
            _PEOPLE_TAGS + ("recur", "most"), entities=named, era=era)


_LIGHT_TAGS = ("light", "golden", "hour", "sunset", "time", "warm", "evening",
               "lighting", "daylight", "day", "night", "dark", "darkness", "bright",
               "indoor", "nighttime", "daytime", "lit")


def _timeofday_nodes(add, scope, era, where, rs, n) -> None:
    # Golden hour needs GPS (sun position). Read it ONLY on photos that have location:
    # an unlocated photo's golden_hour=0 means UNKNOWN, not 'not golden'. Without enough
    # located photos in a period (e.g. the 2000s) we say nothing rather than invent a
    # 'daylight' habit from missing data.
    located = [r for r in rs if r.get("has_gps")]
    if len(located) >= 30:
        golden = _frac(located, lambda r: r.get("golden_hour"))
        if golden >= 0.08:
            add(f"light:{scope}:golden", "light",
                f"You reach for golden hour {where} — among the photographs with location "
                f"data, a notable share were made in that low, warm, end-of-day light.",
                _LIGHT_TAGS, direction="higher", magnitude=round(golden, 3), era=era)
        elif golden >= 0.03:
            add(f"light:{scope}:golden", "light",
                f"Golden hour is a small but steady thread {where} — among the "
                f"photographs with location data, some were made in that low, warm, "
                f"end-of-day light, though it is not where most of them were made.",
                _LIGHT_TAGS, direction="higher", magnitude=round(golden, 3), era=era)
        else:
            add(f"light:{scope}:daylight", "light",
                f"You mostly shoot {where} in ordinary daylight rather than the golden ends "
                f"of the day — among the photographs with location data, very few were "
                f"made at golden hour.",
                _LIGHT_TAGS, direction="lower", magnitude=round(golden, 3), era=era)

    # Night vs day, read from exposure value (EV): a low EV means a dark scene (night,
    # or a dim indoor room), a high EV a bright daylit one. This answers "do I prefer
    # night or day" — a dimension golden-hour alone cannot. Guard the one inf glitch.
    evs = [r["ev"] for r in rs
           if r.get("ev") is not None and math.isfinite(r["ev"])]
    if len(evs) >= 20:
        night = sum(1 for e in evs if e < 4) / len(evs)      # dark: night / dim interior
        day = sum(1 for e in evs if e >= 8) / len(evs)       # clearly daylit
        if day >= 0.45 and day - night >= 0.15:
            add(f"light:{scope}:day", "light",
                f"You are a daylight photographer {where} — most of your frames are made "
                f"in good light, the bright, legible world rather than the dark.",
                _LIGHT_TAGS, direction="higher", magnitude=round(day, 2), era=era)
        elif night >= 0.3:
            add(f"light:{scope}:night", "light",
                f"You are drawn to the dark {where} — a real share of your photographs "
                f"are made at night or in dim, low-lit rooms, not in bright daylight.",
                _LIGHT_TAGS, direction="higher", magnitude=round(night, 2), era=era)
        else:
            add(f"light:{scope}:mixedlight", "light",
                f"You shoot across the light {where} — mostly daytime, but with a real "
                f"thread of darker, night and low-lit frames too.",
                _LIGHT_TAGS, direction="higher", magnitude=round(day, 2), era=era)


def _solitude_nodes(add, scope, era, where, rs, n) -> None:
    group = _frac(rs, lambda r: (r.get("face_count") or 0) >= 3)
    alone = _frac(rs, lambda r: (r.get("face_count") or 0) == 0)
    if alone >= 0.3:
        add(f"solitude:{scope}", "solitude",
            f"A large share of your work {where} is unpeopled — landscapes, objects, "
            f"quiet spaces rather than gatherings.",
            ("solitude", "alone", "quiet", "landscape", "empty", "people", "group",
             "nature"),
            direction="higher", magnitude=round(alone, 2), era=era)
    if group >= 0.12:
        add(f"groups:{scope}", "groups",
            f"Group photographs recur {where} — you gather several people into the frame "
            f"fairly often.",
            ("groups", "gathering", "crowd", "together", "people", "social"),
            direction="higher", magnitude=round(group, 2), era=era)


_LANDSCAPE_TAGS = (
    "landscape", "landscapes", "nature", "scenery", "scene", "scenes", "outdoors",
    "vista", "vistas", "place", "places", "wild", "wilderness", "scenic", "view",
    "views", "horizon", "sky", "space",
)


def _landscape_nodes(add, scope, era, where, rs, n) -> None:
    """Describe the KINDS of landscapes/scenes the owner shoots, over the unpeopled frames.

    Landscape intent is read through the same matrix as portraits but via physical cues
    (intent doc §5): vast negative space -> Q4 open/solitary; a low horizon under big
    sky -> Q2 towering scale; warm golden light -> Q1 nostalgic; a tight, near-filled
    frame -> Q3 'intimate landscape'. Each kind is emitted only when a meaningful share
    of the scene photos show it, and a comparative node names which kind dominates so
    'what landscapes do I shoot MORE of' gets a direct answer.
    """
    scenes = [r for r in rs if (r.get("face_count") or 0) == 0]
    if len(scenes) < max(20, n * 0.1):
        return   # too few unpeopled frames here to speak about landscapes honestly
    sn = len(scenes)

    def frac(pred):
        return sum(1 for r in scenes if pred(r)) / sn

    # The landscape "kinds", each a share of the scene photos. The floors are low (8%)
    # because the question 'what do I shoot MORE of' is comparative — a kind need only be
    # a real, distinguishable minority to be worth naming alongside the dominant one. The
    # magnitudes rank them so the comparative node names the true leader.
    kinds: dict[str, tuple[float, str, str]] = {}
    open_f = frac(lambda r: (r.get("negative_space") or 0) > 0.6)
    if open_f >= 0.08:
        kinds["open"] = (open_f, "open",
            "wide, open scenes — vast skies, flat water, a single subject in lots of "
            "empty space, the kind of frame that leaves room to breathe")
    low_f = frac(lambda r: (r.get("horizon_y_ratio") is not None
                            and r["horizon_y_ratio"] < 0.3))
    if low_f >= 0.08:
        kinds["towering"] = (low_f, "towering",
            "scenes shot with a low horizon so the land or sky towers above it — "
            "reaching for scale and a sense of something larger than yourself")
    golden_f = frac(lambda r: r.get("golden_hour"))
    if golden_f >= 0.08:
        kinds["golden"] = (golden_f, "golden",
            "landscapes caught in warm, low, end-of-day light — familiar ground made "
            "golden and nostalgic")
    intimate_f = frac(lambda r: (r.get("negative_space") is not None
                                 and r["negative_space"] < 0.3))
    if intimate_f >= 0.08:
        kinds["intimate"] = (intimate_f, "intimate",
            "intimate, close scenes that fill the frame — texture and detail rather "
            "than a sweeping vista")

    tags = _LANDSCAPE_TAGS
    for share, slug, desc in kinds.values():
        add(f"landscape:{scope}:{slug}", "landscape",
            f"Among your landscapes {where}, a notable share are {desc}.",
            tags, direction="higher", magnitude=round(share, 2), era=era)

    # Comparative node: which KIND dominates (answers 'more of'). Only when we found >=2.
    if len(kinds) >= 2:
        ranked = sorted(kinds.values(), key=lambda t: -t[0])
        top_desc = ranked[0][2]
        add(f"landscape:{scope}:dominant", "landscape",
            f"Of the landscapes you shoot {where}, you lean most toward {top_desc}.",
            tags, direction="higher", magnitude=round(ranked[0][0], 2), era=era)


_STYLE_TAGS = ("style", "feeling", "mood", "emotion", "quadrant", "vibe", "tone")


def _scene_set(r) -> set[str]:
    """The Vision scene labels present on one photo above the confidence bar."""
    raw = r.get("scene_labels")
    if not raw:
        return set()
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return set()
    out = set()
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and item[1] >= _SUBJECT_CONF:
            out.add(item[0])
    return out


def _subject_nodes(add, scope, era, where, rs, n) -> None:
    """WHAT the owner photographs — the recurring subjects, from Vision scene labels.

    The intent doc is about technique AS emotional posture, but WHAT is in the frame is
    half the story (a baby on a beach is a subject, not just a focal length). Vision
    already classified every photo; we persist those labels and here finally turn them
    into content insight. Curated groups only — Vision junk ('structure', 'textile') is
    never a subject — and share-gated so a rare thing does not read as a theme.
    """
    withlabels = [r for r in rs if _scene_set(r)]
    if len(withlabels) < 30:
        return
    sets = [_scene_set(r) for r in withlabels]
    m = len(sets)
    ranked: list[tuple[float, str, str, tuple[str, ...]]] = []
    for slug, tags, labels, phrase in _SUBJECT_GROUPS:
        share = sum(1 for s in sets if s & labels) / m
        if share >= 0.12:
            ranked.append((share, slug, phrase, tags))
            add(f"subject:{scope}:{slug}", "subject",
                f"You often photograph {phrase} {where}.",
                _SUBJECT_TAGS + tags, direction="higher",
                magnitude=round(share, 2), era=era)
    # A comparative lead: the single most-photographed subject, for "what do I shoot most".
    if ranked:
        ranked.sort(key=lambda t: -t[0])
        top_share, top_slug, top_phrase, top_tags = ranked[0]
        add(f"subject:{scope}:dominant", "subject",
            f"More than anything else {where}, your camera turns to {top_phrase}.",
            _SUBJECT_TAGS + top_tags, direction="higher",
            magnitude=round(top_share, 2), era=era)


# Shared golden-hour vocabulary. Subject/person vocabulary is added per family, so a
# question about SUBJECTS matches subject nodes and not every person node.
_CONJ_TAGS = ("golden", "hour", "sunset", "light", "warm", "evening", "most", "what",
              "often")


def _golden_conjunction_nodes(add, canon, created, owner) -> None:
    """CROSS-DIMENSION lift: what appears disproportionately in your golden-hour frames.

    Each dimension alone (subject, person, light) can be true and still miss the point; the
    intent doc's thesis is that technique and subject TOGETHER reveal intent. Here we ask:
    of the golden-hour photos, which subjects and which people are over-represented
    compared with your archive overall? A lift (share-in-golden / share-overall) above a
    real bar becomes a node. Counts are gated so a handful of frames never becomes a theme.
    """
    from collections import Counter

    # Golden hour is only observable on located photos, so the whole comparison runs on
    # that population (otherwise every unlocated photo dilutes the golden share to zero).
    located = [r for r in canon if r.get("has_gps")]
    gh = [r for r in located if r.get("golden_hour")]
    n_all = len(located)
    if len(gh) < 30 or n_all == 0:
        return
    base = len(gh) / n_all

    # Subject lift: nature, water, etc. over-indexed in golden light.
    for slug, tags, labels, phrase in _SUBJECT_GROUPS:
        g = sum(1 for r in gh if _scene_set(r) & labels)
        a = sum(1 for r in located if _scene_set(r) & labels)
        if g < 10 or a == 0:
            continue
        lift = (g / len(gh)) / (a / n_all)
        if lift >= 1.3:
            add(f"golden:subject:{slug}", "conjunction",
                f"When the light turns golden, you reach for {phrase} — that subject "
                f"leans into your golden-hour frames more than it does in the rest of your "
                f"photographs.",
                _CONJ_TAGS + _SUBJECT_TAGS + tags, direction="higher",
                magnitude=round(lift, 2))

    # Person lift: who turns up disproportionately in golden frames (owner excluded).
    pc_all: Counter = Counter()
    pc_gh: Counter = Counter()
    for r in located:
        pc_all.update(set(_names(r)))
    for r in gh:
        pc_gh.update(set(_names(r)))
    # Frequency is a different question from lift: surface the people who actually appear
    # most often in golden frames before any over-representation claims.
    frequent = [(name, count) for name, count in pc_gh.most_common(3)
                if not owner.is_self(name) and count >= 10]
    if frequent:
        names = [name for name, _count in frequent]
        add(
            "golden:people:frequent", "conjunction",
            f"The people who appear most often in your located golden-hour frames are "
            f"{', '.join(names[:-1])}{' and ' if len(names) > 1 else ''}{names[-1]}; "
            "this is frequency, not a claim that they are disproportionately golden-hour "
            "subjects.",
            _CONJ_TAGS + ("people", "who", "most", "frequent", "frequency"),
            entities=tuple(names), direction="higher", magnitude=float(frequent[0][1]),
        )

    for name, total in pc_all.most_common(15):
        if owner.is_self(name) or total < 20:
            continue
        g = pc_gh.get(name, 0)
        if g < 10:
            continue
        lift = (g / total) / base
        # A modest lift on a tiny denominators is often a one-off trip. Require either
        # durable support (>=200 located frames and >=20 golden frames) or an unusually
        # strong lift, as in the small synthetic/test case.
        supported = (
            (total >= 1000 and g >= 50 and lift >= 1.5)
            or (g / total >= 0.4 and g >= 10 and lift >= 1.5)
        )
        if supported:
            add(f"golden:person:{name}".replace(" ", "_"), "conjunction",
                f"{name} shows up in your golden-hour frames far more than in your other "
                f"light — they belong to the warm, end-of-day photographs.",
                _CONJ_TAGS + ("people", "who", "with", "family"), entities=(name,),
                direction="higher", magnitude=round(lift, 2))


def _quadrant_nodes(add, scope, era, where, rs, n) -> None:
    from memory_graph.narration.meaning import quadrant_meaning

    counts = {q: 0 for q in range(1, 6)}
    total = 0
    for r in rs:
        q = r.get("dominant_quadrant")
        if q in counts:
            counts[q] += 1
            total += 1
    if total == 0:
        return
    ranked = sorted(counts, key=lambda q: -counts[q])
    top = ranked[0]
    add(f"style:{scope}", "style",
        f"The dominant feeling {where} is {_QUADRANT_NAMES[top].lower()} — "
        f"{quadrant_meaning(top)}.",
        _STYLE_TAGS, direction="higher", magnitude=round(counts[top] / total, 2),
        era=era)
    # Full distribution: the SECOND feeling is often the richer insight (a solitary
    # photographer whose second voice is tender tells a fuller story than #1 alone).
    second = ranked[1]
    if counts[second] / total >= 0.15:
        add(f"style:{scope}:second", "style",
            f"A strong secondary current {where} is {_QUADRANT_NAMES[second].lower()} — "
            f"{quadrant_meaning(second)}.",
            _STYLE_TAGS, direction="higher", magnitude=round(counts[second] / total, 2),
            era=era)


def _change_nodes(add, by_era, eras) -> None:
    labels = [er.label for er in eras.ranges if er.label in by_era]
    if len(labels) < 2:
        return
    first, last = by_era[labels[0]], by_era[labels[-1]]
    span = f"{labels[0]} to {labels[-1]}"
    change_tags = ("change", "over time", "evolved", "changed", "decade", "journey",
                   "years", "history", "shift")

    kf = _frac(first, lambda r: r.get("known_person"))
    kl = _frac(last, lambda r: r.get("known_person"))
    if abs(kl - kf) >= 0.1:
        add("change:people", "change",
            f"Over the {span} your photography turned toward the people you love — the "
            f"share holding a recognized loved one rose markedly.",
            change_tags + ("people", "family"),
            direction="higher" if kl > kf else "lower", magnitude=round(kl - kf, 2))

    gf = _frac(first, lambda r: (r.get("face_count") or 0) >= 3)
    gl = _frac(last, lambda r: (r.get("face_count") or 0) >= 3)
    if abs(gl - gf) >= 0.05:
        add("change:groups", "change",
            f"Over the {span} your big-group photographs gave way to closer, more "
            f"intimate moments with a few people.",
            change_tags + ("groups", "intimacy", "people"),
            direction="higher" if gl > gf else "lower", magnitude=round(gl - gf, 2))

    mf = _median([r["focal_length_35mm"] for r in first if r.get("focal_length_35mm")])
    ml = _median([r["focal_length_35mm"] for r in last if r.get("focal_length_35mm")])
    if mf and ml and abs(ml - mf) >= 6:
        add("change:framing", "change",
            f"Over the {span} your framing distance shifted — the way you place yourself "
            f"relative to a subject changed.",
            change_tags + ("framing", "focal", "lens", "distance"),
            direction="higher" if ml > mf else "lower", magnitude=round(ml - mf, 1))


_REL_TAGS = ("people", "who", "relationship", "together", "with", "family", "friends",
             "appears", "appeared", "whom", "entered", "left", "began", "stopped")


def _relationship_nodes(add, canon, by_era, eras, created, owner) -> None:
    """Who appears WITH whom, and who entered or left your frame when.

    The intent doc is about WHO the photography is for. These nodes read the named-people
    graph: pairings (two people photographed together most), and the arrival/departure of
    a face across eras (a person who became central, or faded out). Names used here are
    above the Library recognition bar, so they are permitted narration entities.
    """
    from collections import Counter
    from itertools import combinations

    # --- Co-occurrence: which pairs of people share your frames most. ---
    pair_counts: Counter = Counter()
    person_counts: Counter = Counter()
    for r in canon:
        ppl = sorted(set(_names(r)))
        person_counts.update(ppl)
        for a, b in combinations(ppl, 2):
            pair_counts[(a, b)] += 1
    for (a, b), cnt in pair_counts.most_common(8):
        if cnt < 8:
            break
        a_self, b_self = owner.is_self(a), owner.is_self(b)
        if a_self and b_self:
            continue
        if a_self or b_self:
            # A pairing WITH the owner: the owner is IN the frame (shot by someone else),
            # so it is "you appear alongside X", not "you photographed X and Y together".
            other = b if a_self else a
            add(f"relationship:with_owner:{other}".replace(" ", "_"), "relationship",
                f"You appear alongside {other} often — the two of you are photographed "
                f"together again and again, a bond someone kept returning to capture.",
                _REL_TAGS + ("alongside", "me", "myself"), direction="higher",
                magnitude=float(cnt), entities=(other,))
        else:
            add(f"relationship:pair:{a}|{b}".replace(" ", "_"), "relationship",
                f"{a} and {b} appear together often in your photographs — a pairing you "
                f"return to, not a one-off.",
                _REL_TAGS, direction="higher", magnitude=float(cnt), entities=(a, b))

    # --- Per-person life trajectory across the eras (arrived / faded / peaked / etc). ---
    _trajectory_nodes(add, by_era, eras, person_counts, owner)


# Minimum NAMED-photo population for an era's share to be trusted (face-naming is sparse
# in the earliest era, so a 56%-of-9 reading is noise, not signal).
_ERA_MIN_NAMED = 40
# A share at/above this reads as a REAL presence in that era — low, because a secondary
# person at 4-6% of a busy era is still a meaningful recurring face whose rise/peak/fade
# is a true life-chapter signal. 'steady' additionally requires real dominance (see the
# _STEADY_SHARE gate) so a minor face is never called a 'constant presence'.
_PRESENT_SHARE = 0.03
_STEADY_SHARE = 0.2


def _trajectory_nodes(add, by_era, eras, person_counts, owner) -> None:
    """Classify each recurring person's per-era SHARE sequence into a life-story shape.

    The earlier version only compared the first vs the last era, so it saw 'arrived' and
    'faded' but missed people who PEAKED in a middle era and receded, or who ROSE steadily
    — exactly the shapes in this archive (a relative prominent only in the late 2010s; a
    new friend growing into the 2020s). This reads the whole trajectory and names the
    shape, which yields one true node per recurring person instead of a single endpoint
    comparison. Shares are only trusted in eras with enough named photos.
    """
    from collections import Counter

    labels = [er.label for er in eras.ranges if er.label in by_era]
    per_era: dict[str, Counter] = {}
    totals: dict[str, int] = {}
    for lab in labels:
        c: Counter = Counter()
        for r in by_era[lab]:
            for name in set(_names(r)):
                c[name] += 1
        per_era[lab] = c
        totals[lab] = sum(c.values())

    # Only eras with enough named photos carry a trustworthy share.
    trusted = [lab for lab in labels if totals[lab] >= _ERA_MIN_NAMED]
    if len(trusted) < 2:
        return

    # A person earns their OWN trajectory sentence only if they reach a real presence in
    # some era (not a 2%->5% blip). Minor recurring faces are true but not individually
    # weighty — giving each a full "a bond deepening over the years" sentence
    # over-weights a friend-of-a-friend next to a spouse. Those below the bar are
    # collected into ONE "supporting cast" aside per shape, so they are acknowledged
    # without being inflated.
    minor: dict[str, list[str]] = {}
    for name, total in person_counts.most_common(25):
        if total < 20 or owner.is_self(name):
            continue
        shares = [(lab, per_era[lab].get(name, 0) / totals[lab]) for lab in trusted]
        shape = _classify_trajectory([s for _, s in shares])
        if shape is None:
            continue
        peak_share = max(s for _, s in shares)
        kind_slug, direction, claim, mag_lab = _trajectory_claim(name, shape, shares)
        peak_era = max(shares, key=lambda ls: ls[1])[0]
        # The individual-sentence bar: a steady person must be dominant (already gated in
        # _classify_trajectory); a rising/arrived/faded/peaked person must have actually
        # become a REAL presence (>= 15% of an era), else they are supporting cast.
        major = shape == "steady" or peak_share >= 0.15
        if major:
            add(f"relationship:{kind_slug}:{name}".replace(" ", "_"), "relationship",
                claim, _REL_TAGS + _TRAJECTORY_TAGS[shape],
                direction=direction, magnitude=round(mag_lab, 2),
                entities=(name,),
                era=peak_era if shape in ("peaked", "arrived") else None)
        else:
            minor.setdefault(shape, []).append(name)

    _supporting_cast_nodes(add, minor)


# How a shape reads when collapsed into a single "supporting cast" aside.
_CAST_PHRASING = {
    "arrived": ("newer", "friends and family who joined your frames along the way"),
    "rising": ("newer", "friends and family who joined your frames along the way"),
    "faded": ("earlier", "faces that were around earlier and drifted out"),
    "peaked": ("earlier", "faces tied to one passing chapter"),
}


def _supporting_cast_nodes(add, minor: dict[str, list[str]]) -> None:
    """Collapse the minor recurring faces of each shape into ONE aside node, so they are
    acknowledged as a 'supporting cast' without each getting a spouse-weight sentence."""
    # Merge arrived+rising (both "newer") and faded+peaked (both "earlier/gone").
    newer = sorted(set(minor.get("arrived", []) + minor.get("rising", [])))
    earlier = sorted(set(minor.get("faded", []) + minor.get("peaked", [])))
    if newer:
        add("relationship:cast:newer", "relationship",
            f"A rotating supporting cast also moves through your frames — "
            f"{', '.join(newer)} — friends and wider family who came into the picture "
            f"along the way, present but not at the center.",
            _REL_TAGS + ("others", "friends", "cast", "minor", "new", "came"),
            entities=tuple(newer))
    if earlier:
        add("relationship:cast:earlier", "relationship",
            f"Others passed through earlier and thinned out — {', '.join(earlier)} — "
            f"faces tied to particular times rather than the whole span.",
            _REL_TAGS + ("others", "friends", "cast", "minor", "faded", "went"),
            entities=tuple(earlier))


# Per-shape retrieval tags — the words people actually use to ASK about a trajectory
# ("who came and went", "who's been constant", "who's new"), so a life-story question
# pulls the right shapes and not just any node that happens to mention people.
_TRAJECTORY_TAGS: dict[str, tuple[str, ...]] = {
    "arrived": ("came", "arrived", "new", "entered", "joined", "appeared", "recent",
                "changed", "over time", "years"),
    "faded": ("went", "left", "gone", "faded", "disappeared", "lost", "drifted",
              "came and went", "changed", "over time", "years"),
    "peaked": ("came", "went", "left", "gone", "phase", "chapter", "period", "season",
               "came and went", "changed", "over time", "years"),
    "steady": ("constant", "always", "steady", "throughout", "consistent", "stayed",
               "center", "core", "years"),
    "rising": ("new", "growing", "more", "lately", "recent", "rising", "changed",
               "over time", "years"),
}


def _classify_trajectory(shares: list[float]) -> str | None:
    """Name the shape of a share sequence across trusted eras.

    Shapes: 'arrived' (absent early, present late), 'faded' (present early, gone late),
    'peaked' (rises to a middle peak then recedes), 'rising' (monotone-ish growth),
    'steady' (consistently present and prominent throughout). None if no clear shape.
    """
    if not shares:
        return None
    first, last = shares[0], shares[-1]
    peak = max(shares)
    peak_i = shares.index(peak)
    present = [s >= _PRESENT_SHARE for s in shares]
    if peak < _PRESENT_SHARE:
        return None   # never really prominent anywhere — not worth a node
    absent_early = first < _PRESENT_SHARE / 2
    absent_late = last < _PRESENT_SHARE / 2
    if absent_early and not absent_late:
        return "arrived"
    if not absent_early and absent_late:
        return "faded"
    # Peaked: a clear interior maximum that then recedes to well below the peak.
    if 0 < peak_i < len(shares) - 1 and last <= peak * 0.4 and first <= peak * 0.6:
        return "peaked"
    if all(present) and min(shares) >= _PRESENT_SHARE and peak >= _STEADY_SHARE:
        return "steady"
    if last >= first * 1.5 and last >= _PRESENT_SHARE:
        return "rising"
    return None


def _trajectory_claim(name, shape, shares):
    """The feeling-framed claim, slug, direction, and magnitude for a trajectory shape."""
    labels = [lab for lab, _ in shares]
    peak_lab, peak_share = max(shares, key=lambda ls: ls[1])
    last_lab = labels[-1]
    if shape == "arrived":
        return ("arrived", "higher",
                f"{name} is absent from your earlier frames and becomes one of the faces "
                f"you keep by the {last_lab} — someone who arrived and moved to the "
                f"center of your photography.", shares[-1][1])
    if shape == "faded":
        return ("faded", "lower",
                f"{name} was a frequent presence in your earlier photographs and fades "
                f"from the frame over time — a face that drifted out of your pictures.",
                shares[0][1])
    if shape == "peaked":
        return ("peaked", "higher",
                f"{name} rises to the center of your photographs around the {peak_lab} "
                f"and then recedes — a face tied to one chapter of your life rather than "
                f"all of it.", peak_share)
    if shape == "steady":
        return ("steady", "higher",
                f"{name} runs steadily through your photographs across the years — a "
                f"constant presence, the kind of person your camera always comes back "
                f"to.", peak_share)
    # rising
    return ("rising", "higher",
            f"{name} grows more present in your photographs over time, more frequent by "
            f"the {last_lab} than before — a bond deepening as the years pass.",
            shares[-1][1])


_SIG_TAGS = ("signature", "how", "style", "shoot", "photograph", "framing", "closeness",
             "candid", "warm", "portrait", "people", "who")


def _signature_nodes(add, canon, created, owner) -> None:
    """Cross-dimension SIGNATURE: how your technique shifts for a specific person.

    This is the intent doc's core thesis made explicit — "you lowered the camera and let
    the background dissolve" is framing + distance + subject together. For each person you
    photograph enough, we compare their photos against your baseline on three joinable
    axes (we have no aperture): how CLOSE you frame (focal + face ratio), how WARM the
    light, and how CANDID vs posed. A difference that stands out becomes a node: the way
    you photograph THEM differs from how you photograph in general.
    """
    named = [r for r in canon if _names(r)]
    if len(named) < 40:
        return

    def avg(rs, col):
        vals = [r[col] for r in rs if r.get(col) is not None]
        return sum(vals) / len(vals) if vals else None

    base_focal = _median([r["focal_length_35mm"] for r in named
                          if r.get("focal_length_35mm")])
    base_warm = avg(named, "warm_cool")
    base_candid = avg(named, "candid_score")
    base_faceratio = avg(named, "face_ratio")

    from collections import Counter
    counts: Counter = Counter()
    for r in named:
        counts.update(set(_names(r)))

    for name, cnt in counts.most_common(10):
        if cnt < 20:
            break
        if owner.is_self(name):
            # A "signature" for the owner would describe how OTHERS framed the owner, and
            # wrongly attribute their choices to the owner. Never generate it.
            continue
        theirs = [r for r in named if name in _names(r)]
        traits: list[str] = []

        pf = _median([r["focal_length_35mm"] for r in theirs
                      if r.get("focal_length_35mm")])
        pfr = avg(theirs, "face_ratio")
        if pf and base_focal:
            if pf >= base_focal * 1.25:
                traits.append("you reach in tighter, singling them out")
            elif pf <= base_focal * 0.8:
                traits.append("you pull wider, keeping them in their world")
        if pfr and base_faceratio and pfr >= base_faceratio * 1.3:
            traits.append("you fill more of the frame with their face")

        pw = avg(theirs, "warm_cool")
        if pw is not None and base_warm is not None and pw - base_warm >= 4:
            traits.append("the light around them runs warmer")
        elif pw is not None and base_warm is not None and base_warm - pw >= 4:
            traits.append("the light around them runs cooler")

        pc = avg(theirs, "candid_score")
        if pc is not None and base_candid is not None and pc - base_candid >= 0.1:
            traits.append("you catch them more unposed and candid")
        elif pc is not None and base_candid is not None and base_candid - pc >= 0.1:
            traits.append("they tend to be posed, looking at the camera")

        if traits:
            joined = traits[0] if len(traits) == 1 else (
                ", ".join(traits[:-1]) + ", and " + traits[-1])
            add(f"signature:{name}".replace(" ", "_"), "signature",
                f"The way you photograph {name} has its own signature — {joined}. The "
                f"technique bends to the person, not the other way around.",
                _SIG_TAGS, entities=(name,), magnitude=float(cnt))


def _self_presence_nodes(add, canon, by_era, eras, owner) -> None:
    """The owner's OWN presence — photos OF the owner, taken by others.

    This is the mirror of every other people node: not 'who you photographed' but 'how
    often you were photographed', which is an emotionally real fact (being documented by
    the people around you). It is a DIFFERENT claim than authorship, and only exists when
    the owner is configured. Framed in the second person but about being a subject.
    """
    if not owner.self_names:
        return
    named = [r for r in canon if _names(r)]
    if not named:
        return
    self_photos = [r for r in named if any(owner.is_self(nm) for nm in _names(r))]
    share = len(self_photos) / len(named)
    if len(self_photos) < 15:
        return
    # Tags use CONCEPT words that survive tokenization (pronouns like 'i'/'me' are
    # stopwords, so a self-question keys off 'own', 'myself', 'self', 'photographed',
    # 'subject', and the owner's own name — added below).
    tags = ("myself", "self", "own", "photographed", "appear", "appears", "subject",
            "pictured", "seen") + tuple(
        t.lower() for nm in owner.self_names for t in nm.split())
    add("self:presence", "people",
        "You are a frequent subject yourself — a real share of your people photographs "
        "are of you, taken by others. These are the moments someone else wanted to keep "
        "you in, the record of being seen rather than seeing.",
        tags, direction="higher", magnitude=round(share, 2),
        entities=tuple(owner.self_names))

    # Who most often photographed WITH you (the people you are pictured beside) — a
    # tender companion to the authored 'people you photograph most'.
    from collections import Counter
    beside: Counter = Counter()
    for r in self_photos:
        for nm in set(_names(r)):
            if not owner.is_self(nm):
                beside[nm] += 1
    top = [nm for nm, c in beside.most_common(3) if c >= 8]
    if top:
        add("self:beside", "people",
            f"When you are in the frame, the ones most often beside you are "
            f"{', '.join(top)} — the people whose life with you someone chose to record.",
            tags + ("alongside", "with", "together"), entities=tuple(top))


_ANONYMOUS_TAGS = (
    "anonymous", "unnamed", "recurring", "people", "faces", "human", "presence",
    "style", "over time",
)


def _anonymous_people_nodes(add, scope, era, where, rs, n) -> None:
    """Describe recurring unnamed human presence without inventing a family/name."""
    cluster_photos: dict[str, set[str]] = {}
    for row in rs:
        raw = row.get("anonymous_cluster_ids")
        if not raw:
            continue
        try:
            cluster_ids = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        for cluster_id in cluster_ids or []:
            if cluster_id:
                cluster_photos.setdefault(str(cluster_id), set()).add(str(row["id"]))
    recurring = [photos for photos in cluster_photos.values() if len(photos) >= 2]
    if not recurring:
        return
    photos_with_recurrence = len(set().union(*recurring))
    share = photos_with_recurrence / n if n else 0.0
    if share < 0.05:
        return
    loc = "across your photographs" if scope == "overall" else where
    largest = max(len(photos) for photos in recurring)
    add(
        f"anonymous_people:{scope}:recurring", "anonymous_people",
        f"Recurring unnamed people appear in {photos_with_recurrence} photos {loc}; "
        f"the strongest recurring presence spans {largest} frames. This is a pattern "
        "of human connection, not a claim about anyone's identity.",
        _ANONYMOUS_TAGS,
        direction="higher", magnitude=round(share, 2), era=era,
    )


_UNKNOWN_TAGS = ("unnamed", "unknown", "strangers", "people", "faces", "others",
                 "crowd", "crowds", "acquaintances", "who", "whom")


def _unknown_presence_nodes(add, scope, era, where, rs, n) -> None:
    """How much of the PEOPLE photography is of faces that are not named.

    We cannot identify unnamed faces across photos (no per-face clustering in V1 — that is
    project B), but we CAN say, truthfully and in aggregate, how large the unnamed world
    is: whether these photographs are mostly of the specific people you name, or of a
    wider field of faces you never tagged. The claim says 'unnamed', never 'strangers' —
    an untagged face may be an unlabeled loved one, especially in older eras where Apple's
    face-naming is sparse.
    """
    people = [r for r in rs if (r.get("face_count") or 0) > 0]
    if len(people) < 30:
        return   # too few people-photos here to characterize the named/unnamed balance
    pn = len(people)
    unnamed_only = sum(1 for r in people if not _names(r))
    mixed = sum(1 for r in people
                if _names(r) and (r.get("face_count") or 0) > len(_names(r)))
    unnamed_frac = unnamed_only / pn

    # A meaningful share of people-photos are of faces you never named.
    # Phrase the location without doubling "your photographs" (the overall `where` is
    # already "across your photographs").
    loc = "overall" if scope == "overall" else where
    if unnamed_frac >= 0.25:
        add(f"unknown_people:{scope}:unnamed", "unknown_people",
            f"A large share of the people you photographed {loc} are faces you never "
            f"named — a wider world of people beyond your tagged inner circle, not only "
            f"the few you return to.",
            _UNKNOWN_TAGS, direction="higher", magnitude=round(unnamed_frac, 2), era=era)

    # You often photograph your named people AMONG others (gatherings, events).
    mixed_frac = mixed / pn
    if mixed_frac >= 0.1:
        add(f"unknown_people:{scope}:amongst", "unknown_people",
            f"You often photograph the people you know {loc} among others — your named "
            f"faces set within a larger group rather than alone in the frame.",
            _UNKNOWN_TAGS + ("group", "gathering", "together", "amongst", "crowd"),
            direction="higher", magnitude=round(mixed_frac, 2), era=era)


def _unknown_trend_nodes(add, by_era, eras) -> None:
    """How the named/unnamed balance SHIFTED — the arc from a wide field of faces toward
    the specific people named. A different lens on the same narrowing the people/change
    nodes describe."""
    labels = [er.label for er in eras.ranges if er.label in by_era]
    if len(labels) < 2:
        return

    def unnamed_share(rs):
        people = [r for r in rs if (r.get("face_count") or 0) > 0]
        if len(people) < 30:
            return None
        return sum(1 for r in people if not _names(r)) / len(people)

    # Use the first and last eras that have enough people-photos to trust the share.
    trusted = [(lab, unnamed_share(by_era[lab])) for lab in labels]
    trusted = [(lab, s) for lab, s in trusted if s is not None]
    if len(trusted) < 2:
        return
    (first_lab, first_s), (last_lab, last_s) = trusted[0], trusted[-1]
    if abs(last_s - first_s) < 0.15:
        return
    if last_s < first_s:
        add("unknown_people:trend", "unknown_people",
            f"Over the {first_lab} to {last_lab} your people photography narrowed from a "
            f"wide field of unnamed faces toward the specific people you name — the "
            f"circle tightened around the ones you chose to hold.",
            _UNKNOWN_TAGS + ("change", "over time", "narrowed", "evolved", "years"),
            direction="lower", magnitude=round(last_s - first_s, 2))
    else:
        add("unknown_people:trend", "unknown_people",
            f"Over the {first_lab} to {last_lab} more of your people photographs became "
            f"of faces you did not name — your lens widened to a larger world of people.",
            _UNKNOWN_TAGS + ("change", "over time", "widened", "evolved", "years"),
            direction="higher", magnitude=round(last_s - first_s, 2))


_TECH_TAGS = (
    "technical", "settings", "camera", "exposure", "iso", "shutter", "motion",
    "horizon", "angle", "tilt", "negative space", "composition", "sharpness",
    "softness", "blur", "candid", "posed", "focus",
)


def _promotion_technical_nodes(add, scope, era, where, rs) -> None:
    """Promote stored technical signals that are otherwise only router inputs."""
    known_iso = [r.get("iso") for r in rs if r.get("iso")]
    if len(known_iso) >= 30:
        high = sum(1 for value in known_iso if value >= 800) / len(known_iso)
        add(
            f"exposure:{scope}:iso", "exposure",
            f"ISO choices {where} range from ordinary light to deliberate low-light work; "
            f"higher-ISO frames form a recurring part of the record.",
            _TECH_TAGS + ("low light", "noise"), direction="higher",
            magnitude=round(high, 2), era=era,
        )

    horizons = [r.get("horizon_tilt_deg") for r in rs if r.get("horizon_tilt_deg") is not None]
    if len(horizons) >= 30:
        tilted = sum(1 for value in horizons if abs(value) >= 3.0) / len(horizons)
        add(
            f"geometry:{scope}:horizon", "geometry",
            f"Your horizons {where} are mostly controlled rather than dramatically tilted; "
            f"a smaller thread deliberately cants the frame for tension.",
            _TECH_TAGS + ("dutch angle", "geometry"), direction="higher",
            magnitude=round(tilted, 2), era=era,
        )

    spaces = [r.get("negative_space") for r in rs if r.get("negative_space") is not None]
    if len(spaces) >= 30:
        open_share = sum(1 for value in spaces if value >= 0.6) / len(spaces)
        add(
            f"geometry:{scope}:negative-space", "geometry",
            f"Open space is a recurring compositional choice {where}; some frames leave "
            f"the subject room rather than filling every edge.",
            _TECH_TAGS + ("open", "space", "solitude"), direction="higher",
            magnitude=round(open_share, 2), era=era,
        )

    sharp = [r.get("sharpness") for r in rs if r.get("sharpness") is not None]
    if len(sharp) >= 30:
        med = _median(sharp)
        soft_share = sum(1 for value in sharp if value < 170) / len(sharp)
        add(
            f"quality:{scope}:sharpness", "quality",
            f"Your archive {where} includes a meaningful soft or motion-affected thread; "
            f"technical sharpness is not the only thing your photographs preserve.",
            _TECH_TAGS + ("image quality", "motion blur"), direction="higher",
            magnitude=round(soft_share, 2), era=era,
        )
        _ = med

    candid = [r.get("candid_score") for r in rs if r.get("candid_score") is not None]
    if len(candid) >= 30:
        candid_share = sum(1 for value in candid if value >= 0.55) / len(candid)
        add(
            f"quality:{scope}:candid", "quality",
            f"Candidness is present {where}; a substantial share of the detected faces are "
            f"turned, unposed, or caught in motion rather than clearly formal portraits.",
            _TECH_TAGS + ("unposed", "natural", "gaze"), direction="higher",
            magnitude=round(candid_share, 2), era=era,
        )


def _model_slug(model: str) -> str:
    return "_".join(model.lower().replace("/", " ").split())[:70]


def _promotion_aggregate_nodes(add, canon, by_era, owner) -> None:
    """Promote camera, season, favorite, resonance, and taste comparisons."""
    # Individual models and camera-by-era presence.
    model_counts: Counter = Counter(
        str(r.get("camera_model")).strip() for r in canon if r.get("camera_model")
    )
    for model, count in model_counts.most_common(8):
        if count < 30:
            continue
        add(
            f"camera:model:{_model_slug(model)}", "camera",
            f"{model} is one of the bodies that shaped your photographic record; it has "
            f"a substantial body of work behind it.",
            ("camera", "gear", "body", "model", "equipment", model.lower()),
            direction="higher", magnitude=float(count), entities=(),
        )

    families = {"iphone": [], "dslr": []}
    for row in canon:
        family = _camera_family(row)
        if family:
            families[family].append(row)
    for era, rows in by_era.items():
        era_ids = {id(row) for row in rows}
        for family, family_rows in families.items():
            count = sum(1 for row in family_rows if id(row) in era_ids)
            if count < 30:
                continue
            add(
                f"camera:{family}:{era}", "camera",
                f"The {family.upper() if family == 'dslr' else 'iPhone'} is a meaningful part "
                f"of your photographic voice in the {era}.",
                ("camera", "gear", "era", family, era.lower()),
                direction="higher", magnitude=float(count), era=era,
            )

    # Season/month distribution is directly available from timestamps; no image reread.
    months: Counter = Counter()
    for row in canon:
        if row.get("timestamp"):
            months[datetime.fromtimestamp(row["timestamp"], UTC).month] += 1
    if months:
        season = {
            12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn",
        }
        by_season = Counter(season[m] for m in months for _ in range(months[m]))
        top = by_season.most_common(1)[0][0]
        add(
            "season:overall:peak", "season",
            f"Your archive has its strongest seasonal concentration in {top}; the rhythm "
            f"of when you photograph is part of the record too.",
            ("season", "month", "calendar", "when", top), direction="higher",
            magnitude=round(by_season[top] / sum(by_season.values()), 2),
        )

    # Camera x subject and camera x people comparisons.
    for family, family_rows in families.items():
        if len(family_rows) < 30:
            continue
        people = _frac(family_rows, lambda r: (r.get("face_count") or 0) > 0)
        labels = Counter()
        for row in family_rows:
            labels.update(_scene_set(row))
        top_subject = labels.most_common(1)[0][0] if labels else None
        if top_subject:
            add(
                f"camera:{family}:subject", "camera",
                f"With the {family.upper() if family == 'dslr' else 'iPhone'}, your most recurring "
                f"content signal is {top_subject}; people are present in a substantial share "
                f"of those frames.",
                ("camera", "subject", "people", "gear", family, top_subject),
                direction="higher", magnitude=round(people, 2),
            )

    # Resonance comparisons use the already-computed cached score; no taste retraining.
    family_scores: dict[str, list[float]] = {"iphone": [], "dslr": []}
    for row in canon:
        family = _camera_family(row)
        if family and row.get("resonance") is not None:
            family_scores[family].append(float(row["resonance"]))
    if all(family_scores.values()):
        averages = {family: sum(values) / len(values) for family, values in family_scores.items()}
        leader = max(averages, key=averages.get)
        other = "dslr" if leader == "iphone" else "iphone"
        leader_label = "DSLR" if leader == "dslr" else "iPhone"
        other_label = "DSLR" if other == "dslr" else "iPhone"
        add(
            "resonance:camera:iphone-dslr", "resonance",
            f"Among the camera families in your archive, the {leader_label} has the "
            f"stronger current memory-resonance signal than the {other_label}. This "
            "describes the engine's ranking signal, not a final truth about your taste.",
            ("resonance", "memory", "keep", "camera", "iphone", "dslr", "gear"),
            direction="higher", magnitude=round(averages[leader] - averages[other], 3),
        )

    # Construct-level resonance: bokeh-friendly proxy versus other frames.
    bokeh = [r for r in canon if r.get("aperture") is not None and r["aperture"] <= 2.8
             and ((r.get("focal_length_35mm") or 0) >= 50 or (r.get("face_ratio") or 0) >= 0.12)]
    bokeh_ids = {id(r) for r in bokeh}
    other = [r for r in canon if id(r) not in bokeh_ids and r.get("resonance") is not None]
    bokeh_scores = [r["resonance"] for r in bokeh if r.get("resonance") is not None]
    other_scores = [r["resonance"] for r in other]
    if len(bokeh_scores) >= 30 and len(other_scores) >= 30:
        delta = sum(bokeh_scores) / len(bokeh_scores) - sum(other_scores) / len(other_scores)
        add(
            "resonance:construct:bokeh", "resonance",
            f"The engine's current memory signal is {'higher' if delta > 0 else 'not higher'} for "
            f"bokeh-friendly shallow-depth conditions than for other frames. This is a ranking "
            f"comparison, not a claim that bokeh causes the memory response.",
            ("resonance", "memory", "keep", "bokeh", "depth", "aperture", "construct"),
            direction="higher" if delta > 0 else "lower", magnitude=round(delta, 3),
        )

    taste = [r.get("taste_score") for r in canon if r.get("taste_score") is not None]
    favorites = [r for r in canon if r.get("favorite")]
    if taste:
        add(
            "taste:explanation", "taste",
            "The memory model learns from your explicit Favorite choices and estimates "
            "which visual patterns resemble photos you chose to keep. It is a return-likelihood "
            "signal, not an oracle of what a photograph means or what you felt.",
            ("taste", "favorite", "memory", "model", "keep", "return", "preference"),
            direction="higher", magnitude=round(sum(taste) / len(taste), 3),
        )
    if favorites:
        favorite_models = Counter(
            str(r.get("camera_model")) for r in favorites if r.get("camera_model")
        )
        if favorite_models:
            model, count = favorite_models.most_common(1)[0]
            add(
                "favorites:camera", "favorites",
                f"Your explicit favorites are most concentrated around {model}; this is a "
                f"behavioral signal from your Apple Photos choices, not a quality score.",
                ("favorites", "favorite", "kept", "camera", "gear", "preference", "taste"),
                direction="higher", magnitude=float(count),
            )

    lens_counts = Counter(str(r.get("lens_model")).strip() for r in canon if r.get("lens_model"))
    for lens, count in lens_counts.most_common(5):
        if count < 20:
            continue
        add(
            f"lens:overall:{_model_slug(lens)}", "lens",
            f"{lens} is a recurring part of your optical vocabulary; the lens itself is part "
            f"of how you approach subjects, not just a metadata detail.",
            ("lens", "lens model", "optics", "gear", "camera", lens.lower()),
            direction="higher", magnitude=float(count),
        )

    places = Counter()
    located = [r for r in canon if r.get("latitude") is not None and r.get("longitude") is not None]
    for row in canon:
        place = row.get("city") or row.get("country") or row.get("state")
        if place:
            places[str(place).strip()] += 1
    if places:
        place, count = places.most_common(1)[0]
        add(
            "place:overall:top", "place",
            f"Your photographs return most often to {place}; place is a recurring part of "
            f"the memory landscape in the archive.",
            ("place", "places", "location", "where", "city", "country", str(place).lower()),
            direction="higher", magnitude=float(count), entities=(),
        )
    elif len(located) >= 30:
        add(
            "place:overall:coverage", "place",
            "A geographic layer exists across a substantial part of the archive, but place "
            "names have not been resolved locally yet. The engine can identify recurring "
            "coordinate regions without sending location data for reverse geocoding.",
            ("place", "places", "location", "where", "region", "map", "gps"),
            direction="higher", magnitude=round(len(located) / len(canon), 2),
        )

    travel_labels = next((labels for slug, _tags, labels, _phrase in _SUBJECT_GROUPS
                          if slug == "travel"), frozenset())
    travel_frames = sum(1 for row in canon if _scene_set(row) & travel_labels)
    if travel_frames >= 30:
        add(
            "travel:overall", "travel",
            "Travel and the road recur as a photographic subject in the archive—vehicles, "
            "luggage, journeys, and places away from home appear as part of the story.",
            ("travel", "trip", "trips", "journey", "road", "away", "vacation", "places"),
            direction="higher", magnitude=round(travel_frames / len(canon), 2),
        )

