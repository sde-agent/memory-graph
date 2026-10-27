"""Insight-graph node generation (the population `ask` selects from)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from memory_graph.config import load_config
from memory_graph.narration.generate import GRAPH_KINDS, generate_graph


def eras():
    return load_config().eras


def _ts(year: int) -> float:
    return datetime(year, 6, 1, tzinfo=UTC).timestamp()


def _row(rid, year, q=1, *, warm=None, focal=None, faces=0, known=False,
         golden=False, names=None) -> dict:
    return {
        "id": rid, "is_canonical": 1, "timestamp": _ts(year), "dominant_quadrant": q,
        "warm_cool": warm, "focal_length_35mm": focal, "face_count": faces,
        "known_person": 1 if known else 0, "golden_hour": 1 if golden else 0,
        "person_names": json.dumps(names) if names else "[]", "resonance": 1.0,
    }


def test_nodes_carry_tags_and_feeling_claim() -> None:
    rows = [_row(f"w{i}", 2012, warm=0.3) for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    color = [n for n in nodes if n.kind == "color" and n.era is None][0]
    assert color.tags                              # retrieval keywords present
    assert "color" in color.tags
    # Feeling-first claim, not a bare statistic.
    assert "warm" in color.claim.lower()
    assert color.created_at == 1.0


def test_warm_palette_detected() -> None:
    rows = [_row(f"w{i}", 2012, warm=0.3) for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    overall = [n for n in nodes if n.id == "color:overall:warm"]
    assert overall and overall[0].direction == "higher"


def test_cool_palette_detected() -> None:
    rows = [_row(f"c{i}", 2012, warm=-0.3) for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    assert any(n.id == "color:overall:cool" for n in nodes)


def test_framing_telephoto_vs_wide() -> None:
    wide = [_row(f"a{i}", 2012, focal=20.0) for i in range(6)]
    tele = [_row(f"z{i}", 2022, focal=100.0) for i in range(6)]
    nodes = generate_graph(wide + tele, eras(), now=1.0)
    framings = {n.era: n for n in nodes if n.kind == "framing" and n.era}
    assert "wide" in framings["2000s" if "2000s" in framings else list(framings)[0]].claim \
        or any("wide" in n.claim for n in nodes if n.kind == "framing")
    assert any("telephoto" in n.claim for n in nodes if n.kind == "framing")


def test_people_top_names_are_allowed_entities() -> None:
    rows = [_row(f"z{i}", 2022, faces=1, known=True, names=["Ada"]) for i in range(40)]
    nodes = generate_graph(rows, eras(), now=1.0)
    top = [n for n in nodes if n.id.startswith("people:") and n.id.endswith(":top")]
    assert top
    assert "Ada" in top[0].claim
    assert "Ada" in top[0].allowed_entities


def test_change_nodes_span_first_to_last_era() -> None:
    rows = [_row(f"a{i}", 2012, faces=0, known=False) for i in range(10)]
    rows += [_row(f"z{i}", 2022, faces=3, known=(i < 8), names=["Ada"])
             for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    change = [n for n in nodes if n.kind == "change"]
    assert change
    assert any("turned toward the people" in n.claim for n in change)
    assert all(n.era is None for n in change)   # change nodes are cross-era


def test_depth_nodes_describe_bokeh_proxy() -> None:
    rows = []
    for i in range(30):
        row = _row(f"b{i}", 2022, focal=85.0, faces=1)
        row.update(aperture=1.8, face_ratio=0.25)
        rows.append(row)
    for i in range(10):
        row = _row(f"d{i}", 2022, focal=24.0)
        row.update(aperture=11.0)
        rows.append(row)
    nodes = generate_graph(rows, eras(), now=1.0)
    depth = [n for n in nodes if n.id == "depth:overall:bokeh"]
    assert depth
    assert "bokeh" in depth[0].claim.lower()
    assert "proxy" in depth[0].claim.lower()



    iphone = [_row(f"i{i}", 2022, focal=24.0, faces=1) for i in range(30)]
    dslr = [_row(f"d{i}", 2012, focal=85.0, faces=0) for i in range(30)]
    for row in iphone:
        row.update(camera_make="Apple", camera_model="iPhone 14 Pro Max")
    for row in dslr:
        row.update(camera_make="NIKON CORPORATION", camera_model="NIKON D700")
    nodes = generate_graph(iphone + dslr, eras(), now=1.0)
    comparison = [n for n in nodes if n.id == "camera:compare:iphone-dslr"]
    assert comparison
    assert "wider" in comparison[0].claim.lower()
    assert "camera" in comparison[0].tags


def test_all_kinds_are_declared() -> None:
    rows = [_row(f"z{i}", 2022, q=3, warm=0.3, focal=85.0, faces=3, known=True,
                 golden=True, names=["Ada"]) for i in range(40)]
    rows += [_row(f"a{i}", 2012, q=1, warm=-0.3, focal=20.0) for i in range(40)]
    nodes = generate_graph(rows, eras(), now=1.0)
    produced = {n.kind for n in nodes}
    # Every kind the generator emits must be in GRAPH_KINDS (persistence relies on it).
    assert produced <= set(GRAPH_KINDS)


def test_empty_archive_yields_no_nodes() -> None:
    assert generate_graph([], eras(), now=1.0) == []


def _row_ev(rid, year, ev) -> dict:
    r = _row(rid, year)
    r["ev"] = ev
    return r


def test_night_day_node_from_exposure_value() -> None:
    # Mostly bright (high EV) photos -> a 'day' light node; the tags include night/day
    # so a "night or day" question can retrieve it.
    rows = [_row_ev(f"d{i}", 2020, 11.0) for i in range(30)]
    rows += [_row_ev(f"n{i}", 2020, 2.0) for i in range(5)]
    nodes = generate_graph(rows, eras(), now=1.0)
    day = [n for n in nodes if n.id.endswith(":day")]
    assert day, "a daylight-dominant archive should yield a 'day' light node"
    assert "night" in day[0].tags and "day" in day[0].tags


def test_night_dominant_yields_night_node() -> None:
    rows = [_row_ev(f"n{i}", 2020, 2.0) for i in range(20)]
    rows += [_row_ev(f"d{i}", 2020, 11.0) for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    assert any(n.id.endswith(":night") for n in nodes)


def _row_scene(rid, year, labels) -> dict:
    # labels: list of (name, conf). Build the scene_labels JSON the generator reads.
    r = _row(rid, year)
    r["scene_labels"] = json.dumps([[n, c] for n, c in labels])
    return r


def test_subject_node_from_scene_labels() -> None:
    # Many photos labeled 'child' -> a 'children' subject node; Vision junk never a node.
    rows = [_row_scene(f"c{i}", 2020, [("child", 0.6), ("structure", 0.9),
                                       ("textile", 0.8)]) for i in range(40)]
    nodes = generate_graph(rows, eras(), now=1.0)
    subj = [n for n in nodes if n.kind == "subject"]
    assert any(n.id.endswith(":children") for n in subj)
    # 'structure'/'textile' are curated OUT — they must never become a subject node.
    assert not any("structure" in n.claim.lower() or "textile" in n.claim.lower()
                   for n in subj)


def test_subject_dominant_names_top_subject() -> None:
    rows = [_row_scene(f"o{i}", 2020, [("outdoor", 0.6), ("sky", 0.5)])
            for i in range(40)]
    rows += [_row_scene(f"c{i}", 2020, [("child", 0.6)]) for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    dom = [n for n in nodes if n.id.endswith("subject:overall:dominant")
           or (n.kind == "subject" and ":dominant" in n.id and n.era is None)]
    assert dom
    assert "outdoor" in dom[0].claim.lower()


def test_rare_subject_below_threshold_omitted() -> None:
    rows = [_row_scene(f"o{i}", 2020, [("outdoor", 0.6)]) for i in range(40)]
    rows += [_row_scene(f"f{i}", 2020, [("food", 0.6)]) for i in range(2)]  # ~5%
    nodes = generate_graph(rows, eras(), now=1.0)
    assert not any(n.kind == "subject" and n.id.endswith(":food") for n in nodes)


def _golden_row(rid, *, golden, scenes=None, names=None) -> dict:
    r = _row(rid, 2020, golden=golden, names=names, faces=1 if names else 0)
    r["has_gps"] = 1          # golden hour is only observable on located photos

    if scenes:
        r["scene_labels"] = json.dumps([[s, 0.9] for s in scenes])
    return r


def test_golden_subject_lift_node() -> None:
    # Golden frames are mostly outdoors; the rest of the archive is not. -> a lift node.
    # Needs >= 30 golden frames to speak (the bar), so the fixture clears it comfortably.
    rows = [_golden_row(f"g{i}", golden=True, scenes=["outdoor", "sky"]) for i in range(35)]
    rows += [_golden_row(f"x{i}", golden=False, scenes=["structure"]) for i in range(60)]
    rows += [_golden_row(f"y{i}", golden=False, scenes=["outdoor"]) for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    conj = [n for n in nodes if n.kind == "conjunction" and n.id.startswith("golden:subject")]
    assert any(n.id.endswith(":nature") for n in conj)


def test_golden_person_lift_node() -> None:
    # A person who appears almost only in golden frames gets a person-lift node. Owner is
    # never a golden-person node.
    rows = [_golden_row(f"p{i}", golden=True, names=["Pet"]) for i in range(12)]
    rows += [_golden_row(f"s{i}", golden=True) for i in range(25)]       # golden, no Pet
    rows += [_golden_row(f"q{i}", golden=False, names=["Pet"]) for i in range(13)]
    rows += [_golden_row(f"n{i}", golden=False) for i in range(100)]
    nodes = generate_graph(rows, eras(), now=1.0)
    assert any(n.id == "golden:person:Pet" for n in nodes)


def _loc_row(rid, year, golden) -> dict:
    r = _row(rid, year, golden=golden)
    r["has_gps"] = 1
    return r


def test_no_golden_claim_from_unlocated_era() -> None:
    # A period with NO location data (the 2000s) must not produce a golden or daylight
    # claim: golden_hour=0 there means unknown, not 'shoots in daylight'.
    rows = [_row(f"u{i}", 2003, golden=False) for i in range(60)]
    nodes = generate_graph(rows, eras(), now=1.0)
    bad = [n for n in nodes if n.kind == "light" and n.era == "2000s"
           and ("golden" in n.id or "daylight" in n.id)]
    assert not bad


def test_golden_share_uses_located_denominator() -> None:
    # Unlocated photos (golden_hour=0 by default) must not dilute the golden share: of the
    # 40 located photos, half are golden, so the share is 0.5 -> a golden node.
    rows = [_row(f"u{i}", 2022, golden=False) for i in range(200)]
    rows += [_loc_row(f"g{i}", 2022, True) for i in range(20)]
    rows += [_loc_row(f"d{i}", 2022, False) for i in range(20)]
    nodes = generate_graph(rows, eras(), now=1.0)
    node = [n for n in nodes if n.id == "light:2020s:golden"]
    assert node and node[0].magnitude == 0.5


def test_thin_golden_is_described_as_a_thread() -> None:
    # About 5% golden among located photos is a small steady thread, not a headline.
    rows = [_loc_row(f"g{i}", 2022, True) for i in range(5)]
    rows += [_loc_row(f"d{i}", 2022, False) for i in range(95)]
    nodes = generate_graph(rows, eras(), now=1.0)
    node = [n for n in nodes if n.id == "light:2020s:golden"]
    assert node and "steady thread" in node[0].claim


def test_no_conjunction_when_too_few_golden_frames() -> None:
    # A handful of golden frames is not a theme — no conjunction nodes at all.
    rows = [_golden_row(f"g{i}", golden=True, scenes=["outdoor"]) for i in range(5)]
    rows += [_golden_row(f"x{i}", golden=False, scenes=["structure"]) for i in range(60)]
    nodes = generate_graph(rows, eras(), now=1.0)
    assert not [n for n in nodes if n.kind == "conjunction"]


def test_infinite_ev_is_ignored() -> None:
    # A glitch EV (inf) must not crash or skew the night/day computation.
    rows = [_row_ev(f"d{i}", 2020, 11.0) for i in range(25)]
    rows.append(_row_ev("bad", 2020, float("inf")))
    nodes = generate_graph(rows, eras(), now=1.0)   # must not raise
    assert any(n.kind == "light" for n in nodes)


def _scene(rid, year, *, neg=None, horizon=None, golden=False) -> dict:
    # An unpeopled (face_count 0) frame — a landscape/scene.
    return {
        "id": rid, "is_canonical": 1, "timestamp": _ts(year), "dominant_quadrant": 4,
        "face_count": 0, "negative_space": neg, "horizon_y_ratio": horizon,
        "golden_hour": 1 if golden else 0, "person_names": "[]", "resonance": 1.0,
    }


def test_landscape_open_node_from_vast_negative_space() -> None:
    # 30 open frames (vast negative space) in one era -> an 'open' landscape node.
    rows = [_scene(f"s{i}", 2022, neg=0.8) for i in range(30)]
    nodes = generate_graph(rows, eras(), now=1.0)
    land = [n for n in nodes if n.kind == "landscape"]
    assert land
    assert any(n.id.endswith(":open") for n in land)
    # Tags include landscape/nature vocabulary so `ask` can retrieve them.
    opens = [n for n in land if n.id.endswith(":open")][0]
    assert "landscape" in opens.tags and "nature" in opens.tags


def test_landscape_comparative_node_names_dominant_kind() -> None:
    # Mostly open, a minority intimate -> a 'dominant' node that leans to open.
    rows = [_scene(f"o{i}", 2022, neg=0.8) for i in range(30)]
    rows += [_scene(f"c{i}", 2022, neg=0.1) for i in range(8)]
    nodes = generate_graph(rows, eras(), now=1.0)
    dom = [n for n in nodes if n.kind == "landscape" and n.id.endswith(":dominant")]
    assert dom
    assert "open" in dom[0].claim


def test_landscape_needs_enough_scene_photos() -> None:
    # Too few unpeopled frames -> no landscape nodes (we don't speak from a handful).
    rows = [_scene(f"s{i}", 2022, neg=0.8) for i in range(5)]
    nodes = generate_graph(rows, eras(), now=1.0)
    assert not [n for n in nodes if n.kind == "landscape"]


def test_landscape_kind_is_declared() -> None:
    from memory_graph.narration.generate import GRAPH_KINDS
    assert "landscape" in GRAPH_KINDS


# --- full-distribution + relaxed-threshold nodes ---------------------------------

def test_framing_emits_distribution_bands() -> None:
    # A mix of wide, normal and tele focals -> a band node for each meaningful share.
    rows = [_row(f"w{i}", 2022, focal=20.0) for i in range(10)]
    rows += [_row(f"n{i}", 2022, focal=45.0) for i in range(10)]
    rows += [_row(f"t{i}", 2022, focal=120.0) for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    band_ids = {n.id for n in nodes if n.kind == "framing" and ":" in n.id.split(":", 1)[1]}
    assert any(i.endswith(":wide") for i in band_ids)
    assert any(i.endswith(":normal") for i in band_ids)
    assert any(i.endswith(":tele") for i in band_ids)


def test_style_emits_secondary_feeling() -> None:
    # A dominant Q4 with a strong Q1 second current -> both a style and a style:second.
    rows = [_row(f"a{i}", 2022, q=4) for i in range(14)]
    rows += [_row(f"b{i}", 2022, q=1) for i in range(6)]
    nodes = generate_graph(rows, eras(), now=1.0)
    styles = [n for n in nodes if n.kind == "style"]
    assert any(n.id.endswith(":second") for n in styles)


def test_color_always_speaks_even_when_balanced() -> None:
    # A near-even warm/cool split must still produce a color node (with its tilt), not
    # stay silent.
    rows = [_row(f"w{i}", 2022, warm=0.2) for i in range(10)]
    rows += [_row(f"c{i}", 2022, warm=-0.2) for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    assert any(n.kind == "color" and n.era is None for n in nodes)


# --- relationship nodes ----------------------------------------------------------

def test_relationship_pair_cooccurrence() -> None:
    # Two people photographed together many times -> a pairing node naming both.
    rows = [_row(f"p{i}", 2018, faces=2, known=True, names=["Ada", "Grace"])
            for i in range(12)]
    rows += [_row(f"s{i}", 2018, faces=1, known=True, names=["Ada"]) for i in range(5)]
    nodes = generate_graph(rows, eras(), now=1.0)
    pairs = [n for n in nodes if n.id.startswith("relationship:pair")]
    assert pairs
    assert set(pairs[0].allowed_entities) == {"Ada", "Grace"}


def test_relationship_arrival_mid_archive() -> None:
    # Someone absent early (naming present) but prominent later -> an 'arrived' node.
    # Each era needs enough named photos to be a TRUSTED era for trajectory shares.
    rows = [_row(f"e{i}", 2012, faces=1, known=True, names=["Ada"]) for i in range(50)]
    rows += [_row(f"l{i}", 2022, faces=1, known=True, names=["Ada", "Zoe"])
             for i in range(50)]
    nodes = generate_graph(rows, eras(), now=1.0)
    arrived = [n for n in nodes if n.id.startswith("relationship:arrived")]
    assert any("Zoe" in n.allowed_entities for n in arrived)


def test_trajectory_peaked_and_faded_shapes() -> None:
    # A person who spikes in a middle era then vanishes -> 'peaked'; one present early
    # then gone -> 'faded'. Needs >=3 trusted eras (>=40 named photos each).
    rows = []
    # early 2010s: Faye present, Peak absent.
    rows += [_row(f"a{i}", 2012, faces=1, known=True, names=["Faye"]) for i in range(45)]
    # late 2010s: Peak spikes, Faye still a bit.
    rows += [_row(f"b{i}", 2017, faces=1, known=True, names=["Peak"]) for i in range(40)]
    rows += [_row(f"c{i}", 2017, faces=1, known=True, names=["Faye", "Core"])
             for i in range(10)]
    # 2020s: both Peak and Faye gone; Core carries the era.
    rows += [_row(f"d{i}", 2022, faces=1, known=True, names=["Core"]) for i in range(50)]
    nodes = generate_graph(rows, eras(), now=1.0)
    shapes = {n.id.split(":")[1] for n in nodes if n.kind == "relationship"}
    assert "peaked" in shapes or "faded" in shapes   # a mid/early shape was detected


def test_minor_person_collapses_into_supporting_cast() -> None:
    # A person who only ever reaches a small share (a friend-of-a-friend) must NOT get an
    # individual "bond deepening" sentence — they belong in the supporting-cast aside.
    rows = []
    # A dominant steady person carries each era (so eras are trusted and the minor
    # person's share stays small).
    for yr, tag in ((2012, "a"), (2017, "b"), (2022, "c")):
        rows += [_row(f"{tag}{i}", yr, faces=1, known=True, names=["Anchor"])
                 for i in range(90)]
    # Minor person: enough photos to pass the person bar (>=20 total) and a RISING shape
    # (absent early, present late) so she classifies — but her peak share stays well
    # below the 15% individual-node bar (a friend's spouse who shows up a bit lately).
    rows += [_row(f"m1{i}", 2012, faces=1, known=True, names=["Minor"]) for i in range(2)]
    rows += [_row(f"m2{i}", 2017, faces=1, known=True, names=["Minor"]) for i in range(3)]
    rows += [_row(f"m3{i}", 2022, faces=1, known=True, names=["Minor"]) for i in range(15)]
    nodes = generate_graph(rows, eras(), now=1.0)
    rel = [n for n in nodes if n.kind == "relationship"]
    # Minor must NOT have an individual trajectory node...
    individual = [n for n in rel if "Minor" in n.allowed_entities
                  and n.id.split(":")[1] in ("arrived", "rising", "faded", "peaked")]
    assert not individual
    # ...but may appear in a 'cast' aside.
    cast = [n for n in rel if n.id.startswith("relationship:cast")]
    assert any("Minor" in n.allowed_entities for n in cast)


def test_major_person_keeps_individual_trajectory_node() -> None:
    # Someone who reaches a real presence (>=15% of an era) still earns their own node.
    rows = [_row(f"a{i}", 2012, faces=1, known=True, names=["Base"]) for i in range(50)]
    rows += [_row(f"b{i}", 2022, faces=1, known=True, names=["Base", "Major"])
             for i in range(50)]   # Major is ~50% of the 2020s -> a real arrival
    nodes = generate_graph(rows, eras(), now=1.0)
    arrived = [n for n in nodes if n.id.startswith("relationship:arrived")
               and "Major" in n.allowed_entities]
    assert arrived


def test_trajectory_needs_trusted_eras() -> None:
    # Too few named photos per era -> no trajectory node (shares untrustworthy).
    rows = [_row(f"e{i}", 2012, faces=1, known=True, names=["Ada"]) for i in range(10)]
    rows += [_row(f"l{i}", 2022, faces=1, known=True, names=["Ada", "Zoe"])
             for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    traj = [n for n in nodes if n.kind == "relationship"
            and n.id.split(":")[1] in ("arrived", "faded", "peaked", "steady", "rising")]
    assert not traj


# --- signature (cross-dimension) nodes -------------------------------------------

def test_signature_detects_tighter_framing_for_a_person() -> None:
    # Baseline is wide; one person is always shot tight -> a signature node for them.
    base = [_row(f"b{i}", 2020, faces=1, known=True, names=["Ada"], focal=24.0)
            for i in range(30)]
    tight = [_row(f"t{i}", 2020, faces=1, known=True, names=["Zoe"], focal=120.0)
             for i in range(25)]
    nodes = generate_graph(base + tight, eras(), now=1.0)
    sigs = [n for n in nodes if n.kind == "signature"]
    zoe = [n for n in sigs if "Zoe" in n.allowed_entities]
    assert zoe
    assert "tighter" in zoe[0].claim


def test_relationship_and_signature_kinds_declared() -> None:
    from memory_graph.narration.generate import GRAPH_KINDS
    assert "relationship" in GRAPH_KINDS
    assert "signature" in GRAPH_KINDS


# --- owner-identity handling (photos OF the owner were shot by others) -----------

def _owner(*names):
    from memory_graph.config import OwnerCfg
    return OwnerCfg(self_names=tuple(names))


def test_owner_excluded_from_people_you_photograph_most() -> None:
    # The owner appears a lot (shot by others) but must NOT be listed as someone the
    # owner "returns to" photographing.
    rows = [_row(f"s{i}", 2020, faces=1, known=True, names=["Me"]) for i in range(40)]
    rows += [_row(f"r{i}", 2020, faces=1, known=True, names=["Ada"]) for i in range(30)]
    nodes = generate_graph(rows, eras(), owner=_owner("Me"), now=1.0)
    top = [n for n in nodes if n.id.endswith(":top")]
    for n in top:
        assert "Me" not in n.allowed_entities
    assert any("Ada" in n.allowed_entities for n in top)


def test_owner_gets_no_signature_node() -> None:
    # A signature for the owner would describe how OTHERS framed them — never generate it.
    rows = [_row(f"s{i}", 2020, faces=1, known=True, names=["Me"], focal=120.0)
            for i in range(30)]
    rows += [_row(f"b{i}", 2020, faces=1, known=True, names=["Ada"], focal=24.0)
             for i in range(30)]
    nodes = generate_graph(rows, eras(), owner=_owner("Me"), now=1.0)
    sigs = [n for n in nodes if n.kind == "signature"]
    assert all("Me" not in n.allowed_entities for n in sigs)


def test_pair_with_owner_reframed_as_appear_alongside() -> None:
    rows = [_row(f"p{i}", 2020, faces=2, known=True, names=["Me", "Ada"])
            for i in range(15)]
    nodes = generate_graph(rows, eras(), owner=_owner("Me"), now=1.0)
    with_owner = [n for n in nodes if "with_owner" in n.id]
    assert with_owner
    assert "appear alongside" in with_owner[0].claim.lower()
    assert "Me" not in with_owner[0].allowed_entities   # only the OTHER person named
    assert "Ada" in with_owner[0].allowed_entities


def test_self_presence_node_created_when_owner_is_frequent_subject() -> None:
    rows = [_row(f"s{i}", 2020, faces=1, known=True, names=["Me"]) for i in range(20)]
    rows += [_row(f"r{i}", 2020, faces=1, known=True, names=["Ada"]) for i in range(20)]
    nodes = generate_graph(rows, eras(), owner=_owner("Me"), now=1.0)
    assert any(n.id == "self:presence" for n in nodes)


def test_no_owner_configured_leaves_behavior_unchanged() -> None:
    # Without an owner, the person is treated as any other subject (self nodes absent).
    rows = [_row(f"s{i}", 2020, faces=1, known=True, names=["Me"]) for i in range(30)]
    nodes = generate_graph(rows, eras(), now=1.0)   # no owner
    assert not any(n.id.startswith("self:") for n in nodes)


# --- unknown / unnamed face presence (project A: aggregate only) -----------------

def test_unnamed_presence_node_when_many_faces_unnamed() -> None:
    # Mostly faces with no name -> an 'unnamed' node (we can't identify them, but the
    # SHARE is a true aggregate finding).
    rows = [_row(f"u{i}", 2020, faces=2) for i in range(40)]          # faces, no names
    rows += [_row(f"k{i}", 2020, faces=1, known=True, names=["Ada"]) for i in range(10)]
    nodes = generate_graph(rows, eras(), now=1.0)
    unnamed = [n for n in nodes if n.id.endswith(":unnamed")]
    assert unnamed
    assert "never named" in unnamed[0].claim


def test_unnamed_node_says_unnamed_not_strangers() -> None:
    # Truthfulness: an untagged face is 'unnamed', never asserted to be a stranger.
    rows = [_row(f"u{i}", 2020, faces=2) for i in range(40)]
    nodes = generate_graph(rows, eras(), now=1.0)
    for n in nodes:
        if n.kind == "unknown_people":
            assert "stranger" not in n.claim.lower()   # tag may say it; the CLAIM must not


def test_unknown_trend_detects_narrowing() -> None:
    # Early: almost all unnamed. Late: mostly named. -> a 'narrowed' trend node.
    rows = [_row(f"e{i}", 2012, faces=2) for i in range(40)]                 # unnamed
    rows += [_row(f"l{i}", 2022, faces=1, known=True, names=["Ada"]) for i in range(40)]
    nodes = generate_graph(rows, eras(), now=1.0)
    trend = [n for n in nodes if n.id == "unknown_people:trend"]
    assert trend
    assert trend[0].direction == "lower"      # unnamed share fell
    assert "narrowed" in trend[0].claim


def test_unknown_people_needs_enough_people_photos() -> None:
    rows = [_row(f"u{i}", 2020, faces=2) for i in range(10)]   # too few
    nodes = generate_graph(rows, eras(), now=1.0)
    assert not any(n.kind == "unknown_people" and n.era == "2020s" for n in nodes)
