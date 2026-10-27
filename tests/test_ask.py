"""Free-form `ask`: node selection (depth-scaling) + grounded narration."""

from __future__ import annotations

from memory_graph.contracts import Insight
from memory_graph.narration.ask import select_nodes
from memory_graph.narration.llm import (
    CallableBackend,
    build_answer_prompt,
    narrate_answer,
)


def _ins(iid, kind, claim, tags, *, era=None, direction=None, entities=()) -> Insight:
    return Insight(id=iid, kind=kind, claim=claim, tags=tuple(tags), era=era,
                   direction=direction, allowed_entities=tuple(entities), created_at=1.0)


def _graph() -> list[Insight]:
    return [
        _ins("color:overall", "color", "Your palette is balanced.",
             ("color", "warm", "cool", "palette")),
        _ins("color:early 2010s", "color", "2010s palette balanced.",
             ("color", "warm", "cool", "palette"), era="early 2010s"),
        _ins("color:late 2010s", "color", "late 2010s palette balanced.",
             ("color", "warm", "cool", "palette"), era="late 2010s"),
        _ins("framing:overall", "framing", "You shoot wide.",
             ("framing", "lens", "wide", "focal")),
        _ins("people:overall:present", "people", "People are central.",
             ("people", "faces", "who"), direction="higher"),
        _ins("people:overall:top", "people", "You return to Ada most.",
             ("people", "who", "family"), entities=("Ada",)),
        _ins("style:early 2010s", "style", "2010s feeling solitary.",
             ("style", "feeling", "mood"), era="early 2010s"),
        _ins("change:people", "change", "You turned toward people you love.",
             ("change", "over time", "people", "evolved"), direction="higher"),
        _ins("change:framing", "change", "Your framing distance shifted.",
             ("change", "over time", "framing"), direction="higher"),
    ]


def test_narrow_question_selects_few_on_topic_nodes() -> None:
    sel = select_nodes("do I prefer warmer or cooler colors?", _graph())
    assert not sel.broad
    assert sel.target_sentences == (1, 3)
    kinds = {n.kind for n in sel.nodes}
    assert kinds == {"color"}                      # only the colour dimension
    assert len(sel.nodes) == 1                     # deduped to one representative


def test_stemming_matches_plural_and_comparative() -> None:
    # "colours" (British) and "warmer" must still hit the color node's tags.
    sel = select_nodes("are my colours warmer than cooler?", _graph())
    assert any(n.kind == "color" for n in sel.nodes)


def test_color_question_excludes_unrelated_golden_person_node() -> None:
    # Regression: the shared "warm" tag on golden-hour conjunctions used to leak a
    # specific person into a color answer.
    g = [
        _ins("color:overall:warm", "color", "Your palette leans warm.",
             ("color", "warm", "cool", "palette")),
        _ins("golden:person:GoldenGuest_Das", "conjunction",
             "GoldenGuest Das shows up in your golden-hour frames.",
             ("golden", "hour", "warm", "people", "who", "with", "family"),
             direction="higher", entities=("GoldenGuest Das",)),
        _ins("signature:FamilyPet", "signature",
             "The way you photograph FamilyPet has its own signature.",
             ("signature", "style", "framing", "warm", "portrait", "people", "who"),
             direction="higher", entities=("FamilyPet",)),
        _ins("light:overall:day", "light",
             "You are a daylight photographer.",
             ("light", "golden", "hour", "warm", "daylight", "day"),
             direction="higher"),
    ]
    sel = select_nodes("Do I prefer warmer or cooler colors?", g)
    assert [n.id for n in sel.nodes] == ["color:overall:warm"]


def test_camera_people_question_excludes_unrelated_golden_person() -> None:
    g = [
        _ins("camera:dslr:subject", "camera", "The DSLR is often used for people.",
             ("camera", "people", "subject")),
        _ins("relationship:pair:Ada|Grace", "relationship", "Ada and Grace recur together.",
             ("people", "relationship", "together"), entities=("Ada", "Grace")),
        _ins("golden:person:GoldenGuest", "conjunction", "GoldenGuest appears in golden light.",
             ("golden", "hour", "warm", "people"), entities=("GoldenGuest",)),
    ]
    sel = select_nodes("How do my camera choices affect the people I photograph?", g)
    ids = {n.id for n in sel.nodes}
    assert "camera:dslr:subject" in ids
    assert "relationship:pair:Ada|Grace" in ids
    assert "golden:person:GoldenGuest" not in ids



    # Regression: generic warm/person/golden tags must not drown out the explicit optical
    # evidence for a bokeh question.
    g = [
        _ins("depth:overall:bokeh", "depth", "Your archive often uses bokeh-friendly settings.",
             ("depth", "bokeh", "blur", "aperture"), direction="higher"),
        _ins("depth:overall:wide-aperture", "depth", "Wide apertures recur.",
             ("depth", "bokeh", "aperture"), direction="higher"),
        _ins("golden:person:GoldenGuest", "conjunction", "GoldenGuest appears in golden light.",
             ("golden", "hour", "warm", "people"), entities=("GoldenGuest",)),
    ]
    sel = select_nodes("Do I prefer my photos with bokeh or not?", g)
    assert {n.id for n in sel.nodes} == {
        "depth:overall:bokeh", "depth:overall:wide-aperture",
    }



    # The guard must remove accidental cross-topic matches without breaking a genuinely
    # relevant golden-hour person question.
    g = [
        _ins("golden:person:GoldenGuest_Das", "conjunction",
             "GoldenGuest Das shows up in your golden-hour frames.",
             ("golden", "hour", "warm", "people", "who", "with", "family"),
             direction="higher", entities=("GoldenGuest Das",)),
        _ins("golden:subject:water", "conjunction",
             "You reach for water in golden-hour light.",
             ("golden", "hour", "warm", "subject", "water"), direction="higher"),
    ]
    sel = select_nodes("Which people show up in golden hour?", g)
    assert "golden:person:GoldenGuest_Das" in {n.id for n in sel.nodes}
    assert "golden:subject:water" not in {n.id for n in sel.nodes}


def test_plural_question_matches_singular_tag() -> None:
    # Regression: 'landscapes' stems to 'landscap'; a node tagged 'landscape' must still
    # match (tag words are stemmed on the node side too, so both meet in the middle).
    g = [_ins("landscape:overall:open", "landscape", "You shoot open scenes.",
              ("landscape", "nature", "scene"))]
    sel = select_nodes("what kind of landscapes do I shoot more of?", g)
    assert len(sel.nodes) == 1
    assert sel.nodes[0].kind == "landscape"


def test_broad_change_question_keeps_steady_anchors() -> None:
    # Regression: a 'how did it change' question scores change-flavored nodes highest and
    # was dropping the STEADY people (a partner present throughout) — the backdrop the
    # change plays against. Steady relationship nodes must survive on a broad question.
    g = [
        _ins(f"relationship:arrived:P{i}", "relationship", f"P{i} arrived.",
             ("came", "arrived", "new", "changed", "over time"), entities=(f"P{i}",))
        for i in range(10)
    ]
    g += [
        _ins("relationship:steady:Partner", "relationship",
             "Partner runs steadily through your photographs.",
             ("constant", "always", "steady", "throughout"), entities=("Partner",)),
        _ins("relationship:steady:Child", "relationship",
             "Child is a constant presence.",
             ("constant", "always", "steady", "throughout"), entities=("Child",)),
    ]
    sel = select_nodes("how has my photography changed over the decades?", g)
    chosen = {e for n in sel.nodes for e in n.allowed_entities}
    assert "Partner" in chosen, "the steady partner must not be filtered out of a change story"
    assert "Child" in chosen


def test_broad_question_pulls_more_nodes_and_longer_target() -> None:
    sel = select_nodes("how has my photography changed over the decades?", _graph())
    assert sel.broad
    assert sel.target_sentences == (4, 7)
    assert len(sel.nodes) > 4                       # the arc + per-era spread
    assert any(n.kind == "change" for n in sel.nodes)


def test_era_targeted_question_pulls_that_era_both_halves() -> None:
    sel = select_nodes("what was my style in the 2010s?", _graph())
    eras = {n.era for n in sel.nodes if n.era}
    # A decade mention surfaces BOTH halves of the decade, not just one.
    assert "early 2010s" in eras
    assert "late 2010s" in eras
    # And it is mid-depth, not the terse narrow target.
    assert sel.target_sentences == (2, 4)


def test_era_targeted_question_pulls_human_context_and_identity_caveat() -> None:
    # Regression: a style question about an era must leave room for people/context nodes;
    # otherwise six style nodes crowd out the fact that identities are mostly unnamed.
    g = _graph() + [
        _ins("people:early 2010s:present", "people", "People are central early.",
             ("people", "faces"), era="early 2010s", direction="higher"),
        _ins("unknown_people:early 2010s:unnamed", "unknown_people",
             "Most faces are unnamed early.", ("people", "unnamed"), era="early 2010s",
             direction="higher",),
        _ins("people:late 2010s:top", "people", "FamilyA and FamilyB recur late.",
             ("people", "who"), era="late 2010s", entities=("FamilyA", "FamilyB")),
    ]
    sel = select_nodes("what was my style in the 2010s?", g)
    ids = {n.id for n in sel.nodes}
    assert "unknown_people:early 2010s:unnamed" in ids
    assert "people:late 2010s:top" in ids


def test_era_targeting_excludes_other_named_era_nodes() -> None:
    g = _graph() + [_ins("color:2020s", "color", "2020s palette.",
                         ("color", "palette"), era="2020s")]
    sel = select_nodes("what were my colors in the 2010s?", g)
    assert all(n.era != "2020s" for n in sel.nodes)   # 2020s pushed out of a 2010s ask


def test_decade_matches_era_by_range_regardless_of_label() -> None:
    # Regression: a decade in the question ('2010s') must implicate an era by its YEAR
    # RANGE, not by the decade token appearing in the label. An era labelled '2013-2014'
    # has no '2010' in its text, so the old string matcher silently missed it and the
    # question fell back to whole-archive mush. With the era config passed, the range
    # [2013,2015) overlaps the 2010s and the node is selected.
    from memory_graph.config import EraRange, ErasCfg

    eras = ErasCfg(ranges=(
        EraRange(label="2006-2007", start=2006, end=2008),
        EraRange(label="2013-2014", start=2013, end=2015),
        EraRange(label="2021-2022", start=2021, end=2023),
    ))
    g = [
        _ins("style:2006-2007", "style", "early feeling.",
             ("style", "feeling"), era="2006-2007"),
        _ins("style:2013-2014", "style", "mid feeling.",
             ("style", "feeling"), era="2013-2014"),
        _ins("style:2021-2022", "style", "recent feeling.",
             ("style", "feeling"), era="2021-2022"),
    ]
    # Without the era config, the decade token is absent from every label, so no era is
    # IMPLICATED: the question falls back to generic tag matching and does NOT single out
    # the correct (2013-2014) era over the others — the silent miss this fix targets.
    sel_old = select_nodes("what was my style in the 2010s?", g)
    old_eras = {n.era for n in sel_old.nodes if n.era}
    assert old_eras != {"2013-2014"}, "without era config the right era is not singled out"
    # With the era config, the 2013-2014 era matches the 2010s by RANGE and is ranked
    # FIRST (the strong era-hit bonus), where before it was not singled out at all. The
    # off-decade eras lose their era bonus and rank below it.
    sel = select_nodes("what was my style in the 2010s?", g, eras=eras)
    assert sel.nodes[0].era == "2013-2014", [n.era for n in sel.nodes]


def test_named_person_in_question_matches_entity_nodes() -> None:
    # A node whose name is only in allowed_entities (not tags) must be found when the
    # question names that person.
    g = [
        _ins("signature:Milo", "signature", "You reach in tighter with Milo.",
             ("signature", "how", "style"), entities=("Milo",)),
        _ins("color:overall", "color", "balanced palette", ("color", "palette")),
    ]
    sel = select_nodes("how do I photograph Milo?", g)
    assert any("Milo" in n.allowed_entities for n in sel.nodes)


def test_strongest_tied_node_ranks_first_by_magnitude() -> None:
    # Two equally tag-relevant pair nodes: the higher-magnitude one must lead, not the
    # alphabetically-first id.
    g = [
        _ins("relationship:pair:Aa|Ri", "relationship", "Aa and Ri appear together.",
             ("relationship", "together", "who"), entities=("Aa", "Ri")),
        _ins("relationship:pair:An|Ri", "relationship", "An and Ri appear together.",
             ("relationship", "together", "who"), entities=("An", "Ri")),
    ]
    # Give the second a higher magnitude.
    import dataclasses
    g[1] = dataclasses.replace(g[1], magnitude=287.0)
    g[0] = dataclasses.replace(g[0], magnitude=47.0)
    sel = select_nodes("who do I photograph together most?", g)
    assert sel.nodes[0].id == "relationship:pair:An|Ri"


def test_came_and_went_retrieves_trajectory_shapes() -> None:
    # A life-story question must pull faded/arrived/peaked nodes via their shape tags,
    # not just whatever generic 'people' node ranks first.
    g = [
        _ins("relationship:faded:Vi", "relationship", "Vi drifted out of your frames.",
             ("people", "went", "left", "faded", "came and went"), entities=("Vi",)),
        _ins("relationship:arrived:Ho", "relationship", "Ho arrived and stayed.",
             ("people", "came", "arrived", "new"), entities=("Ho",)),
        _ins("people:overall:present", "people", "People are central.", ("people",),
             direction="higher"),
    ]
    sel = select_nodes("which people came and went over the years?", g)
    shapes = {n.id.split(":")[1] for n in sel.nodes if n.kind == "relationship"}
    assert "faded" in shapes and "arrived" in shapes


def test_self_question_retrieves_self_node_despite_pronoun_stopwords() -> None:
    # "how often am I in my own photos" strips the pronouns I/me (stopwords), so the
    # self node must be reachable by concept words like 'own'/'myself'.
    g = [_ins("self:presence", "people",
              "You are a frequent subject yourself, photographed by others.",
              ("myself", "self", "own", "photographed", "subject"))]
    sel = select_nodes("how often am I in my own photos?", g)
    assert any(n.id == "self:presence" for n in sel.nodes)


def test_distinct_conjunction_nodes_are_not_collapsed() -> None:
    # Regression: a narrow question must keep distinct golden-hour facts (each subject and
    # person), not collapse them to one because they share kind and direction.
    g = [
        _ins("golden:subject:nature", "conjunction", "outdoors lift", ("golden", "subject"),
             direction="higher"),
        _ins("golden:subject:water", "conjunction", "water lift", ("golden", "subject"),
             direction="higher"),
    ]
    sel = select_nodes("What subjects do I shoot most at golden hour?", g)
    ids = {n.id for n in sel.nodes}
    assert {"golden:subject:nature", "golden:subject:water"} <= ids


def test_all_stopword_subject_question_still_matches() -> None:
    # "what do I photograph most?" is ALL stopwords -> empty tokens. The intent fallback
    # must still route it to the subject nodes rather than dead-ending.
    g = [
        _ins("subject:overall:dominant", "subject", "Your camera turns most to nature.",
             ("subject", "what", "photograph", "nature")),
        _ins("color:overall", "color", "balanced palette", ("color", "palette")),
    ]
    sel = select_nodes("what do I photograph most?", g)
    assert any(n.kind == "subject" for n in sel.nodes)


def test_unmatched_question_returns_no_nodes() -> None:
    g = [_ins("color:overall", "color", "palette", ("color", "palette"))]
    sel = select_nodes("what camera body did I use?", g)
    assert sel.nodes == ()


def test_narrate_answer_grounded_passes_through() -> None:
    nodes = [_ins("people:overall:top", "people", "You return to Ada most.",
                  ("people",), entities=("Ada",))]
    good = CallableBackend(lambda _p: "You keep coming back to Ada, again and again.")
    ans = narrate_answer("who do I photograph most?", nodes, good,
                         target_sentences=(1, 3))
    assert ans.grounded and not ans.fell_back
    assert "Ada" in ans.narrative
    assert ans.n_nodes == 1
    assert "Ada" in ans.evidence                    # evidence block (B) present


def test_narrate_answer_invented_name_falls_back() -> None:
    nodes = [_ins("people:overall:top", "people", "You return to Ada most.",
                  ("people",), entities=("Ada",))]
    bad = CallableBackend(lambda _p: "You and Grace are inseparable in these frames.")
    ans = narrate_answer("who?", nodes, bad, target_sentences=(1, 3),
                         max_regenerations=0)
    assert ans.fell_back
    assert ans.narrative == ""                       # fabricated name rejected
    assert "Ada" in ans.evidence                    # honest facts still returned


def test_narrate_answer_allows_grounded_camera_terms() -> None:
    node = _ins("camera:model:iphone", "camera",
                 "iPhone 14 Pro Max is a recurring camera in your record.",
                 ("camera", "iphone", "gear"))
    good = CallableBackend(lambda _p: "Your iPhone and DSLR choices shaped how you see.")
    ans = narrate_answer("which cameras shaped my photography?", [node], good,
                         target_sentences=(1, 3))
    assert ans.grounded and not ans.fell_back



    nodes = [_ins("color:overall", "color", "Your palette is balanced.", ("color",))]
    ans = narrate_answer("colors?", nodes, None)
    assert ans.fell_back
    assert ans.narrative == ""
    assert "balanced" in ans.evidence


def test_narrate_answer_empty_selection() -> None:
    ans = narrate_answer("anything?", [], CallableBackend(lambda _p: "x"))
    assert ans.fell_back
    assert ans.n_nodes == 0


def test_compact_tier_prompt_differs_and_forbids_bullets() -> None:
    # The 'compact' tier (for a small on-device model) must add tighter scaffolding than
    # the default, and must explicitly forbid starting lines with a dash/bullet (a small
    # model otherwise echoes the instruction format into its output).
    from memory_graph.narration.llm import build_answer_prompt
    nodes = [_ins("change:people", "change", "You turned toward people you love.",
                  ("change", "people"), direction="higher")]
    default = build_answer_prompt("how has it changed?", nodes, (4, 7), tier="default")
    compact = build_answer_prompt("how has it changed?", nodes, (4, 7), tier="compact")
    assert default != compact
    low = compact.lower()
    assert "never begin a line with a dash" in low or "no" in low and "bullet" in low
    assert "at most two or three people" in low   # name budget for the small model


def test_answer_prompt_frames_notes_not_data() -> None:
    # The MCP fallback directive must instruct a connected model to WRITE FROM the
    # observations, not display them — this is what prevents stat-salad tables over MCP.
    nodes = [_ins("change:people", "change", "You turned toward people you love.",
                  ("change", "people"), direction="higher")]
    prompt = build_answer_prompt("how has it changed?", nodes, (4, 7))
    low = prompt.lower()
    assert "not a dataset to display" in low or "notes" in low
    assert "do not" in low and ("table" in low or "list" in low)
    assert "4 to 7 sentences" in low            # depth guidance carried through
    assert "turned toward people you love" in prompt   # the observation is included


def test_depth_guidance_reaches_the_prompt() -> None:
    seen = {}
    nodes = [_ins("color:overall", "color", "balanced.", ("color",))]

    def capture(prompt):
        seen["p"] = prompt
        return "A brief, grounded line."

    narrate_answer("colors?", nodes, CallableBackend(capture),
                   target_sentences=(1, 3))
    assert "1 to 3 sentences" in seen["p"]            # length guidance injected
