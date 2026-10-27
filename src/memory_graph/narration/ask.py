"""`ask(question)` — select from the insight graph and narrate at matching depth.

The insight graph (generate.py) is a large population of committed nodes. `ask` does the
SELECTION the intent doc describes: it reads a free-form question, picks the nodes that
bear on it, and hands exactly those to the LLM to weave into a personal, grounded
reflection. The engine never lets the model compute from raw attributes — it only lets
the model choose among facts the engine already committed.

Depth scales with the question, because depth IS node count:
  - a NARROW question ("do I prefer warmer or cooler colours?") matches one dimension →
    a few nodes → a sentence or two;
  - a BROAD question ("how has my photography changed over the last two decades?") is
    about change across eras → many nodes (the per-era + change nodes) → a paragraph.

Selection is pure over (question, nodes); it is unit-testable without a DB or an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memory_graph.contracts import Insight

if TYPE_CHECKING:
    from memory_graph.config import ErasCfg

# Question words that signal a BROAD, longitudinal, "tell me the arc" ask. When present
# (and especially with a change/time cue) we widen selection to the change + per-era
# nodes and let the narrative run longer.
_BROAD_CUES = {
    "change", "changed", "changing", "evolve", "evolved", "evolution", "journey",
    "over", "time", "decade", "decades", "years", "history", "grow", "grown", "grew",
    "develop", "developed", "trajectory", "story", "arc", "overall", "generally",
    "summarize", "summarise", "describe", "tell",
}
# Cheap stopwords so tag matching keys off meaningful tokens only.
_STOP = {
    "a", "an", "the", "of", "in", "on", "to", "do", "did", "does", "i", "my", "me",
    "is", "are", "was", "were", "you", "your", "about", "what", "whats",
    "how", "has", "have", "had", "which", "or", "and", "more", "most", "prefer",
    "like", "tend", "usually", "typically", "really", "photos", "photo",
    "photography", "photograph", "photographs", "picture", "pictures", "images",
    "shoot", "shot", "take", "took", "capture", "when",
}
# Tokens that also appear in the broad-cue set but should NOT survive as topic tokens
# (they steer breadth, not tag matching). Kept separate from _STOP so breadth detection
# still sees them in the raw question.
_WORD_RE = re.compile(r"[a-z0-9']+")

# Conjunction nodes intentionally share these words so a question about golden-hour
# relationships or subjects can retrieve them. They are not enough to establish the
# node's dimension: "warm" in a golden:person node must not make it answer a color ask.
_CONJ_SHARED_TAGS = frozenset({
    "golden", "hour", "sunset", "light", "warm", "evening", "most", "what", "often",
})

# Explicit color words define a narrow color ask's admissible insight dimension. We keep
# this deliberately narrow: questions about "style" are intentionally cross-dimensional,
# while a direct color preference question must not admit generic warm/light/person facts.
_COLOR_TOPIC_TERMS = frozenset({
    "color", "colour", "palette", "temperature", "tone",
})
_DEPTH_TOPIC_TERMS = frozenset({
    "bokeh", "depth", "depths", "blur", "blurry", "focus", "dof", "aperture",
})
_CAMERA_TOPIC_TERMS = frozenset({"camera", "cameras", "gear", "body", "bodies"})
_PEOPLE_TOPIC_TERMS = frozenset({"people", "person", "family", "faces", "who"})


def _narrow_topic_kinds(qtokens: set[str]) -> frozenset[str]:
    """Restrict explicit color/depth asks; other narrow topics keep existing retrieval."""
    if qtokens & _CAMERA_TOPIC_TERMS and qtokens & _PEOPLE_TOPIC_TERMS:
        return frozenset({"camera", "people", "relationship", "signature"})
    if qtokens & _COLOR_TOPIC_TERMS:
        return frozenset({"color"})
    if qtokens & _DEPTH_TOPIC_TERMS:
        return frozenset({"depth"})
    return frozenset()


def _stem(w: str) -> str:
    """Very light stemming so singular/plural and simple inflections meet at one form.

    'colors'/'colours' -> 'color'-ish, 'warmer' -> 'warm', and crucially 'landscapes'
    and 'landscape' BOTH collapse to the same stem (we strip a trailing bare 'e' as a
    final step so '...es' and '...e' do not diverge, which was a real miss).
    """
    for suf in ("iest", "est", "ing", "ers", "er", "ies", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if suf == "ies":
                return base + "y"
            w = base
            break
    # Normalize a trailing silent 'e' so 'landscape' -> 'landscap' matches 'landscapes'.
    if len(w) >= 4 and w.endswith("e"):
        w = w[:-1]
    return w


@dataclass(frozen=True)
class Selection:
    nodes: tuple[Insight, ...]   # the chosen insight nodes, best match first
    broad: bool                  # whether the question reads as longitudinal/overview
    target_sentences: tuple[int, int]  # (min, max) guidance for narration length


def _tokens(text: str) -> list[str]:
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP]
    # Keep both the raw and the stemmed form so a match hits whichever the tag uses.
    out: set[str] = set()
    for w in words:
        out.add(w)
        out.add(_stem(w))
    return list(out)


# Decade tokens in a question map to the era labels that fall inside that decade, so
# "the 2010s" surfaces both "early 2010s" and "late 2010s" nodes.
_DECADE_RE = re.compile(r"\b((?:19|20)\d0)s\b")


def _era_mentions(
    question: str, nodes: list[Insight], eras: ErasCfg | None = None
) -> set[str]:
    """Era labels implicated by the question — matched by YEAR RANGE when the era config
    is available, else by the label text as a fallback.

    A decade in the question ('the 2010s') implicates every era whose year range overlaps
    that decade [2010, 2020). This is robust to how the era is LABELLED: a label like
    '2013-2014' or 'the kids were small' still matches the 2010s because its RANGE does,
    where the old string match (looking for '2010' inside the label) silently missed it.

    When `eras` is not provided (e.g. a unit test that passes only nodes), fall back to
    the original behaviour: exact label-in-question, or the decade token appearing inside
    the label text.
    """
    labels = {n.era for n in nodes if n.era}
    ql = question.lower()
    decades = {int(m.group(1)) for m in _DECADE_RE.finditer(ql)}  # e.g. {2010}
    hit: set[str] = set()

    # Exact label mention always counts ('what was my style in the late 2010s').
    for lab in labels:
        if lab and lab.lower() in ql:
            hit.add(lab)

    if not decades:
        return hit

    if eras is not None:
        # RANGE-BASED: an era is implicated when its [start, end) overlaps any named
        # decade [d, d+10). Independent of the label's wording.
        label_range = {r.label: (r.start, r.end) for r in eras.ranges}
        for lab in labels:
            if lab in hit or lab not in label_range:
                continue
            start, end = label_range[lab]
            if any(start < d + 10 and end > d for d in decades):
                hit.add(lab)
        return hit

    # FALLBACK (no era config): decade token appears inside the label text.
    for lab in labels:
        if lab in hit:
            continue
        lab_decades = {int(m.group(1)) for m in _DECADE_RE.finditer(lab.lower())}
        if lab_decades & decades:
            hit.add(lab)
    return hit


def select_nodes(
    question: str, insights: list[Insight], *, max_nodes: int = 12,
    eras: ErasCfg | None = None,
) -> Selection:
    """Pick the nodes that bear on `question`, ranked by relevance.

    Scoring per node: tag/token overlap + bonuses for a named era match and for change
    nodes when the question is longitudinal. Narrow questions return the top handful at
    the 'overall' scope; broad ones pull the change + per-era population.

    `eras`, when supplied, lets a decade in the question match an era by its YEAR RANGE
    rather than by its label text — robust to any era labelling (see `_era_mentions`).
    """
    qtokens = set(_tokens(question))
    raw_tokens = set(_WORD_RE.findall(question.lower()))
    broad = bool(raw_tokens & _BROAD_CUES)
    eras_named = _era_mentions(question, insights, eras)

    # Intent fallback: a natural question like "what do I photograph most?" is ALL
    # stopwords ("what/do/i/photograph/most"), so it tokenizes to nothing and would
    # dead-end. But its intent is clear — the recurring SUBJECT. When the question
    # signals that intent and has no content tokens, seed the subject vocabulary so the
    # subject nodes (what you shoot) can match.
    if not qtokens and raw_tokens & {
        "photograph", "photographs", "shoot", "take", "capture", "subject", "subjects",
        "what", "most",
    }:
        qtokens = {"subject", "what", "photograph", "things", "themes"}

    topic_kinds = frozenset() if broad else _narrow_topic_kinds(qtokens)

    scored: list[tuple[float, Insight]] = []
    for ins in insights:
        if topic_kinds and ins.kind not in topic_kinds:
            continue
        tagset = {t.lower() for t in ins.tags}
        # Normalize tag words through the SAME stemmer as the question tokens, so a plural
        # tag ('landscapes') and a stemmed query token ('landscap') meet in the middle and
        # singular/plural never silently miss.
        tag_words = {w for t in tagset for w in t.split()}
        tag_match = tag_words | {_stem(w) for w in tag_words}
        overlap = len(qtokens & tag_match)
        # A NAMED person in the question should pull that person's nodes — the name lives
        # in allowed_entities, not the (generic) tags. Match the question's tokens against
        # each entity's name tokens so "how do I photograph <person>" finds that person's
        # signature/relationship nodes even though the name is not a tag.
        entity_tokens = {t for name in ins.allowed_entities
                         for t in name.lower().split()}
        entity_hit = bool(qtokens & entity_tokens)
        era_hit = bool(ins.era and ins.era in eras_named)

        # Conjunction nodes carry shared golden-hour vocabulary plus a family-specific
        # vocabulary. Shared words such as "warm" and "light" are useful for finding a
        # golden-hour question, but are too broad to make a person/subject fact relevant
        # to a narrow color question. Require a family-specific match or an explicitly
        # named entity before admitting a conjunction node.
        if ins.kind == "conjunction" and not entity_hit:
            shared = _CONJ_SHARED_TAGS | {_stem(w) for w in _CONJ_SHARED_TAGS}
            specific_overlap = qtokens & (tag_match - shared)
            if not specific_overlap:
                continue

        # TOPICAL relevance: does this node actually bear on the question? Only such nodes
        # are eligible for a NARROW answer, so a colour question is never padded with an
        # unrelated framing or change node just to reach a count.
        topical = overlap > 0 or era_hit or entity_hit

        score = overlap * 2.0
        if entity_hit:
            score += 4.0          # a named-person match is a strong, specific signal
        if ins.era:
            if era_hit:
                score += 3.0
            elif eras_named:
                score -= 2.0          # a non-asked era is off-topic for a targeted ask
        if broad:
            # Longitudinal questions want the change arc and the per-era spread even when
            # a node shares no keyword — the arc IS the answer.
            if ins.kind == "change":
                score += 2.5
                topical = True
            if ins.era is not None:
                score += 0.5
                topical = True
            # A life-change story is told AGAINST its constants. The 'steady' people (a
            # partner, a child — present throughout, so they share no 'changed' tag) are
            # the backdrop the change plays on; without them the most important person can
            # vanish from a question about one's own life. Make them topical, and give a
            # gentle boost so a couple of anchors survive among the change nodes.
            if ins.id.startswith("relationship:steady:"):
                score += 1.5
                topical = True
        else:
            if ins.era is None:
                score += 0.75         # narrow asks prefer the whole-archive summary
        if topical and (score > 0 or broad):
            scored.append((score, ins))

    # Rank by score, then by MAGNITUDE so the strongest node of a tied group wins (e.g.
    # the most-frequent co-occurrence pair leads, not whichever sorts first by name).
    # Id is the final, stable tiebreaker for determinism.
    scored.sort(key=lambda s: (-s[0], -(s[1].magnitude or 0.0), s[1].id))
    ranked = [ins for _, ins in scored]

    # Depth: narrow asks stay terse; broad asks earn room. Cap by node count so verbosity
    # tracks how much the engine actually has to say about THIS question. For a NARROW
    # ask we also drop near-identical nodes (e.g. the same 'balanced palette' repeated per
    # era) so the answer is not padded with echoes; a BROAD ask keeps the per-era spread
    # because that spread IS the arc it is describing.
    if broad:
        # A "how did it change" question is ABOUT change, so change-flavored nodes rank
        # highest and can crowd out the 'steady' people. But the constants are the
        # backdrop the change plays against — "who changed" is only meaningful relative to
        # "who stayed". So on a broad question we GUARANTEE the steady anchors are kept
        # (the most important people — a partner, a child — are steady, not changing).
        keep = _anchor_steady(ranked, max_nodes)
        target = (4, 7)
    elif eras_named:
        # A targeted-era question wants the full picture OF that era (both halves of a
        # decade, several dimensions), but style nodes must not consume every slot. Keep
        # one human-context node per requested era when available: named people first,
        # then the unnamed-identity caveat, then generic people presence. This lets a
        # style answer say both what the camera did and who/what the evidence can support.
        context = _era_context_nodes(ranked, eras_named)
        context_ids = {id(n) for n in context}
        rest = [n for n in ranked if id(n) not in context_ids]
        keep_ids = {id(n) for n in context + rest[: max(0, 6 - len(context))]}
        keep = [n for n in ranked if id(n) in keep_ids]
        target = (2, 4)
    else:
        keep = _dedupe_claims(ranked)[:4]
        target = (1, 3)
    return Selection(nodes=tuple(keep), broad=broad, target_sentences=target)


def _era_context_nodes(ranked: list[Insight], eras_named: set[str]) -> list[Insight]:
    """Choose one human-context node per requested era for targeted-era answers.

    Prefer a named people summary, then the unnamed-identity caveat, then generic people
    presence. The fallback order is deliberate: a sparse face-name era must be described
    as identity-limited instead of being silently treated as relationship-free.
    """
    chosen: list[Insight] = []
    for era in sorted(eras_named):
        candidates = [
            n for n in ranked
            if n.era == era and n.kind in {"people", "unknown_people", "anonymous_people"}
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda n: (
            0 if n.id.endswith(":top") else
            1 if n.kind == "anonymous_people" else
            2 if n.kind == "unknown_people" else
            3,
            -n.magnitude if n.magnitude is not None else 0.0,
            n.id,
        ))
        chosen.append(candidates[0])
    return chosen


def _is_steady(ins: Insight) -> bool:
    return ins.id.startswith("relationship:steady:")


def _anchor_steady(ranked: list[Insight], max_nodes: int) -> list[Insight]:
    """Keep the top `max_nodes` by rank, but GUARANTEE any 'steady' relationship nodes
    are included — the constant people are the backdrop a change story needs. Steady
    anchors go first (they frame the arc), then the highest-ranked remaining nodes fill
    the rest, preserving overall rank order among the non-anchors.
    """
    steady = [n for n in ranked if _is_steady(n)]
    if not steady:
        return ranked[:max_nodes]
    anchors = steady[: max(1, max_nodes // 3)]   # don't let anchors swamp the answer
    anchor_ids = {id(n) for n in anchors}
    rest = [n for n in ranked if id(n) not in anchor_ids]
    kept = anchors + rest[: max_nodes - len(anchors)]
    # Return in rank order so the strongest overall signal still leads the prose, but the
    # anchors are now guaranteed present.
    kept_ids = {id(n) for n in kept}
    return [n for n in ranked if id(n) in kept_ids]


def _dedupe_claims(nodes: list[Insight]) -> list[Insight]:
    """Collapse nodes that restate one theme (same kind + direction) to the single
    best-ranked representative. For a NARROW, non-era question the whole-archive summary
    ranks first, so the per-era echoes of the same theme fall away and the answer stays
    terse. (Era-targeted questions take a different branch that keeps the per-era spread.)
    """
    out: list[Insight] = []
    seen: set = set()
    for ins in nodes:
        # Conjunction nodes are DISTINCT facts (each subject or person is its own finding),
        # so they dedupe on their own id, not on kind+direction (which would collapse every
        # golden-hour lift into one node).
        if ins.kind == "conjunction":
            key = ("id", ins.id)
        elif ins.kind == "depth":
            # Keep distinct optical constructs (bokeh proxy vs wide aperture), but collapse
            # repeated per-era echoes for a narrow whole-archive question.
            key = ("depth", ins.id.rsplit(":", 1)[-1])
        else:
            key = (ins.kind, ins.direction)
        if key in seen:
            continue
        seen.add(key)
        out.append(ins)
    return out
