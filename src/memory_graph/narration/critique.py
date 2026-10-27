"""Longitudinal style critique = the insight graph (spec.md §6/§8, evaluation.md C).

Answers "what was my style in the 2010s?" by deriving engine-authored Insights from DB
aggregates, then narrating them. Two views per era, per the resonance contract (Part A):
  - the UNWEIGHTED true distribution (what you actually shot), and
  - the resonance-weighted view (what you are drawn back to).
The GAP between them is one of the richest insights ("your most-shot quadrant was Q2,
but the photos you return to are Q3").

CRITICAL (C.0): the Insight's claim, direction, and magnitude are committed by CODE
here, before any LLM runs — so narration can never contradict them. The LLM only adds
voice, and its output is verified against the committed direction (C.3).

The aggregation is pure over photo rows (evaluation.md C.8: the critique's FACTS are
code-computed and unit-tested), so it is testable without a DB or an LLM.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from memory_graph.config import ErasCfg
from memory_graph.contracts import Insight

_QUADRANT_NAMES = {
    1: "Nostalgic & Reflective",
    2: "Visceral & High-Arousal",
    3: "Empathetic & Connected",
    4: "Solitary & Introspective",
    5: "Provocative & Dynamic",
}


@dataclass(frozen=True)
class EraSummary:
    era: str
    count: int
    shot_shares: dict[int, float]        # unweighted quadrant distribution (what you shot)
    resonant_shares: dict[int, float]    # resonance-weighted (what you return to)
    top_shot: int                        # argmax of shot_shares
    top_resonant: int                    # argmax of resonant_shares
    # Richer per-era stats for trend insights (all code-computed from the rows).
    median_focal: float | None = None    # median 35mm-equiv focal length
    people_share: float = 0.0            # fraction with any face
    known_share: float = 0.0             # fraction with a named (above-bar) person
    group_share: float = 0.0             # fraction with 3+ faces
    golden_share: float = 0.0            # fraction shot in golden hour
    top_people: tuple[tuple[str, int], ...] = ()  # most-photographed named people


def _year(ts: float | None) -> int | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, UTC).year


def summarize_eras(rows: list[dict], eras: ErasCfg) -> list[EraSummary]:
    """Per-era unweighted and resonance-weighted quadrant distributions, from photo rows."""
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        if not r.get("is_canonical"):
            continue
        era = eras.era_of(_year(r.get("timestamp")))
        if era is None:
            continue
        buckets.setdefault(era, []).append(r)

    summaries: list[EraSummary] = []
    # Preserve era order as configured.
    for er in eras.ranges:
        era = er.label
        rs = buckets.get(era)
        if not rs:
            continue
        shot_counts = {q: 0.0 for q in range(1, 6)}
        res_weight = {q: 0.0 for q in range(1, 6)}
        res_total = 0.0
        for r in rs:
            q = r.get("dominant_quadrant")
            if q not in shot_counts:
                continue
            shot_counts[q] += 1.0
            w = float(r.get("resonance") or 0.0)
            res_weight[q] += w
            res_total += w
        n = len(rs)
        shot_shares = {q: shot_counts[q] / n for q in range(1, 6)}
        resonant_shares = (
            {q: res_weight[q] / res_total for q in range(1, 6)}
            if res_total > 0
            else dict(shot_shares)
        )
        summaries.append(
            EraSummary(
                era=era,
                count=n,
                shot_shares=shot_shares,
                resonant_shares=resonant_shares,
                top_shot=max(range(1, 6), key=lambda q: shot_shares[q]),
                top_resonant=max(range(1, 6), key=lambda q: resonant_shares[q]),
                median_focal=_median([r["focal_length_35mm"] for r in rs
                                      if r.get("focal_length_35mm")]),
                people_share=sum(1 for r in rs if (r.get("face_count") or 0) > 0) / n,
                known_share=sum(1 for r in rs if r.get("known_person")) / n,
                group_share=sum(1 for r in rs if (r.get("face_count") or 0) >= 3) / n,
                golden_share=sum(1 for r in rs if r.get("golden_hour")) / n,
                top_people=_top_people(rs),
            )
        )
    return summaries


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _top_people(rs: list[dict], k: int = 3) -> tuple[tuple[str, int], ...]:
    from collections import Counter

    c: Counter = Counter()
    for r in rs:
        pn = r.get("person_names")
        if pn:
            try:
                c.update(json.loads(pn) if isinstance(pn, str) else pn)
            except (json.JSONDecodeError, TypeError):
                pass   # narrow: only tolerate a malformed value, not a coding error
    return tuple(c.most_common(k))


def build_insights(summaries: list[EraSummary], *, now: float | None = None) -> list[Insight]:
    """Derive committed Insights from era summaries (direction/magnitude fixed by code)."""
    from memory_graph.narration.meaning import quadrant_meaning

    created = now if now is not None else time.time()
    insights: list[Insight] = []
    for s in summaries:
        # Insight A: the dominant shooting style of the era — framed as POSTURE/feeling,
        # with the share as supporting evidence (not the headline). The LLM narrates the
        # feeling; it keeps the number out of the prose unless it sharpens meaning.
        top_share = s.shot_shares[s.top_shot]
        insights.append(
            Insight(
                id=f"ins:{s.era}:style",
                kind="era_style",
                # Claim is FEELING-only: no percentage or count in the prose-facing text
                # (the number lives in `magnitude`). Numbers baked into the claim were an
                # engraved invitation for a narrator to tabulate — the one thing the voice
                # forbids. "the leading feeling" conveys dominance without a figure.
                claim=(
                    f"In the {s.era}, your photographs mostly expressed "
                    f"{_QUADRANT_NAMES[s.top_shot].lower()} — {quadrant_meaning(s.top_shot)}; "
                    f"it was the leading feeling of that era."
                ),
                direction="higher",
                magnitude=round(top_share, 3),
                resonance_angle=None,
                allowed_entities=(),
                era=s.era,
                created_at=created,
                tags=_STYLE_TAGS,
            )
        )
        # Insight B: the resonance gap — what you shot vs what you return to.
        if s.top_resonant != s.top_shot:
            gap = s.resonant_shares[s.top_resonant] - s.shot_shares[s.top_resonant]
            insights.append(
                Insight(
                    id=f"ins:{s.era}:gap",
                    kind="resonance_gap",
                    claim=(
                        f"In the {s.era}, you shot {_QUADRANT_NAMES[s.top_shot]} most, "
                        f"but the photos you are drawn back to are "
                        f"{_QUADRANT_NAMES[s.top_resonant]}."
                    ),
                    direction="higher",
                    magnitude=round(gap, 3),
                    resonance_angle=(
                        f"{_QUADRANT_NAMES[s.top_resonant]} resonance share exceeds its "
                        f"shooting share"
                    ),
                    allowed_entities=(),
                    era=s.era,
                    created_at=created,
                    tags=_GAP_TAGS,
                )
            )

    # Whole-archive resonance PREFERENCE — "which photos do I keep / return to most".
    # This fires even when no era's top quadrant FLIPS (the gap node needs a flip), so a
    # "what do I keep" question is always answerable from what the owner is drawn back to.
    insights.extend(_resonance_preference(summaries, created))

    # Cross-era TREND insights (first era vs last) — the "how has it changed" arcs.
    insights.extend(_trend_insights(summaries, created))
    return insights


def _resonance_preference(summaries: list[EraSummary], created: float) -> list[Insight]:
    """The quadrant the owner is most DRAWN BACK TO across the whole archive — the
    'which photos do I keep more' answer, pooled over all eras by photo count."""
    from memory_graph.narration.meaning import quadrant_meaning

    if not summaries:
        return []
    pooled = {q: 0.0 for q in range(1, 6)}
    total = 0
    for s in summaries:
        for q in range(1, 6):
            pooled[q] += s.resonant_shares.get(q, 0.0) * s.count
        total += s.count
    if total == 0:
        return []
    top = max(pooled, key=lambda q: pooled[q])
    return [Insight(
        id="ins:resonance:keep", kind="resonance_gap",
        claim=(
            f"The photographs you keep coming back to — the ones that hold you — are "
            f"your {_QUADRANT_NAMES[top].lower()} images: {quadrant_meaning(top)}."
        ),
        direction="higher", magnitude=round(pooled[top] / total, 3),
        resonance_angle="highest pooled resonance share across eras",
        allowed_entities=(), era=None, created_at=created, tags=_GAP_TAGS,
    )]


# Retrieval tags for the critique nodes (they predate the tagging system and shipped with
# EMPTY tags, so `ask` could never reach them by topic — only by era or the broad path).
_STYLE_TAGS = ("style", "feeling", "mood", "emotion", "quadrant", "vibe", "era")
# The resonance gap IS the "which photos do I keep / return to / am drawn back to" answer.
_GAP_TAGS = ("keep", "kept", "return", "drawn", "resonance", "favourite", "favorite",
             "love", "cherish", "feeling", "style", "shot", "most")
_TREND_TAGS = ("change", "changed", "evolved", "over time", "years", "decade", "journey")


def _dir(a: float, b: float) -> str:
    return "higher" if b > a else "lower" if b < a else "flat"


def _trend_insights(summaries: list[EraSummary], created: float) -> list[Insight]:
    """Compare the first and last eras to surface how the photography changed. Each is a
    code-committed claim (direction/magnitude fixed) the LLM narrates within its fence."""
    out: list[Insight] = []
    if len(summaries) < 2:
        return out
    first, last = summaries[0], summaries[-1]
    span = f"{first.era} to {last.era}"

    # 1. People-centrality: how much more (or less) your photos are about people.
    if abs(last.known_share - first.known_share) >= 0.1:
        out.append(Insight(
            id="trend:people", kind="trend_people",
            claim=(
                f"From the {span}, your photographs turned much more toward the people "
                f"you love — a recognized loved one is present far more often now than at "
                f"the start."
                if last.known_share > first.known_share else
                f"From the {span}, your photographs turned away from recognized loved "
                f"ones — they appear far less often now than at the start."
            ),
            direction=_dir(first.known_share, last.known_share),
            magnitude=round(last.known_share - first.known_share, 3),
            allowed_entities=(), era=None, created_at=created,
            tags=_TREND_TAGS + ("people", "family", "loved"),
        ))

    # 2. Focal-length drift: how your framing distance changed.
    focal_moved = (
        first.median_focal and last.median_focal
        and abs(last.median_focal - first.median_focal) >= 6
    )
    if focal_moved:
        out.append(Insight(
            id="trend:focal", kind="trend_focal",
            claim=(
                f"From the {span}, your framing moved wider and closer — you place "
                f"yourself nearer your subject, taking in their surroundings rather than "
                f"singling them out from a distance."
                if last.median_focal < first.median_focal else
                f"From the {span}, your framing moved tighter — you reach in and isolate "
                f"a subject more than you used to."
            ),
            direction=_dir(first.median_focal, last.median_focal),
            magnitude=round(last.median_focal - first.median_focal, 1),
            allowed_entities=(), era=None, created_at=created,
            tags=_TREND_TAGS + ("framing", "focal", "lens", "distance", "close"),
        ))

    # 3. The person who became the heart of your photography (most-photographed in the
    #    last era, if they are prominent). Names are allowed_entities (above the bar).
    if last.top_people:
        name, cnt = last.top_people[0]
        if cnt >= 20:
            out.append(Insight(
                id="trend:person", kind="trend_person",
                claim=(
                    f"By the {last.era}, {name} is the person who appears most in your "
                    f"photographs — the face you return to more than any other."
                ),
                direction="higher", magnitude=float(cnt),
                allowed_entities=(name,), era=last.era, created_at=created,
                tags=_TREND_TAGS + ("people", "who", "most", "person"),
            ))

    # 4. Intimacy vs groups: big-group shots shrinking is a move toward intimacy.
    if abs(last.group_share - first.group_share) >= 0.05:
        out.append(Insight(
            id="trend:groups", kind="trend_groups",
            claim=(
                f"From the {span}, your big-group photographs gave way to closer, "
                f"quieter moments with just a few people."
                if last.group_share < first.group_share else
                f"From the {span}, you gather more people into the frame than you used "
                f"to — bigger groups, more togetherness."
            ),
            direction=_dir(first.group_share, last.group_share),
            magnitude=round(last.group_share - first.group_share, 3),
            allowed_entities=(), era=None, created_at=created,
            tags=_TREND_TAGS + ("groups", "intimacy", "together", "crowd", "people"),
        ))

    return out


# Every kind the critique CAN emit — used by `graph` to clear ALL critique nodes on a
# rebuild, even a kind not produced this run (e.g. resonance_gap when no era's top flips),
# so stale nodes from an earlier build never linger.
CRITIQUE_KINDS: tuple[str, ...] = (
    "era_style", "resonance_gap", "trend_people", "trend_focal", "trend_person",
    "trend_groups",
)


def critique(rows: list[dict], eras: ErasCfg, *, now: float | None = None) -> list[Insight]:
    """End-to-end: rows -> era summaries -> committed insights (facts only; narration later)."""
    return build_insights(summarize_eras(rows, eras), now=now)
