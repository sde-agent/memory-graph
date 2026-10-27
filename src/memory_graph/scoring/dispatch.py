"""Daily dispatch selection (spec.md §6, evaluation.md A.4/A.5).

Picks the 1-3 resonance-ranked photos to surface today. Guardrails baked in:
  - repeat-avoidance (DR-4): a photo (or its burst siblings) is not re-surfaced within
    a configured window.
  - diversity floor / no-collapse (DR-3): the favorite share of a dispatch is capped so
    favorites are a signal, not a dominator; quadrant variety is preferred.
  - privacy: photos marked `private` (hidden album) are never selected for off-device
    dispatch (they are served only via a local fallback, not here).
  - reproducibility: selection is a deterministic function of (DB state, date, seed),
    so a dispatch can be explained and audited after the fact.

Selection is pure over a list of candidate rows, so it is unit-testable without a DB.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from memory_graph.config import DispatchCfg


@dataclass(frozen=True)
class DispatchItem:
    id: str
    resonance: float
    dominant_quadrant: int | None
    favorite: bool
    source: str


def _recently_dispatched(row: dict[str, Any], now: float, window_days: int) -> bool:
    last = row.get("last_dispatched")
    if not last:
        return False
    return (now - last) < window_days * 86400


def select_dispatch(
    rows: list[dict[str, Any]],
    cfg: DispatchCfg,
    *,
    now: float,
    n: int | None = None,
) -> list[DispatchItem]:
    """Select the daily dispatch from candidate photo rows.

    `rows` are canonical, non-private photo rows with a `resonance` score. Returns up to
    `n` items (default cfg.count_max), honoring repeat-avoidance, the favorite-share
    cap, and quadrant diversity, deterministically for a given (rows, now, seed).
    """
    target = n if n is not None else cfg.count_max
    target = max(cfg.count_min, min(cfg.count_max, target))

    # Eligible: canonical, not private, not recently dispatched, has a resonance score.
    eligible = [
        r
        for r in rows
        if r.get("is_canonical")
        and not r.get("private")
        and r.get("resonance") is not None
        and not _recently_dispatched(r, now, cfg.repeat_window_days)
    ]
    if not eligible:
        return []

    rng = random.Random(cfg.seed)
    # Rank by resonance; a tiny seeded jitter breaks exact ties reproducibly.
    ranked = sorted(
        eligible,
        key=lambda r: (r["resonance"] + rng.random() * 1e-9),
        reverse=True,
    )

    max_favorites = int(target * cfg.max_favorite_share)
    chosen: list[dict[str, Any]] = []
    fav_count = 0
    seen_quadrants: set[int] = set()

    # First pass: take the top items, enforcing the favorite cap and preferring new
    # quadrants (diversity floor) when a slot would otherwise duplicate a quadrant.
    deferred: list[dict[str, Any]] = []
    for r in ranked:
        if len(chosen) >= target:
            break
        is_fav = bool(r.get("favorite"))
        if is_fav and fav_count >= max_favorites:
            continue
        q = r.get("dominant_quadrant")
        if q in seen_quadrants and len(chosen) < target:
            deferred.append(r)  # prefer variety; revisit if we run short
            continue
        chosen.append(r)
        fav_count += 1 if is_fav else 0
        if q is not None:
            seen_quadrants.add(q)

    # Backfill from deferred (and any remaining) if diversity left us short.
    if len(chosen) < target:
        for r in deferred + ranked:
            if len(chosen) >= target:
                break
            if r in chosen:
                continue
            if bool(r.get("favorite")) and fav_count >= max_favorites:
                continue
            chosen.append(r)
            fav_count += 1 if r.get("favorite") else 0

    return [
        DispatchItem(
            id=r["id"],
            resonance=float(r["resonance"]),
            dominant_quadrant=r.get("dominant_quadrant"),
            favorite=bool(r.get("favorite")),
            source=r.get("source", ""),
        )
        for r in chosen[:target]
    ]
