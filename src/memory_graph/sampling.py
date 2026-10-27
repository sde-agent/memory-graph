"""Stratified sampling for the tuning run (evaluation.md A.2/A.3, B.6).

To tune the quadrant thresholds and resonance weights we need a BALANCED sample across
eras and sources — not whatever the first N files happen to be. This draws ~target
canonical photos stratified over an (era x source) grid, drawing from a pool that has
ALREADY been cleaned (clutter rejected) and deduped (bursts collapsed), so the sample is
clean-and-unique by construction.

`plan_grid` reports the fillable counts per cell BEFORE committing, so we never promise
a cell we cannot fill and can see the coverage we will get.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from memory_graph.config import ErasCfg
from memory_graph.contracts import Candidate


def _year(ts: float | None) -> int | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, UTC).year


def _cell(cand: Candidate, eras: ErasCfg) -> tuple[str, str] | None:
    era = eras.era_of(_year(cand.timestamp))
    if era is None:
        return None
    return (era, cand.source.value)


@dataclass
class GridPlan:
    available: dict[tuple[str, str], int]   # (era, source) -> how many are available
    allocation: dict[tuple[str, str], int]  # (era, source) -> how many we will sample
    target: int

    def report(self) -> str:
        lines = [f"Stratified sample plan (target {self.target}):"]
        for key in sorted(self.available):
            era, src = key
            lines.append(
                f"  {era:14} {src:8} available {self.available[key]:6}  "
                f"-> sample {self.allocation.get(key, 0)}"
            )
        lines.append(f"  TOTAL planned: {sum(self.allocation.values())}")
        return "\n".join(lines)


def plan_grid(pool: list[Candidate], eras: ErasCfg, target: int) -> GridPlan:
    """Plan an even (era x source) allocation, capped by what each cell actually has.

    Even split across non-empty cells; any shortfall in a thin cell is redistributed to
    cells that still have headroom, so we hit `target` when the pool is large enough.
    """
    available: dict[tuple[str, str], int] = defaultdict(int)
    for c in pool:
        key = _cell(c, eras)
        if key is not None:
            available[key] += 1

    cells = sorted(available)
    allocation: dict[tuple[str, str], int] = {k: 0 for k in cells}
    remaining = target
    # Water-filling: repeatedly give each cell with headroom an equal share.
    while remaining > 0:
        headroom = [k for k in cells if allocation[k] < available[k]]
        if not headroom:
            break
        share = max(1, remaining // len(headroom))
        progressed = False
        for k in headroom:
            take = min(share, available[k] - allocation[k], remaining)
            if take <= 0:
                continue
            allocation[k] += take
            remaining -= take
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return GridPlan(available=dict(available), allocation=allocation, target=target)


def stratified_sample(
    pool: list[Candidate], eras: ErasCfg, target: int, *, seed: int = 0
) -> list[Candidate]:
    """Draw a stratified sample of up to `target` candidates across (era x source)."""
    plan = plan_grid(pool, eras, target)
    by_cell: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in pool:
        key = _cell(c, eras)
        if key is not None:
            by_cell[key].append(c)

    rng = random.Random(seed)
    chosen: list[Candidate] = []
    for key, n in plan.allocation.items():
        items = by_cell.get(key, [])
        rng.shuffle(items)
        chosen.extend(items[:n])
    rng.shuffle(chosen)
    return chosen
