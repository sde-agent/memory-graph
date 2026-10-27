"""Deterministic, metric-grounded narration facts (spec.md §8, evaluation.md C.2).

This is the TRUSTED layer: every value comes straight from the DB, so a template
sentence can be unit-tested to an exact string. It is both (a) the grounding context
handed to the LLM and (b) the honest fallback shown when the LLM cannot be trusted.

`photo_facts` returns a STRUCTURED fact set (what the verifier checks the LLM against);
`photo_sentence` renders the human-readable one-liner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

_QUADRANT_NAMES = {
    1: "Nostalgic & Reflective",
    2: "Visceral & High-Arousal",
    3: "Empathetic & Connected",
    4: "Solitary & Introspective",
    5: "Provocative & Dynamic",
}

_YEAR_SECONDS = 365.25 * 24 * 3600


@dataclass(frozen=True)
class PhotoFacts:
    """The exact facts the LLM is allowed to restate (and the verifier enforces)."""

    focal_35mm: float | None
    aperture: float | None
    shutter: float | None
    iso: int | None
    golden_hour: bool
    age_years: int | None
    quadrant: int | None
    quadrant_name: str | None
    people: tuple[str, ...] = field(default_factory=tuple)

    def numbers(self) -> list[float]:
        """Every numeric value the prose is permitted to state (for verification)."""
        out: list[float] = []
        for v in (self.focal_35mm, self.aperture, self.shutter, self.iso, self.age_years):
            if v is not None:
                out.append(float(v))
        return out


def _age_years(timestamp: float | None, now: float | None) -> int | None:
    if not timestamp:
        return None
    ref = now if now is not None else datetime.now(UTC).timestamp()
    age = (ref - timestamp) / _YEAR_SECONDS
    return round(age) if age >= 1 else 0


def photo_facts(row: dict, now: float | None = None) -> PhotoFacts:
    """Extract the structured facts from a DB photo row."""
    names: tuple[str, ...] = ()
    raw = row.get("person_names")
    if raw:
        try:
            names = tuple(json.loads(raw)) if isinstance(raw, str) else tuple(raw)
        except Exception:
            names = ()
    quad = row.get("dominant_quadrant")
    return PhotoFacts(
        focal_35mm=row.get("focal_length_35mm"),
        aperture=row.get("aperture"),
        shutter=row.get("shutter_speed"),
        iso=row.get("iso"),
        golden_hour=bool(row.get("golden_hour")),
        age_years=_age_years(row.get("timestamp"), now),
        quadrant=quad,
        quadrant_name=_QUADRANT_NAMES.get(quad) if quad else None,
        people=names,
    )


def photo_sentence(facts: PhotoFacts) -> str:
    """Render the deterministic one-liner from the structured facts."""
    parts: list[str] = []
    if facts.focal_35mm:
        parts.append(f"{round(facts.focal_35mm)}mm")
    if facts.aperture:
        parts.append(f"f/{facts.aperture:.1f}")
    if facts.shutter:
        s = facts.shutter
        parts.append(f"{s:.1f}s" if s >= 1 else f"1/{round(1 / s)}s")
    if facts.golden_hour:
        parts.append("golden hour")
    if facts.age_years and facts.age_years >= 1:
        parts.append(f"{facts.age_years} years ago")

    lead = f"{facts.quadrant_name}. " if facts.quadrant_name else ""
    body = ", ".join(parts)
    who = ""
    if facts.people:
        who = f" With {', '.join(facts.people)}."
    return (lead + body + ("." if body else "") + who).strip()
