"""Groundedness verification of LLM narration (spec.md §8, evaluation.md C.3).

Code, not trust. The LLM is handed ONLY an insight's facts and asked to narrate; this
module then checks the returned prose stays inside the fence:
  - no numeric setting (focal/aperture/shutter/ISO/age/count) that is not in the facts;
  - no person name outside the allowed entities (no below-bar / invented names);
  - no comparison that contradicts the insight's committed direction (we forbid the
    OPPOSITE committed terms rather than parse English for direction — direction is
    already fixed in the record).
A failure drops the narration (the caller regenerates once, then falls back honestly).

This is the single most important guardrail in the product: it makes a fluent-but-wrong
narration structurally impossible to ship.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numbers that appear in prose as part of ordinary words/units we never flag as facts.
_NUMBER_RE = re.compile(r"(?<![\w/])(\d+(?:\.\d+)?)")
# Fraction shutter notation like "1/125".
_FRACTION_RE = re.compile(r"\b1/(\d+)\b")

# Direction antonyms: if the insight commits to one, the prose must not assert the other.
_DIRECTION_ANTONYMS = {
    "higher": ("lower", "less", "fewer", "decreased", "declined", "dropped"),
    "lower": ("higher", "more", "greater", "increased", "rose", "grew"),
    "more": ("less", "fewer", "fewer", "decreased", "declined"),
    "less": ("more", "greater", "increased", "rose", "grew"),
    "increased": ("decreased", "declined", "dropped", "fell", "less"),
    "decreased": ("increased", "rose", "grew", "more"),
}


@dataclass(frozen=True)
class GroundednessResult:
    ok: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _allowed_numbers(numbers: list[float]) -> set[float]:
    """Expand each allowed fact number into the forms prose might render it as."""
    allowed: set[float] = set()
    for n in numbers:
        allowed.add(round(n, 3))
        allowed.add(float(round(n)))         # rounded (e.g. 49.6mm -> "50mm")
        if 0 < n < 1:
            allowed.add(float(round(1 / n)))  # shutter 0.008 -> "1/125"
    return allowed


def _prose_numbers(text: str) -> list[float]:
    nums: list[float] = []
    for m in _FRACTION_RE.finditer(text):
        # A "1/125" fraction: record the denominator (matches the 1/shutter form).
        nums.append(float(m.group(1)))
    # Remove fraction spans so their digits are not double-counted as bare numbers.
    stripped = _FRACTION_RE.sub(" ", text)
    for m in _NUMBER_RE.finditer(stripped):
        nums.append(float(m.group(1)))
    return nums


def verify_narration(
    prose: str,
    *,
    allowed_numbers: list[float],
    allowed_names: list[str],
    committed_direction: str | None = None,
    number_tolerance: float = 0.5,
) -> GroundednessResult:
    """Verify prose against an insight's facts. Returns ok=False with reasons on any breach."""
    reasons: list[str] = []
    text = prose.strip()
    lower = text.lower()

    # 1. No invented numbers.
    allowed = _allowed_numbers(allowed_numbers)
    for n in _prose_numbers(text):
        if not any(abs(n - a) <= number_tolerance for a in allowed):
            reasons.append(f"unlisted number: {n}")

    # 2. No invented person names. A name is a MID-SENTENCE capitalized token the facts
    #    did not supply. Sentence-initial capitals ("Over the decade...", "Frozen at...")
    #    are grammar, not names, so we only inspect capitalized words that are NOT the
    #    first word of the text or the first word after sentence-ending punctuation. This
    #    is far more robust than enumerating every English word that can start a sentence.
    allowed_name_tokens = {t.lower() for name in allowed_names for t in name.split()}
    for m in re.finditer(r"([.!?]\s+|^\s*)?\b([A-Z][a-z]{2,})\b", text):
        sentence_initial = m.group(1) is not None
        token = m.group(2)
        if sentence_initial:
            continue
        if token.lower() in _COMMON_CAPITALS:
            continue
        if token.lower() not in allowed_name_tokens:
            reasons.append(f"possible unlisted name: {token}")

    # 3. No contradicted direction.
    if committed_direction:
        antonyms = _DIRECTION_ANTONYMS.get(committed_direction.lower(), ())
        for anti in antonyms:
            if re.search(rf"\b{re.escape(anti)}\b", lower):
                reasons.append(f"contradicts committed direction '{committed_direction}': '{anti}'")
                break

    # De-duplicate while preserving order.
    seen: set[str] = set()
    uniq = [r for r in reasons if not (r in seen or seen.add(r))]
    return GroundednessResult(ok=not uniq, reasons=tuple(uniq))


# Capitalized words that routinely start narration sentences or name quadrants/seasons —
# not person names. Keeps the name check from false-flagging ordinary prose.
_COMMON_CAPITALS = {
    "the", "you", "your", "this", "that", "here", "notice", "shot", "with", "a", "an",
    "nostalgic", "reflective", "visceral", "empathetic", "connected", "solitary",
    "introspective", "provocative", "dynamic", "arousal", "high",
    "golden", "hour", "spring", "summer", "autumn", "fall", "winter",
    "years", "ago", "it", "its", "and", "but", "not", "when", "where", "what",
    "interpretation", "guess", "there", "these", "those", "they", "their",
}
