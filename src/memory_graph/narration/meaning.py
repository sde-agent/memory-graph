"""Psychological meaning of the quadrants and signals (the 'leap' vocabulary).

The intent doc demands that the critique translate TECHNIQUE into EMOTIONAL INTENT —
"50mm at eye-level = preserving a memory as the eye saw it", not "most-shot quadrant
Q1, 44%". This module holds that translation vocabulary: the posture/feeling each
quadrant expresses, in the second-person, feeling-first language the narration builds
on. It is grounding for the LLM's interpretive leap, not a template to recite.
"""

from __future__ import annotations

# Each quadrant's lived meaning — the posture toward the subject, phrased as feeling.
QUADRANT_MEANING = {
    1: "preserving a memory the way the eye lived it — warm, familiar, unforced",
    2: "reaching for awe and scale, the thrill of something larger than yourself",
    3: "leaning in close with tenderness, holding a person rather than a scene",
    4: "seeking stillness and solitude, letting quiet and space do the speaking",
    5: "courting tension and unease, a restless, questioning eye",
}

# Short feeling-words for a quadrant (for compact phrasing).
QUADRANT_FEELING = {
    1: "nostalgic, domestic",
    2: "visceral, awed",
    3: "tender, intimate",
    4: "quiet, solitary",
    5: "restless, provocative",
}


def quadrant_meaning(q: int | None) -> str:
    return QUADRANT_MEANING.get(q or 0, "")


def quadrant_feeling(q: int | None) -> str:
    return QUADRANT_FEELING.get(q or 0, "")
