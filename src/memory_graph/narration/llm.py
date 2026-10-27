"""The grounded LLM narration layer (spec.md §8, evaluation.md C.3–C.6).

Orchestrates: build a prompt from ONE insight's facts -> call a LOCAL model
(arm's-length, out-of-process) -> verify the prose is grounded -> on failure regenerate
ONCE -> on a second failure, say so HONESTLY and fall back to the deterministic template.

The backend is pluggable (`Backend` protocol). V1 ships an Apple Foundation Models
adapter (portable) and an optional `claude -f` headless adapter (author-only); both are
held to the SAME groundedness verification. A `CallableBackend` makes the whole loop
unit-testable without any model.

The orchestration is backend-independent, so the honest-fallback behavior and the
groundedness gate are fully tested even where no local model exists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from memory_graph.narration.templates import PhotoFacts, photo_sentence
from memory_graph.narration.verify import verify_narration


class Backend(Protocol):
    def generate(self, prompt: str) -> str | None:  # None => generation failed
        ...


@dataclass
class CallableBackend:
    """A backend wrapping a plain function — used in tests and for simple adapters."""

    fn: Callable[[str], str | None]

    def generate(self, prompt: str) -> str | None:
        return self.fn(prompt)


@dataclass(frozen=True)
class Narration:
    text: str
    grounded: bool            # True if an LLM narration passed verification
    fell_back: bool           # True if we had to use the template (LLM untrusted/absent)
    notice: str | None = None  # the honest "couldn't generate" message when fell_back


@dataclass(frozen=True)
class Critique:
    narrative: str            # the interpretive, personal prose (A — the experience)
    evidence: str             # the grounding facts the narrative rests on (B — eval guard)
    grounded: bool            # the narrative invented no named person / reversed no direction
    fell_back: bool


@dataclass(frozen=True)
class Answer:
    """A grounded answer to a free-form `ask` question: narrative (A) + evidence (B)."""

    narrative: str            # the personal, depth-matched reflection (A)
    evidence: str             # the insight nodes it rests on (B — eval guard)
    grounded: bool
    fell_back: bool
    n_nodes: int = 0          # how many insight nodes were woven (depth signal)


_SYSTEM = (
    "You are a perceptive photography companion. You are given ONLY the facts below "
    "about one photograph. Write two or three warm, second-person sentences about what "
    "the photographer may have felt or intended. You MAY interpret feeling, but you MUST "
    "NOT state any camera setting, number, name, place, or date that is not in the facts, "
    "and you must not contradict them. Do not invent people."
)


def _build_prompt(facts: PhotoFacts, extra: str = "") -> str:
    lines = ["FACTS:", photo_sentence(facts)]
    if facts.quadrant_name:
        lines.append(f"Emotional quadrant: {facts.quadrant_name}.")
    if facts.people:
        lines.append(f"People present (you may name only these): {', '.join(facts.people)}.")
    if extra:
        lines.append(extra)
    return _SYSTEM + "\n\n" + "\n".join(lines)


def narrate_photo(
    facts: PhotoFacts,
    backend: Backend | None,
    *,
    max_regenerations: int = 1,
) -> Narration:
    """Narrate one photo's facts with a grounded LLM, falling back honestly.

    Returns a Narration. If the backend is absent, or fails/regenerates past the limit,
    `fell_back` is True, `text` is the deterministic template sentence, and `notice`
    carries the honest message that the richer narration could not be produced.
    """
    template = photo_sentence(facts)
    if backend is None:
        return Narration(text=template, grounded=False, fell_back=True, notice=None)

    allowed_numbers = facts.numbers()
    allowed_names = list(facts.people)

    attempts = max_regenerations + 1
    sharper = ""
    for _attempt in range(attempts):
        prose = backend.generate(_build_prompt(facts, sharper))
        if prose:
            result = verify_narration(
                prose,
                allowed_numbers=allowed_numbers,
                allowed_names=allowed_names,
                committed_direction=None,  # single-photo narration makes no comparison
            )
            if result.ok:
                return Narration(text=prose.strip(), grounded=True, fell_back=False)
        # Tighten the instruction for the retry.
        sharper = (
            "Your previous attempt stated something not in the facts. Stay strictly "
            "within the facts above; interpret feeling only."
        )

    return Narration(
        text=template,
        grounded=False,
        fell_back=True,
        notice=(
            "Couldn't generate a trustworthy reflection for this one, "
            "so here are the plain facts."
        ),
    )


# --------------------------------------------------------------- longitudinal critique

# The critique voice is DIFFERENT from the dispatch: it is allowed to interpret and make
# a grounded personal leap (the intent doc demands it — "to freeze unposed domestic
# presence before life changed"). It must not read like a table or an analyst's report.
_CRITIQUE_SYSTEM = (
    "You are an expert visual psychologist and the owner's lifelong photographic "
    "companion. From the grounded observations below, write a SHORT, warm, second-person "
    "reflection (one flowing paragraph, ~4-6 sentences) on how their photography — and "
    "what it reveals about them — changed over time.\n\n"
    "VOICE: interpretive and personal, like the example:\n"
    "  \"Over the years your camera turned inward and toward the people you love. You "
    "stopped documenting rooms and started keeping people.\"\n\n"
    "RULES:\n"
    "- Translate technique into FEELING and INTENT, never list statistics. Do NOT write "
    "a table, bullet points, percentages, or photo counts. Numbers are for your "
    "understanding only; keep them out of the prose unless one single number genuinely "
    "sharpens a feeling.\n"
    "- You MAY make an interpretive leap about what the change meant for them as a "
    "person, as long as it is consistent with the observations.\n"
    "- You may ONLY name people listed in ALLOWED PEOPLE. Never invent a name, place, or "
    "date. Never reverse a stated direction (if a share rose, do not say it fell)."
)


def _critique_grounding(insights) -> tuple[str, list[str], list[str]]:
    """Build the grounding bundle for the critique: observation lines (feeling-framed),
    the allowed people, and the committed direction phrases (for verification)."""
    from memory_graph.narration.meaning import quadrant_feeling

    lines: list[str] = []
    allowed_people: list[str] = []
    directions: list[str] = []
    for ins in insights:
        lines.append(ins.claim)
        for name in ins.allowed_entities:
            if name not in allowed_people:
                allowed_people.append(name)
        if ins.direction in ("higher", "lower"):
            directions.append(ins.direction)
    _ = quadrant_feeling  # meaning vocabulary is embedded in the claims upstream
    return "\n".join(f"- {line}" for line in lines), allowed_people, directions


def _evidence_block(insights) -> str:
    """The plain-facts evidence section (B) — the eval guard the owner can check against."""
    rows = [f"- {ins.claim}" for ins in insights]
    return "The evidence behind this:\n" + "\n".join(rows)


def build_critique_prompt(insights) -> str:
    """The narration directive for the longitudinal critique — the voice prompt + the
    grounded per-era observations, framed as notes to write FROM. Handed to a connected
    model (over MCP) so IT composes the reflection; the engine never pre-writes it."""
    grounding, allowed_people, _ = _critique_grounding(list(insights))
    return (
        _CRITIQUE_SYSTEM
        + "\n\nThese observations are NOTES for you to write FROM — not a dataset to "
        "display. Weave them into one flowing paragraph. Do not reproduce them as a "
        "list, a table, or with their numbers.\n"
        + "\nGROUNDED OBSERVATIONS (your notes):\n" + grounding
        + "\n\nALLOWED PEOPLE (you may name only these): "
        + (", ".join(allowed_people) if allowed_people else "(none)")
    )


def narrate_critique(insights, backend: Backend | None, *, max_regenerations: int = 1) -> Critique:
    """Narrate the longitudinal critique: an interpretive paragraph (A) + evidence (B).

    The paragraph is allowed to make a grounded personal leap (the intent voice). It is
    verified only against the honest constraints: no invented named person, no reversed
    direction. On failure (or no backend) it falls back to the plain evidence so the
    user still gets the facts — never a fabricated story.
    """
    evidence = _evidence_block(insights)
    if backend is None or not insights:
        return Critique(narrative="", evidence=evidence, grounded=False, fell_back=True)

    _, allowed_people, _directions = _critique_grounding(insights)
    prompt = build_critique_prompt(insights)

    sharper = ""
    for _attempt in range(max_regenerations + 1):
        prose = backend.generate(prompt + sharper)
        if prose and _critique_ok(prose, allowed_people, insights):
            return Critique(
                narrative=prose.strip(), evidence=evidence, grounded=True, fell_back=False
            )
        sharper = (
            "\n\nYour previous attempt either listed statistics, named someone not "
            "allowed, or reversed a direction. Write flowing prose, name only allowed "
            "people, and keep every stated change in its true direction."
        )
    return Critique(narrative="", evidence=evidence, grounded=False, fell_back=True)


def _critique_ok(prose: str, allowed_people: list[str], insights) -> bool:
    """Honest-leap verification: interpretation is allowed, but no invented named person
    and no reversed committed direction. (Numbers are permitted as evidence.)"""
    import re

    from memory_graph.narration.verify import _COMMON_CAPITALS

    allowed_tokens = {t.lower() for name in allowed_people for t in name.split()}
    grounded_tokens = {
        match.group(1).lower()
        for ins in insights
        for match in re.finditer(r"\b([A-Z][a-z]{2,})\b", ins.claim)
    }
    grounded_words = {
        word.lower()
        for ins in insights
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]+", ins.claim)
    }
    for m in re.finditer(r"([.!?]\s+|^\s*)?\b([A-Z][a-z]{2,})\b", prose):
        if m.group(1) is not None:
            continue   # sentence-initial capitals are grammar, not names
        tok = m.group(2).lower()
        if (tok in _COMMON_CAPITALS or tok in allowed_tokens or tok in grounded_tokens
                or tok in grounded_words):
            continue
        # A mid-sentence capitalized word that is not an allowed person: reject as a
        # possible invented name. (Conservative — favors not fabricating identities.)
        return False
    # Direction-reversal in free, interpretive prose is hard to localize reliably, so we
    # do not hard-check it here; the evidence block (B) is the owner-facing backstop that
    # keeps the narrative honest. Named-person invention IS hard-checked above.
    _ = insights
    return True


# ------------------------------------------------------------------ free-form `ask`

# The `ask` voice is the same companion as the critique, but it answers a SPECIFIC
# question, and its length tracks the question's scope (depth = how many nodes were
# selected). A narrow question gets a sentence or two; a broad one gets a short
# paragraph. Same honesty fence: interpret feeling, make a grounded leap, name only
# allowed people, never invent facts or reverse a direction.
_ASK_SYSTEM = (
    "You are an expert visual psychologist and the owner's lifelong photographic "
    "companion. Answer their QUESTION using ONLY the grounded observations below, in a "
    "warm, second-person voice.\n\n"
    "VOICE: interpretive and personal — translate technique into FEELING and INTENT. "
    "Make a grounded leap about what it reveals about them, consistent with the "
    "observations. Do NOT write a table, bullet points, or a list of statistics; numbers "
    "are for your understanding only and stay out of the prose unless one genuinely "
    "sharpens a feeling.\n\n"
    "LENGTH: match the question's scope. Answer in about {lo} to {hi} sentences — a "
    "narrow question deserves a brief, direct answer; a broad one a short reflective "
    "paragraph. Do not pad.\n\n"
    "RULES: you may ONLY name people listed in ALLOWED PEOPLE; never invent a name, "
    "place, or date, and never reverse a stated direction. If the observations do not "
    "address the question, say so plainly rather than guessing.\n\n"
    "EMPHASIS: the observations are ordered by importance — the FIRST ones are the "
    "strongest signals and deserve the most weight. Give the earlier observations center "
    "stage; mention the later, minor ones only lightly or in passing (do not give a "
    "friend-of-a-friend the same weight as a spouse or a child)."
)

# A SMALL on-device model (Apple Foundation Models, ~3B) needs more scaffolding than a
# large model: explicit length, a name budget, a structure to follow, and concrete
# anti-patterns to avoid. The large-model prompt trusts the narrator to weave; this one
# constrains it so it does not ramble, name-dump, or over-write. Observed FM failure
# modes this targets: listing every allowed name in one breath, purple phrasing
# ("presence hums beneath the surface"), and restating notes instead of building an arc.
_ASK_SYSTEM_COMPACT = (
    "You are the owner's warm, perceptive photographic companion. Using ONLY the "
    "observations below, answer their QUESTION as ONE flowing paragraph of prose. "
    "Write {lo} to {hi} sentences in the second person ('you'), then stop — do not pad. "
    "Your entire reply must be plain sentences: never begin a line with a dash, bullet, "
    "number, or heading.\n\n"
    "Build an arc rather than restating the notes: say how things were, then how they "
    "changed, then what that suggests about the person. "
    "Name at most two or three people — only the most important ones, which are the "
    "first observations; refer to any others as 'a few friends' and never list several "
    "names in a row. "
    "Use plain, sincere language with no ornate metaphors (avoid words like 'etched', "
    "'hums', 'breaths', 'rhythm', 'orbit').\n\n"
    "Never write a table, bullet points, percentages, or counts; never invent a name, "
    "place, or date; never name anyone outside ALLOWED PEOPLE; never reverse a stated "
    "direction. The observations are ordered by importance — weight the first ones most."
)


def _order_by_weight(insights):
    """Order notes so the strongest signals lead — LLMs emphasize earlier content, and a
    major person (high magnitude) must not read with the same weight as a minor one. A
    'supporting cast' aside (no magnitude) always sorts last. Ties keep input order."""
    items = list(insights)

    def weight(ins):
        if ins.id.startswith("relationship:cast:"):
            return -1.0        # the aside goes last, always
        return ins.magnitude if ins.magnitude is not None else 0.0

    return sorted(items, key=lambda i: -weight(i))


def build_answer_prompt(question, insights, target_sentences: tuple[int, int],
                        *, tier: str = "default") -> str:
    """The full narration DIRECTIVE for an `ask` — the voice prompt + grounded notes +
    allowed names. Used both to drive a local backend AND, when no local narrator exists,
    to hand a connected model (Claude/ChatGPT over MCP) a proper instruction so IT writes
    the flowing paragraph rather than tabulating the raw evidence. The observations are
    framed as 'NOTES TO WRITE FROM', not data to display, which is what keeps a hosted
    model from rendering them as a table. Notes are ORDERED BY WEIGHT so the narrator
    emphasizes the people/signals that matter most.

    `tier`: "default" trusts a large model (Claude) to weave; "compact" adds the tighter
    scaffolding a small on-device model (Apple Foundation Models) needs to avoid
    rambling, name-dumping, and over-writing.
    """
    grounding, allowed_people, _ = _critique_grounding(_order_by_weight(insights))
    lo, hi = target_sentences
    system = (_ASK_SYSTEM_COMPACT if tier == "compact" else _ASK_SYSTEM).format(lo=lo, hi=hi)
    return (
        system
        + "\n\nThese observations are NOTES for you to write FROM — not a dataset to "
        "display. Weave them into prose. Do not reproduce them as a list, a table, or "
        "with their numbers.\n"
        + "\nQUESTION: " + question.strip()
        + "\n\nGROUNDED OBSERVATIONS (your notes):\n" + grounding
        + "\n\nALLOWED PEOPLE (you may name only these): "
        + (", ".join(allowed_people) if allowed_people else "(none)")
    )


def narrate_answer(
    question: str,
    insights,
    backend: Backend | None,
    *,
    target_sentences: tuple[int, int] = (1, 3),
    max_regenerations: int = 1,
    tier: str = "default",
) -> Answer:
    """Narrate an answer to `question` from the SELECTED insight nodes (A) + evidence (B).

    Length is steered by `target_sentences` (from ask.select_nodes), so verbosity scales
    with the question's scope. Grounded exactly like the critique: no invented named
    person, no reversed direction; on failure/no-backend it falls back to the evidence so
    the owner still gets honest facts rather than a fabricated story.
    """
    nodes = list(insights)
    evidence = _evidence_block(nodes) if nodes else "No matching observations yet."
    if backend is None or not nodes:
        return Answer(narrative="", evidence=evidence, grounded=False,
                      fell_back=True, n_nodes=len(nodes))

    _, allowed_people, _directions = _critique_grounding(nodes)
    prompt = build_answer_prompt(question, nodes, target_sentences, tier=tier)

    sharper = ""
    for _attempt in range(max_regenerations + 1):
        prose = backend.generate(prompt + sharper)
        if prose and _critique_ok(prose, allowed_people, nodes):
            return Answer(narrative=prose.strip(), evidence=evidence, grounded=True,
                          fell_back=False, n_nodes=len(nodes))
        sharper = (
            "\n\nYour previous attempt either listed statistics, named someone not "
            "allowed, or reversed a direction. Write flowing prose, name only allowed "
            "people, and keep every stated change in its true direction."
        )
    return Answer(narrative="", evidence=evidence, grounded=False,
                  fell_back=True, n_nodes=len(nodes))


# --------------------------------------------------------------------------- backends


def make_backend(cfg_backend: str) -> Backend | None:
    """Construct the configured local backend, or None if unavailable.

    - "foundation_models": Apple Foundation Models via an arm's-length helper (V1
      default, portable to V2/V3). Returns None until the helper is wired (P8 ships the
      orchestration + verification; the helper binary is a follow-up).
    - "claude_cli": headless `claude` (author-only upgrade).
    Any failure to construct returns None, and narration falls back to the template.
    """
    if cfg_backend == "claude_cli":
        return _claude_cli_backend()
    if cfg_backend == "foundation_models":
        return _foundation_models_backend("device")
    if cfg_backend == "foundation_models_pcc":
        return _foundation_models_backend("pcc")
    return None


def _find_claude() -> str | None:
    """Locate the `claude` CLI, even under a stripped PATH.

    When Claude Desktop launches the MCP server it passes a minimal PATH (often just
    /usr/bin:/bin), so `shutil.which('claude')` misses a user install in ~/.local/bin or
    Homebrew. We check PATH first, then the common install locations, and honor an
    explicit MEMORY_GRAPH_CLAUDE_BIN override. This is the author-only upgrade backend;
    V2 users rely on Apple Foundation Models (no `claude` needed)."""
    import os
    import shutil
    from pathlib import Path

    override = os.environ.get("MEMORY_GRAPH_CLAUDE_BIN")
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    found = shutil.which("claude")
    if found:
        return found
    home = Path.home()
    for cand in (home / ".local/bin/claude", Path("/opt/homebrew/bin/claude"),
                 Path("/usr/local/bin/claude"), home / "bin/claude"):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _claude_cli_backend() -> Backend | None:
    claude_bin = _find_claude()
    if claude_bin is None:
        return None

    def _run(prompt: str) -> str | None:
        import subprocess

        try:
            proc = subprocess.run(
                [claude_bin, "-p", "--output-format", "text"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = (proc.stdout or "").strip()
            return out or None
        except Exception:
            return None

    return CallableBackend(_run)


def _foundation_models_backend(tier: str = "device") -> Backend | None:
    """Apple Foundation Models via an arm's-length Swift helper (no key, no third-party
    egress). `tier`: "device" = the on-device ~3B model (works from an unsigned helper);
    "pcc" = the larger Private Cloud Compute server model (privacy-preserving, higher
    capability). PCC requires a code-signed, ENTITLED app, so it is unavailable from a
    dev command-line helper and the backend returns None there — honest fallback. The
    code path is ready for when the engine ships as a signed app. The helper is compiled
    on first use and cached."""
    helper = _ensure_foundation_helper()
    if helper is None:
        return None
    args = [str(helper)] + (["pcc"] if tier == "pcc" else [])

    def _run(prompt: str) -> str | None:
        import subprocess

        try:
            proc = subprocess.run(
                args, input=prompt, capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                return None   # model unavailable / generation failed -> honest fallback
            out = (proc.stdout or "").strip()
            return out or None
        except Exception:
            return None

    return CallableBackend(_run)


def _ensure_foundation_helper():
    """Build (once) and return the path to the compiled Swift narrator, or None.

    Compiles helpers/foundation_narrate.swift to a cached binary under the user data dir.
    A rebuild is triggered only when the source is newer than the binary."""
    import subprocess
    from pathlib import Path

    # llm.py is at src/memory_graph/narration/llm.py -> repo root is parents[3].
    src = Path(__file__).resolve().parents[3] / "helpers" / "foundation_narrate.swift"
    if not src.is_file():
        return None
    try:
        from memory_graph.paths import DATA_DIR
        out_dir = Path(DATA_DIR)
    except Exception:
        out_dir = src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = out_dir / "foundation_narrate"

    fresh = binary.is_file() and binary.stat().st_mtime >= src.stat().st_mtime
    if not fresh:
        import shutil
        if shutil.which("swiftc") is None:
            return None
        try:
            proc = subprocess.run(
                ["swiftc", "-O", "-o", str(binary), str(src)],
                capture_output=True, text=True, timeout=180,
            )
            if proc.returncode != 0 or not binary.is_file():
                return None
        except Exception:
            return None
    return binary
