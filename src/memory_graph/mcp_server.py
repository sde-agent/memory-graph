"""MCP dispatch surface (spec.md §9).

A local stdio MCP server that lets a connected LLM (Claude / ChatGPT) pull the daily
memory. It is strictly READ-ONLY over the engine's own database.

ZERO-EGRESS (hard rule): photo PIXELS never leave this Mac. Showing a photo to a hosted
LLM IS sending it to that service's servers, so this server NEVER returns image bytes.
It returns only TEXT (grounded facts) plus a `http://127.0.0.1` link served by the
local viewer (viewer.py) — the owner clicks it and the photo renders in their browser,
pixels transiting only localhost. The connected LLM only ever sees text.

Privacy: photos marked `private` (in a hidden album) are never returned here; the
dispatch query excludes them by default (default-deny).

Tools:
  daily_dispatch(n)   -> 1-3 resonance-ranked photos as facts + a local view link.
  style_over_time()   -> the longitudinal critique (interpretive narrative + evidence).
  what_changed(era)   -> the insights for one named era.
  ask(question)       -> answer a free-form question by selecting from the insight graph
                         (generate.py) and narrating at a depth matching the question's
                         scope (narrow -> a sentence; broad -> a short reflection).
"""

from __future__ import annotations

# Block the network BEFORE importing the SDK or anything else (local-first guarantee).
from memory_graph.netguard import block_network

block_network()

from datetime import UTC, datetime  # noqa: E402

from memory_graph import db as dbmod  # noqa: E402
from memory_graph import viewer  # noqa: E402
from memory_graph.config import load_config  # noqa: E402
from memory_graph.narration.llm import make_backend, narrate_photo  # noqa: E402
from memory_graph.narration.templates import photo_facts, photo_sentence  # noqa: E402
from memory_graph.scoring.dispatch import select_dispatch  # noqa: E402

_QUADRANT_NAMES = {
    1: "Nostalgic & Reflective",
    2: "Visceral & High-Arousal",
    3: "Empathetic & Connected",
    4: "Solitary & Introspective",
    5: "Provocative & Dynamic",
}


def _narration_text(row: dict, backend) -> str:
    """Grounded narration for one photo: template facts, optionally enriched by a local
    LLM. Always falls back honestly to the deterministic sentence."""
    facts = photo_facts(row)
    if backend is None:
        return photo_sentence(facts)
    result = narrate_photo(facts, backend)
    if result.fell_back and result.notice:
        return f"{result.text}\n({result.notice})"
    return result.text


def build_server():
    """Construct the MCPServer with the dispatch tools registered."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import TextContent

    server = MCPServer(name="memory-graph", version="0.0.1")
    cfg = load_config()
    # A local narrator is used ONLY to enrich single-photo dispatch facts; it is optional
    # and degrades to the deterministic template. The reflection tools (`ask`,
    # `style_over_time`) deliberately do NOT pre-write prose — they return grounded notes
    # for the CONNECTED model to narrate (the insight graph is the product, the narrator
    # is swappable).
    backend = make_backend(cfg.narration.backend)
    # Start the loopback photo viewer so the dispatch links are immediately clickable.
    viewer.ensure_running()

    @server.tool(
        description=(
            "Pull today's 1-3 resonance-ranked memory photos. Returns the facts and a "
            "local view link for each. ZERO-EGRESS: images are NEVER sent; the link opens "
            "the photo in the owner's own browser (pixels stay on their Mac)."
        )
    )
    def daily_dispatch(n: int = 3) -> list:
        conn = dbmod.connect()
        try:
            rows = dbmod.dispatch_candidates(conn)   # excludes private by default
            by_id = {r["id"]: r for r in rows}
            now = datetime.now(UTC).timestamp()
            items = select_dispatch(rows, cfg.dispatch, now=now, n=n)
            out: list = []
            for it in items:
                row = by_id.get(it.id, {})
                facts = _narration_text(row, backend)
                link = viewer.view_url(it.id)
                # TEXT ONLY — never an image. The link renders locally in the browser.
                out.append(TextContent(type="text", text=f"{facts}\nView: {link}"))
                dbmod.mark_dispatched(conn, it.id, now)
            conn.commit()
            if not out:
                return [TextContent(type="text", text="No eligible photos to dispatch yet.")]
            out.insert(0, TextContent(type="text", text=(
                "Your memories for today (open each link to view the photo locally — "
                "the images stay on your Mac):"
            )))
            return out
        finally:
            conn.close()

    @server.tool(
        description=(
            "Longitudinal style critique: how the owner's photography changed across life "
            "eras. RETURNS GROUNDED OBSERVATION NOTES per era (what they shot vs what they "
            "return to) plus a WRITING INSTRUCTION — NOT a finished answer. YOU, the "
            "assistant, must weave the notes into one warm, flowing, second-person "
            "reflection. Do NOT present them as a table or statistics."
        )
    )
    def style_over_time() -> list:
        from mcp.types import TextContent as TC

        from memory_graph.narration.critique import critique
        from memory_graph.narration.llm import build_critique_prompt

        conn = dbmod.connect()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM photos WHERE is_canonical = 1"
            )]
        finally:
            conn.close()
        insights = critique(rows, cfg.eras)
        if not insights:
            return [TC(type="text", text="No analyzed photos spanning your eras yet.")]
        # The engine supplies the grounded per-era notes + the voice; the CONNECTED model
        # composes the reflection (product thesis). No pre-written narrative here.
        return [TC(type="text", text=build_critique_prompt(insights))]

    @server.tool(
        description=(
            "Reflect on the owner's photography and inner life (e.g. 'how has my "
            "photography changed over the last two decades?', 'do I prefer warmer or "
            "cooler colors?', 'which people came and went?'). This tool RETURNS GROUNDED "
            "OBSERVATION NOTES selected from the engine's insight graph — it does NOT "
            "return a finished answer. YOU, the assistant, must WEAVE those notes into "
            "one warm, flowing, second-person reflection. Do NOT show the notes as a "
            "table, bullet list, or statistics — they are raw material for prose, not a "
            "report. Follow the WRITING INSTRUCTION included in the result. TEXT ONLY; no "
            "photo pixels ever leave the Mac."
        )
    )
    def ask(question: str) -> list:
        from mcp.types import TextContent as TC

        from memory_graph.narration.ask import select_nodes
        from memory_graph.narration.generate import GRAPH_KINDS, generate_graph
        from memory_graph.narration.llm import build_answer_prompt

        conn = dbmod.connect()
        try:
            insights = dbmod.get_insights(conn)
            graph_nodes = [i for i in insights if i.kind in GRAPH_KINDS]
            if not graph_nodes:
                # No persisted graph yet — build it in-memory so `ask` still answers.
                rows = [dict(r) for r in conn.execute(
                    "SELECT * FROM photos WHERE is_canonical = 1"
                )]
                insights = generate_graph(rows, cfg.eras, owner=cfg.owner)
        finally:
            conn.close()

        if not insights:
            return [TC(type="text", text="No analyzed photos to reflect on yet.")]

        sel = select_nodes(question, insights, eras=cfg.eras)
        if not sel.nodes:
            return [TC(type="text", text=(
                "I don't have observations that speak to that yet."
            ))]
        # PRODUCT THESIS: the engine supplies the grounded notes + the writing
        # instruction; the CONNECTED model (Claude/ChatGPT) composes the reflection. We
        # never pre-write the answer here — the insight graph is the product, the narrator
        # is swappable, and grounding travels with the notes (the WRITING INSTRUCTION
        # fences invented names and reversed directions).
        directive = build_answer_prompt(question, sel.nodes, sel.target_sentences)
        return [TC(type="text", text=directive)]

    @server.tool(
        description=(
            "Notes on a specific life era (e.g. '2020s'). RETURNS GROUNDED OBSERVATION "
            "NOTES + a WRITING INSTRUCTION — not a finished answer. YOU, the assistant, "
            "weave them into a warm, flowing reflection; do not tabulate. Prefer `ask` "
            "for open-ended questions."
        )
    )
    def what_changed(era: str) -> list:
        from mcp.types import TextContent as TC

        from memory_graph.narration.critique import critique

        conn = dbmod.connect()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM photos WHERE is_canonical = 1"
            )]
        finally:
            conn.close()
        from memory_graph.narration.llm import build_critique_prompt

        insights = [i for i in critique(rows, cfg.eras) if i.era == era]
        if not insights:
            return [TC(type="text", text=f"No insights for '{era}'.")]
        # Return notes + writing instruction, like the other reflection tools — the
        # connected model narrates, never a bare claim list to tabulate.
        return [TC(type="text", text=build_critique_prompt(insights))]

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
