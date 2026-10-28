"""Memory Graph — local reflection app (Streamlit search bar).

Why this exists: over MCP, the connected model (Claude Desktop) owns its own system
prompt and treats tool output as DATA to present, so it tabulated our grounded notes no
matter how we framed them. Here we own the whole loop — question -> retrieve insight
nodes -> call an LLM with OUR voice as the SYSTEM PROMPT -> render — so the narrator
actually narrates. Same architecture as the product thesis (the insight graph is the
product; the narrator is a swappable component), just with us as the one connecting it.

Two narrators are selectable so they can be compared head to head:
  - Apple Foundation Models (on-device, no key, no egress) — the V2 standalone ideal.
  - Local `claude` CLI (dev convenience) — the author's current upgrade.
If Foundation Models reads as well as Claude, this app can ship with ZERO third-party
dependency: a fully standalone, local-first, zero-egress reflection tool.

Zero-egress: only TEXT (the grounded notes) is sent to a narrator, never photo pixels.
Foundation Models runs on device; `claude` is a local process. Nothing leaves the Mac.

Run:  make app   (or: streamlit run app.py)
"""

from __future__ import annotations

import streamlit as st

from memory_graph.config import load_config
from memory_graph.db import connect, get_insights
from memory_graph.narration.ask import select_nodes
from memory_graph.narration.llm import make_backend, narrate_answer


@st.cache_resource
def _load():
    """Load config + the whole persisted insight graph once per session."""
    cfg = load_config()
    conn = connect()
    try:
        insights = get_insights(conn)
    finally:
        conn.close()
    return cfg, insights


@st.cache_resource
def _backend(name: str):
    """Construct (and cache) a narrator backend by name, or None if unavailable."""
    return make_backend(name)


def _answer(question: str, insights, backend, backend_label: str, eras):
    """Retrieve nodes for the question and narrate them with the chosen backend.

    A small on-device model (Apple Foundation Models) gets the tighter 'compact' prompt
    scaffolding; a large model (Claude) gets the default prompt that trusts it to weave.
    """
    sel = select_nodes(question, insights, eras=eras)
    if not sel.nodes:
        return None, None
    tier = "compact" if "on-device" in backend_label else "default"
    ans = narrate_answer(question, sel.nodes, backend,
                         target_sentences=sel.target_sentences, tier=tier)
    return sel, ans


def _render_answer(col, label: str, sel, ans):
    with col:
        st.caption(label)
        if ans is None:
            st.info("No observations speak to that yet — try a different question.")
            return
        if ans.narrative:
            st.write(ans.narrative)
        else:
            st.warning(
                "This narrator is unavailable, so here are the grounded observations "
                "it would have written from:"
            )
            st.write(ans.evidence)
        with st.expander(f"Evidence — the {ans.n_nodes} observations this rests on"):
            st.markdown(ans.evidence.replace("The evidence behind this:\n", ""))


def main() -> None:
    st.set_page_config(page_title="Memory Graph", page_icon="📷", layout="wide")
    st.title("📷 Memory Graph")
    st.caption(
        "Ask about your photography and inner life. The engine selects grounded "
        "observations from your insight graph; an on-device model weaves them into a "
        "reflection. Nothing leaves your Mac."
    )

    cfg, insights = _load()
    if not insights:
        st.error("No insight graph yet. Run `memory-graph graph` after analyzing photos.")
        return

    # Which narrators are actually available right now.
    fm = _backend("foundation_models")
    claude = _backend("claude_cli")
    options: dict[str, object] = {}
    if fm is not None:
        options["Apple Foundation Models (on-device)"] = fm
    if claude is not None:
        options["Claude (local CLI)"] = claude
    if not options:
        st.error(
            "No narrator available. Install the Foundation Models helper (macOS 26+) or "
            "put the `claude` CLI on PATH."
        )
        return

    labels = list(options)
    compare = len(labels) >= 2 and st.toggle(
        "Compare both narrators side by side", value=False,
        help="See how Apple Foundation Models and Claude narrate the SAME grounded notes."
    )
    chosen = None
    if not compare:
        chosen = st.radio("Narrator", labels, horizontal=True)

    examples = [
        "How has my photography changed over the last two decades?",
        "Do I prefer warmer or cooler colors?",
        "Which people came and went in my photos over the years?",
        "What kind of landscapes do I shoot most?",
        "How often am I in my own photos?",
    ]
    st.write("Try:  " + "  ·  ".join(f"*{e}*" for e in examples))

    question = st.text_input(
        "Ask a question", placeholder="e.g. how has my photography changed?",
        label_visibility="collapsed",
    )
    if not question.strip():
        return

    with st.spinner("Reflecting…"):
        if compare:
            cols = st.columns(len(labels))
            for col, label in zip(cols, labels, strict=False):
                sel, ans = _answer(question, insights, options[label], label, cfg.eras)
                _render_answer(col, label, sel, ans)
        else:
            sel, ans = _answer(question, insights, options[chosen], chosen, cfg.eras)
            _render_answer(st.container(), chosen, sel, ans)


if __name__ == "__main__":
    main()
