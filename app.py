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

import json
import time
from dataclasses import replace
from pathlib import Path

import streamlit as st

from memory_graph.config import (
    load_config,
    save_local_privacy_exclusions,
    save_local_resonance_profile,
)
from memory_graph.db import all_canonical_rows, connect, get_insights
from memory_graph.narration.ask import select_nodes
from memory_graph.narration.llm import make_backend, narrate_answer
from memory_graph.paths import TASTE_MODEL_PATH
from memory_graph.pipeline import rescorable, rescore_row
from memory_graph.scoring.dispatch import select_dispatch
from memory_graph.scoring.taste import taste_model_status


@st.cache_resource
def _load():
    """Load config + the whole persisted insight graph once per session."""
    cfg = load_config()
    conn = connect()
    try:
        insights = get_insights(conn)
        rows = all_canonical_rows(conn)
    finally:
        conn.close()
    return cfg, insights, rows


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


def _tuned_dispatch(rows, cfg, weights, modifiers):
    """Preview a dispatch under in-memory controls without writing photo rows."""
    tuned_cfg = replace(cfg, resonance_weights=weights, resonance_modifiers=modifiers)
    preview_rows = []
    for row in rows:
        copy = dict(row)
        if rescorable(copy):
            copy.update(rescore_row(copy, tuned_cfg))
        preview_rows.append(copy)
    return select_dispatch(preview_rows, cfg.dispatch, now=time.time(), n=3,
                           privacy=cfg.privacy)


def _render_dispatch_preview(title: str, items, rows_by_id: dict[str, dict]) -> None:
    st.markdown(f"**{title}**")
    for item in items:
        row = rows_by_id.get(item.id, {})
        path = row.get("deriv_path")
        if path and Path(path).is_file():
            st.image(path, width=130)
        st.caption(f"{item.id} · resonance {item.resonance:.3f}")


def _tuning_panel(cfg, rows) -> None:
    """Stage 1 manual global controls and an in-memory top-3 preview."""
    with st.sidebar:
        st.header("Memory tuning")
        st.caption("Preview changes before saving. The graph and photo rows are unchanged.")
        with st.form("resonance_tuning"):
            st.subheader("Emotional affinity")
            q = {
                key: st.slider(label, 0.0, 2.0, float(value), 0.05)
                for key, label, value in (
                    ("q1", "Nostalgic / Reflective", cfg.resonance_weights.q1),
                    ("q2", "Visceral / High-Arousal", cfg.resonance_weights.q2),
                    ("q3", "Empathetic / Connected", cfg.resonance_weights.q3),
                    ("q4", "Solitary / Introspective", cfg.resonance_weights.q4),
                    ("q5", "Provocative / Dynamic", cfg.resonance_weights.q5),
                )
            }
            st.subheader("Behavioral signals")
            known = st.slider("Known person boost", 0.0, 2.0,
                              float(cfg.resonance_modifiers.known_person_bonus), 0.05)
            face = st.slider("Any face boost", 0.0, 2.0,
                             float(cfg.resonance_modifiers.face_present_bonus), 0.05)
            favorite = st.slider("Favorite boost", 0.0, 2.0,
                                 float(cfg.resonance_modifiers.favorite_bonus), 0.05)
            quadrant = st.slider("Quadrant contribution", 0.0, 2.0,
                                 float(cfg.resonance_modifiers.quadrant_weight), 0.05)
            candid = st.slider("Candid bonus", 0.0, 2.0,
                               float(cfg.resonance_modifiers.candid_bonus), 0.05)
            posed = st.slider("Posed penalty", 0.0, 2.0,
                              float(cfg.resonance_modifiers.posed_penalty), 0.05)
            taste_meta = taste_model_status()
            taste_available = TASTE_MODEL_PATH.is_file() and not (
                taste_meta and taste_meta.get("status") == "stale"
            )
            if taste_meta and taste_meta.get("status") == "stale":
                taste_status = "stale — retraining required"
            else:
                taste_status = "available" if taste_available else "not available"
            st.caption("Archive-specific Favorite-trained model: " + taste_status)
            use_taste = st.checkbox(
                "Use learned taste model",
                value=taste_available and cfg.resonance_modifiers.taste_weight > 0,
                disabled=not taste_available,
                help=("Turn off to isolate manual resonance controls from "
                      "Favorite-trained behavior."),
            )
            preview = st.form_submit_button("Preview top 3")
        if preview:
            weights = replace(cfg.resonance_weights, **q)
            modifiers = replace(
                cfg.resonance_modifiers,
                known_person_bonus=known,
                face_present_bonus=face,
                favorite_bonus=favorite,
                quadrant_weight=quadrant,
                candid_bonus=candid,
                posed_penalty=posed,
                taste_weight=1.0 if use_taste else 0.0,
            )
            st.session_state["tuning_profile"] = (weights, modifiers)
        profile = st.session_state.get("tuning_profile")
        if profile:
            weights, modifiers = profile
            st.divider()
            before = select_dispatch(rows, cfg.dispatch, now=time.time(), n=3,
                                     privacy=cfg.privacy)
            after = _tuned_dispatch(rows, cfg, weights, modifiers)
            rows_by_id = {row["id"]: row for row in rows}
            _render_dispatch_preview("Current V1 top 3", before, rows_by_id)
            _render_dispatch_preview("Proposed personal top 3", after, rows_by_id)
            if st.button("Save personal profile"):
                save_local_resonance_profile(weights, modifiers)
                st.cache_resource.clear()
                st.success("Saved to local config.local.toml")
                st.rerun()
            st.caption(
                "Taste model is archive-specific; disabling it is useful for clean weight "
                "experiments and does not delete the model."
            )

        with st.expander("Hard exclusions", expanded=False):
            people = sorted({
                str(name)
                for row in rows
                for name in (
                    json.loads(row.get("person_names") or "[]")
                    if isinstance(row.get("person_names"), str)
                    else (row.get("person_names") or [])
                )
                if name
            })
            subjects = ["children", "nature", "water", "animals", "food", "travel", "celebration"]
            selected_people = st.multiselect(
                "People never surfaced in dispatch", people,
                default=list(cfg.privacy.exclude_people),
            )
            selected_subjects = st.multiselect(
                "Subjects never surfaced in dispatch", subjects,
                default=list(cfg.privacy.exclude_subjects),
            )
            clusters_text = st.text_input(
                "Anonymous cluster IDs (comma-separated)",
                value=", ".join(cfg.privacy.exclude_anonymous_clusters),
            )
            if st.button("Save hard exclusions"):
                clusters = tuple(x.strip() for x in clusters_text.split(",") if x.strip())
                save_local_privacy_exclusions(
                    people=tuple(selected_people), anonymous_clusters=clusters,
                    subjects=tuple(selected_subjects),
                    camera_models=cfg.privacy.exclude_camera_models,
                    places=cfg.privacy.exclude_places,
                )
                st.cache_resource.clear()
                st.success("Saved to local config.local.toml")
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Memory Graph", page_icon="📷", layout="wide")
    st.title("📷 Memory Graph")
    st.caption(
        "Ask about your photography and inner life. The engine selects grounded "
        "observations from your insight graph; an on-device model weaves them into a "
        "reflection. Nothing leaves your Mac."
    )

    cfg, insights, rows = _load()
    if not insights:
        st.error("No insight graph yet. Run `memory-graph graph` after analyzing photos.")
        return

    _tuning_panel(cfg, rows)

    # Which narrators are actually available right now.
    fm = _backend("foundation_models")
    claude = _backend("claude_cli")
    options: dict[str, object] = {}
    if fm is not None:
        options["Apple Foundation Models (on-device)"] = fm
    if claude is not None:
        options["Claude (local CLI)"] = claude
    if claude is None:
        st.info(
            "Claude is not currently detected. In the same terminal, run "
            "`command -v claude`. If it prints a path, run "
            "`export MEMORY_GRAPH_CLAUDE_BIN=\"$(command -v claude)\"` and restart "
            "with `make app`. If it prints nothing, install/authenticate the Claude CLI first."
        )
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
