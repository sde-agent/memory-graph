"""Command-line entry point.

P0 provides only `init` (create the database and data dirs) and `doctor` (report
config + environment health). Later phases register their subcommands here. Every
command blocks the network first — the engine is local-first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from memory_graph import __version__
from memory_graph.netguard import block_network


def _cmd_init(_args: argparse.Namespace) -> int:
    from memory_graph.db import connect, init_db

    conn = connect()
    init_db(conn)
    conn.close()
    print("Initialized the memory graph database.")
    return 0


def _cmd_ingest_library(args: argparse.Namespace) -> int:
    from memory_graph.config import load_config
    from memory_graph.ingest.library import LibraryStats, stream_library

    cfg = load_config()
    stats = LibraryStats()
    count = 0
    # stream_library builds its own stats internally; for a quick probe we just count.
    try:
        for cand in stream_library(cfg, dbfile=args.dbfile):
            count += 1
            if args.limit and count >= args.limit:
                break
            if cand.thumb is not None:
                cand.release_thumb()
    except ImportError:
        print("osxphotos is not available; cannot read the Photos library.")
        return 1
    except Exception as exc:  # noqa: BLE001 — report, do not crash the probe
        print(f"library ingest error: {exc}")
        return 1
    suffix = f" (limited to {args.limit})" if args.limit else ""
    print(f"library: produced {count} candidate(s){suffix}")
    _ = stats
    return 0


def _cmd_ingest_hdd(args: argparse.Namespace) -> int:
    from memory_graph.config import load_config
    from memory_graph.ingest.hdd import HddNotReadOnly, stream_hdd

    cfg = load_config()
    count = 0
    try:
        for cand in stream_hdd(args.root, cfg):
            count += 1
            if args.limit and count >= args.limit:
                break
            if cand.thumb is not None:
                cand.release_thumb()
    except HddNotReadOnly as exc:
        print(f"HDD not read-only: {exc}")
        return 1
    suffix = f" (limited to {args.limit})" if args.limit else ""
    print(f"hdd: produced {count} candidate(s){suffix}")
    return 0


def _cmd_cluster_hdd(args: argparse.Namespace) -> int:
    from memory_graph.cluster.bursts import cluster_and_elect
    from memory_graph.config import load_config
    from memory_graph.filters.rejection import classify_clutter
    from memory_graph.ingest.hdd import HddNotReadOnly, stream_hdd_with_stats

    cfg = load_config()
    try:
        cands, stats = stream_hdd_with_stats(args.root, cfg)
    except HddNotReadOnly as exc:
        print(f"HDD not read-only: {exc}")
        return 1
    if args.limit:
        cands = cands[: args.limit]
    kept = [c for c in cands if not classify_clutter(c, cfg).any]
    rejected = len(cands) - len(kept)
    shots = cluster_and_elect(kept, cfg)
    multi = [s for s in shots if s.size > 1]
    print(
        f"hdd: scanned {stats.seen}, ingested {stats.yielded}, "
        f"decode-skipped {stats.skipped_decode}"
    )
    print(f"rejection: kept {len(kept)}, flagged {rejected}")
    print(
        f"clustering: {len(shots)} shots from {len(kept)} photos; "
        f"{len(multi)} multi-frame bursts"
    )
    if multi:
        biggest = max(multi, key=lambda s: s.size)
        print(f"  largest burst: {biggest.size} frames -> canonical {biggest.representative.id}")
    return 0


def _cmd_analyze_hdd(args: argparse.Namespace) -> int:
    from memory_graph.cluster.bursts import cluster_and_elect
    from memory_graph.config import load_config
    from memory_graph.db import connect, count_photos, init_db, upsert_photo
    from memory_graph.filters.rejection import classify_clutter
    from memory_graph.ingest.hdd import HddNotReadOnly, stream_hdd_with_stats
    from memory_graph.pipeline import analyze_shot

    cfg = load_config()
    try:
        cands, stats = stream_hdd_with_stats(args.root, cfg)
    except HddNotReadOnly as exc:
        print(f"HDD not read-only: {exc}")
        return 1
    if args.limit:
        cands = cands[: args.limit]
    kept = [c for c in cands if not classify_clutter(c, cfg).any]
    shots = cluster_and_elect(kept, cfg)
    from memory_graph.paths import TASTE_MODEL_PATH
    from memory_graph.scoring.taste import TasteModel

    taste_model = TasteModel.load(TASTE_MODEL_PATH)
    if taste_model is not None:
        print("using trained taste model")
    conn = connect()
    init_db(conn)
    for shot in shots:
        row = analyze_shot(shot, cfg, use_vision=not args.no_vision, taste_model=taste_model)
        upsert_photo(conn, row)
        shot.representative.release_thumb()
    conn.commit()
    total = count_photos(conn)
    conn.close()
    print(f"analyzed {len(shots)} shots (from {len(kept)} kept photos); db now holds {total}")
    return 0


def _cmd_train_taste(args: argparse.Namespace) -> int:
    import numpy as np

    from memory_graph import vision
    from memory_graph.config import load_config
    from memory_graph.ingest.library import LibraryStats, candidate_from_photo
    from memory_graph.paths import TASTE_MODEL_PATH
    from memory_graph.scoring.taste import evaluate

    if not vision.available():
        print("Apple Vision unavailable; cannot embed photos for the taste model.")
        return 1
    cfg = load_config()
    import osxphotos

    db = osxphotos.PhotosDB()
    embeddings, labels = [], []
    scanned = 0
    for p in db.photos():
        if args.limit and scanned >= args.limit:
            break
        scanned += 1
        cand = candidate_from_photo(p, cfg, LibraryStats())
        if cand is None or cand.thumb is None:
            continue
        emb = vision.feature_print(cand.thumb)
        cand.release_thumb()
        if emb is None:
            continue
        embeddings.append(emb)
        labels.append(1 if (cand.library_meta or {}).get("favorite") else 0)
    if len(labels) < 20 or sum(labels) < 5:
        print(f"Not enough data to train (got {len(labels)} photos, {sum(labels)} favorites).")
        return 1
    model, ev = evaluate(np.array(embeddings), np.array(labels), seed=cfg.dispatch.seed)
    model.save(TASTE_MODEL_PATH)
    print(f"taste model trained on {ev.n} photos.")
    print(f"  base rate (favorites): {ev.base_rate:.1%}  ({ev.n_favorites} favorites)")
    print(f"  held-out AUC: {ev.auc:.3f}  (0.5 = chance; set the success bar vs base rate)")
    print(f"  saved to {TASTE_MODEL_PATH}")
    return 0


def _cmd_sample_run(args: argparse.Namespace) -> int:
    """Stratified 10k tuning run: cache-ingest both sources -> reject -> dedup ->
    stratified-sample -> optics top-up -> analyze (Vision) -> persist."""
    from memory_graph.cluster.bursts import cluster_and_elect
    from memory_graph.config import load_config
    from memory_graph.db import connect, count_photos, init_db, upsert_photo
    from memory_graph.filters.rejection import classify_clutter
    from memory_graph.ingest.archive_cache import (
        available as cache_available,
    )
    from memory_graph.ingest.archive_cache import (
        top_up_optics,
    )
    from memory_graph.paths import TASTE_MODEL_PATH
    from memory_graph.sampling import plan_grid, stratified_sample
    from memory_graph.scoring.taste import TasteModel

    cfg = load_config()

    # 1. Gather the pool from both sources with the CHEAP rejection gate only (no Vision)
    #    — so we never run Vision on the ~50k we will sample away. For memory, keep the
    #    pool as lightweight records (id, source, timestamp) and re-decode the sampled
    #    few; but since previews are cheap we keep thumbs and release rejected ones.
    pool = []
    if cache_available():
        from memory_graph.ingest.archive_cache import build_pool_parallel

        workers = min(8, (os.cpu_count() or 4))
        hdd_kept, rejected = build_pool_parallel(
            cfg, lambda c: not classify_clutter(c, cfg, cheap_only=True).any, workers=workers
        )
        pool.extend(hdd_kept)
        print(f"HDD cache ({workers} workers): {len(hdd_kept)} kept, {rejected} cheap-rejected")
    else:
        print("HDD cache not found; skipping HDD source.")

    if not args.hdd_only:
        from memory_graph.ingest.library import build_pool_parallel as lib_build_parallel

        workers = min(8, (os.cpu_count() or 4))
        try:
            lib_kept, lib_rej, lib_total = lib_build_parallel(
                cfg,
                lambda c: not classify_clutter(c, cfg, cheap_only=True).any,
                workers=workers,
                dbfile=args.dbfile,
            )
        except FileNotFoundError as exc:
            print("Photos Library database not found.")
            print("Open Photos once, set the intended library as System Photo Library, "
                  "grant Full Disk Access to Terminal, or pass --dbfile to its database.")
            print(f"details: {exc}")
            return 1
        pool.extend(lib_kept)
        print(
            f"Library ({workers} workers): {len(lib_kept)} kept, {lib_rej} cheap-rejected "
            f"of {lib_total}"
        )

    # 2. Dedup: burst-collapse time-adjacent frames, THEN a global near-duplicate pass
    #    that merges pHash-duplicates across time/source (catches the dupes that leaked).
    from memory_graph.cluster.bursts import global_dedup

    shots = cluster_and_elect(pool, cfg)
    before_global = len(shots)
    shots = global_dedup(shots, cfg)
    canonicals = [s.representative for s in shots]
    print(
        f"after dedup: {len(canonicals)} canonicals from {len(pool)} kept "
        f"(burst {before_global} -> global {len(shots)})"
    )

    # 3. Stratified sample across (era x source).
    target = len(canonicals) if args.all else args.target
    plan = plan_grid(canonicals, cfg.eras, target)
    print(plan.report())
    sample = stratified_sample(canonicals, cfg.eras, target, seed=cfg.dispatch.seed)
    print(f"sampled {len(sample)}")

    # 4. Optics top-up for sampled HDD photos (batched read off the read-only drive).
    hdd_sample = [c for c in sample if c.source.value == "hdd"]
    topped = top_up_optics(hdd_sample)
    print(f"optics top-up: {topped}/{len(hdd_sample)} HDD photos")

    # 5. Re-decode the sampled thumbnails (released during pool-building), run the FULL
    #    rejection gate (now incl. Vision document/text) on the small sample, then
    #    analyze + persist. Resumable: skip already-analyzed ids.
    taste_model = TasteModel.load(TASTE_MODEL_PATH)
    conn = connect()
    init_db(conn)
    from memory_graph.db import photo_ids
    from memory_graph.pipeline import analyze_candidate

    done = photo_ids(conn)
    n = vision_rej = 0
    for cand in sample:
        if cand.id in done:
            continue
        if not _rehydrate_thumb(cand, cfg):
            continue
        # Full rejection gate (Vision) on the sample only.
        if not args.no_vision and classify_clutter(cand, cfg).any:
            vision_rej += 1
            cand.release_thumb()
            continue
        row = analyze_candidate(cand, cfg, use_vision=not args.no_vision, taste_model=taste_model)
        upsert_photo(conn, row)
        cand.release_thumb()
        n += 1
        if n % 500 == 0:
            conn.commit()
            print(f"  analyzed {n}...")
    conn.commit()
    total = count_photos(conn)
    conn.close()
    print(f"done: analyzed {n}, vision-rejected {vision_rej} this run; db now holds {total}")
    return 0


def _rehydrate_thumb(cand, cfg) -> bool:
    """Re-decode a sampled candidate's thumbnail. Both sources released their pixels
    during pool-building so dedup stayed lightweight; here we re-decode only the sampled
    few. HDD from the cached preview; Library from the stashed local derivative path
    (never PhotosDB, never a master)."""
    if cand.thumb is not None:
        return True
    try:
        from PIL import Image

        from memory_graph.ingest.imaging import to_thumb

        if cand.source.value == "hdd":
            from memory_graph.ingest.archive_cache import _PREVIEW_DIR

            uuid = cand.id.split(":", 1)[1]
            path = _PREVIEW_DIR / f"{uuid}.jpg"
        else:
            path = cand.exif.get("deriv_path")
        if not path or not os.path.isfile(path):
            return False
        with open(path, "rb") as fh:
            cand.thumb = to_thumb(Image.open(fh), cfg.ingest.thumb_long_edge)
        return True
    except Exception:
        return False


def _cmd_dispatch(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from memory_graph.config import load_config
    from memory_graph.db import connect, dispatch_candidates
    from memory_graph.scoring.dispatch import select_dispatch

    cfg = load_config()
    conn = connect()
    rows = dispatch_candidates(conn)
    by_id = {r["id"]: r for r in rows}
    items = select_dispatch(rows, cfg.dispatch, now=datetime.now(UTC).timestamp(), n=args.n or None)
    conn.close()
    if not items:
        print("No eligible photos to dispatch (analyze some first).")
        return 0
    print(f"Today's dispatch ({len(items)}):")
    for it in items:
        row = by_id.get(it.id, {})
        print(f"  [{it.source}] q{it.dominant_quadrant} resonance={it.resonance:.3f} "
              f"fav={it.favorite} ref={row.get('original_ref', '')[:48]}")
    return 0


def _cmd_critique(args: argparse.Namespace) -> int:
    from memory_graph.config import load_config
    from memory_graph.db import connect, init_db, insert_insights
    from memory_graph.narration.critique import critique

    cfg = load_config()
    conn = connect()
    init_db(conn)   # ensure the insights schema (incl. the tags column) is migrated
    rows = [dict(r) for r in conn.execute("SELECT * FROM photos WHERE is_canonical = 1")]
    insights = critique(rows, cfg.eras)
    insert_insights(conn, insights)
    conn.commit()
    conn.close()
    if not insights:
        print("No insights yet (analyze photos spanning your eras first).")
        return 0
    from memory_graph.narration.llm import make_backend, narrate_critique

    backend = make_backend(cfg.narration.backend) or make_backend("claude_cli")
    crit = narrate_critique(insights, backend)
    if crit.narrative:
        print(crit.narrative)
        print()
    elif crit.fell_back:
        print("(No narrative backend available — showing the evidence only.)\n")
    print(crit.evidence)
    return 0


def _cmd_rescore(_args: argparse.Namespace) -> int:
    """Recompute the quadrant vector + resonance for every canonical photo from stored
    signals (no Vision re-decode) after an optics/config fix — e.g. a camera crop factor.

    Faithful because the content cues (scene_labels) and the horizon dutch gate are
    persisted. Rows written before those columns existed are NOT re-scorable and are
    reported so they can be re-analyzed from pixels instead."""
    from memory_graph.config import load_config
    from memory_graph.db import all_canonical_rows, connect, init_db, update_photo_fields
    from memory_graph.pipeline import rescorable, rescore_row

    cfg = load_config()
    conn = connect()
    init_db(conn)
    rows = all_canonical_rows(conn)
    changed = skipped = quad_moved = 0
    for row in rows:
        if not rescorable(row):
            skipped += 1
            continue
        new = rescore_row(row, cfg)
        if new["dominant_quadrant"] != row.get("dominant_quadrant"):
            quad_moved += 1
        update_photo_fields(conn, row["id"], new)
        changed += 1
        if changed % 1000 == 0:
            conn.commit()
            print(f"  re-scored {changed}...")
    conn.commit()
    conn.close()
    print(f"re-scored {changed} photos ({quad_moved} changed dominant quadrant).")
    if skipped:
        print(f"skipped {skipped} rows lacking persisted re-score signals — "
              f"re-analyze them from pixels (run `sample-run` to refresh).")
    return 0


def _cmd_backfill_gps(args: argparse.Namespace) -> int:
    """Set `has_gps` on EXISTING rows from source metadata (no pixels, no masters).

    HDD rows: latitude/longitude from the prototype's archive_records.json (keyed by uuid).
    Library rows: photo.latitude from osxphotos (metadata only). Golden hour is only
    meaningful where GPS exists, so this flag is what lets nodes gate on it.
    """
    from memory_graph.db import connect, init_db
    from memory_graph.ingest.archive_cache import _RECORDS

    records_path = args.archive_records or _RECORDS
    conn = connect()
    init_db(conn)
    rows = conn.execute("SELECT id, source FROM photos").fetchall()

    hdd_gps: set[str] = set()
    if records_path is not None and Path(records_path).is_file():
        data = json.loads(Path(records_path).read_text())
        for rec in data["records"]:
            if rec.get("latitude") is not None and rec.get("longitude") is not None:
                hdd_gps.add(rec["uuid"])
    else:
        print("HDD records not found (set --archive-records); HDD rows left unchanged.")

    lib_gps: set[str] = set()
    if any(r["source"] == "library" for r in rows):
        import osxphotos

        for p in osxphotos.PhotosDB().photos():
            if p.latitude is not None and p.longitude is not None:
                lib_gps.add(p.uuid)

    updated = {"hdd": 0, "library": 0}
    for r in rows:
        uuid = r["id"].split(":", 1)[1]
        if r["source"] == "hdd" and records_path is not None and Path(records_path).is_file():
            has = 1 if uuid in hdd_gps else 0
        elif r["source"] == "library":
            has = 1 if uuid in lib_gps else 0
        else:
            continue
        conn.execute("UPDATE photos SET has_gps = ? WHERE id = ?", (has, r["id"]))
        updated[r["source"]] += 1
    conn.commit()
    conn.close()
    print(f"has_gps set: hdd {updated['hdd']} rows, library {updated['library']} rows.")
    return 0


def _holdout_rows(conn, ids):
    rows = {}
    for pid in ids:
        r = conn.execute("SELECT * FROM photos WHERE id = ?", (pid,)).fetchone()
        if r is not None:
            rows[pid] = dict(r)
    return rows


def _cmd_holdout_build(args: argparse.Namespace) -> int:
    """Choose the blinded items for one round (evaluation.md Parts U and V). One-shot."""
    import time

    from memory_graph.config import load_config
    from memory_graph.db import connect, init_db
    from memory_graph.evaluation.holdout import build_set

    cfg = load_config()
    conn = connect()
    init_db(conn)
    try:
        out = build_set(conn, cfg, now=time.time(), round_no=args.round)
    except (FileExistsError, KeyError) as exc:
        print(str(exc))
        return 1
    finally:
        conn.close()
    spec = out["spec"]
    print(f"round {args.round} built: {len(out['items'])} blinded items from a pool of "
          f"{spec['pool_size']} untouched photos (seed {spec['seed']}).")
    print(f"Next: `memory-graph holdout-label --round {args.round}` opens the page.")
    return 0


def _cmd_holdout_label(args: argparse.Namespace) -> int:
    """Open the blinded labelling page on this Mac (127.0.0.1 only). Resumable."""
    import json

    from memory_graph.evaluation import holdout as H
    from memory_graph.evaluation.holdout_app import serve

    set_file, label_file = H.round_files(args.round)
    if not set_file.is_file():
        print(f"No round {args.round} yet. Run `memory-graph holdout-build --round {args.round}`.")
        return 1
    # Round 3 (Part W) asks every photo for all feelings that apply; rounds 1 and 2 are
    # the single-choice page.
    serve(json.loads(set_file.read_text()), label_file, multi=args.round == 3)
    return 0


def _cmd_holdout_score(_args: argparse.Namespace) -> int:
    """Pool every round with labels and apply the pre-registered rule (Parts U and V)."""
    import json

    from memory_graph.db import connect
    from memory_graph.evaluation import holdout as H

    rounds = []
    for n in sorted(H.ROUND_SEEDS):
        set_file, label_file = H.round_files(n)
        if set_file.is_file():
            rounds.append((json.loads(set_file.read_text()), label_file))
    if not rounds:
        print("No holdout rounds yet. Run `memory-graph holdout-build --round 1`.")
        return 1
    conn = connect()
    ids = [it["id"] for set_obj, _ in rounds for it in set_obj["items"]]
    rows = _holdout_rows(conn, ids)
    conn.close()
    rep = H.score_rounds(rounds, rows)
    for r in rep["rounds"]:
        print(f"round {r['round']}: top-3 {r['top']}/{r['expected_top']}, "
              f"controls {r['control']}/{r['expected_control']} labelled")
    if rep["decision"] == "INCOMPLETE":
        print("Not enough labels yet to score (need at least one top-3 and one control).")
        return 0
    k_top = sum(r["k_top"] for r in rep["rounds"])
    k_ctrl = sum(r["k_control"] for r in rep["rounds"])
    print(f"pooled: P(worthy | engine top-3) = {rep['p_top']:.0%} "
          f"({k_top}/{rep['labelled_top']})")
    print(f"pooled: P(worthy | random control) = {rep['p_control']:.0%} "
          f"({k_ctrl}/{rep['labelled_control']})")
    print(f"lift = {rep['lift'] * 100:+.0f} points  (95% bound "
          f"{rep['lower'] * 100:+.0f} to {rep['upper'] * 100:+.0f})")
    print(f"DECISION (pooled, pre-registered rule): {rep['decision']}")
    q = rep["quadrant"]
    if q["rate"] is not None:
        lo, hi = q["wilson95"]
        print(f"single-feeling match, rounds 1-2, yes answers (context): {q['agree']}/{q['n']} = "
              f"{q['rate']:.0%} (95% {lo:.0%} to {hi:.0%})")
    # Round 3 multiple-feelings test (Part W), reported separately from the yes/no decision.
    set3, label3 = H.round_files(3)
    if set3.is_file() and label3.is_file():
        f = H.score_feelings(json.loads(set3.read_text()), label3, rows)
        if f["decision"] != "INCOMPLETE":
            print(f"\nmultiple feelings, round 3 ({f['n']} photos):")
            print(f"  engine's top feeling is among the owner's picks: {f['observed']:.0%}")
            print(f"  chance, given how many feelings were picked: {f['chance']:.0%}")
            print(f"  lift = {f['lift'] * 100:+.0f} points (95% bound {f['lower'] * 100:+.0f} "
                  f"to {f['upper'] * 100:+.0f})")
            print(f"  average feelings picked: {f['avg_feelings']:.1f}; all five on "
                  f"{f['all_five_share']:.0%} of photos")
            if f["padding_flag"]:
                print("  WARNING: average 4 or more feelings, so this test is flagged "
                      "uninformative.")
            print(f"  FEELINGS DECISION: {f['decision']} (does not change the yes/no decision)")
    return 0


def _cmd_reidentify(args: argparse.Namespace) -> int:
    """Match local face crops to Library names and cluster recurring anonymous faces."""
    from memory_graph.config import load_config
    from memory_graph.db import connect, init_db
    from memory_graph.identity.reidentify import (
        IdentityUnavailable,
        cluster_persisted,
        reidentify,
    )

    cfg = load_config()
    conn = connect()
    try:
        init_db(conn)
        if args.cluster_only:
            clusters = cluster_persisted(
                conn, cfg,
                anonymous_threshold=args.anonymous_threshold,
                anonymous_min_cluster_size=args.anonymous_min_cluster_size,
            )
            print(f"reidentify: clustered persisted vectors into {clusters} anonymous clusters")
            return 0
        stats = reidentify(
            conn, cfg,
            limit=args.limit,
            skip_gallery=args.skip_gallery,
            recognition_min_similarity=args.recognition_min_similarity,
            anonymous_threshold=args.anonymous_threshold,
            anonymous_min_cluster_size=args.anonymous_min_cluster_size,
        )
    except IdentityUnavailable as exc:
        print(f"reidentify unavailable: {exc}")
        return 1
    finally:
        conn.close()
    print(
        f"reidentify: scanned {stats.scanned}, processed {stats.processed}, "
        f"faces {stats.detected_faces}, named {stats.named_faces}, "
        f"anonymous clusters {stats.anonymous_clusters}"
    )
    if stats.skipped_no_derivative:
        print(f"skipped {stats.skipped_no_derivative} rows without a local derivative")
    if stats.failed_decode:
        print(f"failed {stats.failed_decode} photo(s) during local identity processing")
    print("feature-print identity remains experimental until precision is calibrated.")
    return 0


def _cmd_graph(_args: argparse.Namespace) -> int:
    """Generate and persist the INSIGHT GRAPH — the population of nodes `ask` selects
    from. Replaces the previous graph nodes so stale observations never linger."""
    from memory_graph.config import load_config
    from memory_graph.db import connect, init_db, replace_insights_of_kinds
    from memory_graph.narration.critique import CRITIQUE_KINDS, critique
    from memory_graph.narration.generate import GRAPH_KINDS, generate_graph

    cfg = load_config()
    conn = connect()
    init_db(conn)   # ensure the insights schema (incl. the tags column) is migrated
    rows = [dict(r) for r in conn.execute("SELECT * FROM photos WHERE is_canonical = 1")]
    # Rebuild the WHOLE insight graph in one command: the generated graph nodes AND the
    # critique's per-era / trend nodes. They were previously split across `graph` and
    # `critique`, which risked a half-refreshed graph (e.g. stale number-laden claims).
    # Clear the FULL known kind sets (not just kinds produced this run) so a kind that
    # stops being emitted — e.g. resonance_gap when no era's top flips — leaves no stale
    # rows behind.
    nodes = generate_graph(rows, cfg.eras, owner=cfg.owner)
    crit_nodes = critique(rows, cfg.eras)
    replace_insights_of_kinds(conn, nodes, GRAPH_KINDS)
    replace_insights_of_kinds(conn, crit_nodes, CRITIQUE_KINDS)
    conn.commit()
    conn.close()
    total = len(nodes) + len(crit_nodes)
    print(f"insight graph: committed {total} nodes "
          f"({len(nodes)} graph + {len(crit_nodes)} critique) "
          f"across {len(GRAPH_KINDS) + len(CRITIQUE_KINDS)} dimensions.")
    if not nodes and not crit_nodes:
        print("(no nodes — analyze photos first)")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """Answer a free-form question by selecting from the insight graph and narrating at
    a depth that matches the question's scope."""
    from memory_graph.config import load_config
    from memory_graph.db import connect, get_insights
    from memory_graph.narration.ask import select_nodes
    from memory_graph.narration.llm import make_backend, narrate_answer

    cfg = load_config()
    conn = connect()
    insights = get_insights(conn)
    conn.close()
    if not insights:
        print("No insight graph yet — run `memory-graph graph` after analyzing photos.")
        return 0

    sel = select_nodes(args.question, insights, eras=cfg.eras)
    if not sel.nodes:
        print("I don't have observations that speak to that yet.")
        return 0

    backend = make_backend(cfg.narration.backend) or make_backend("claude_cli")
    ans = narrate_answer(
        args.question, sel.nodes, backend, target_sentences=sel.target_sentences
    )
    if ans.narrative:
        print(ans.narrative)
        print()
    elif ans.fell_back:
        print("(No narrative backend available — showing the matched observations.)\n")
    print(ans.evidence)
    return 0


def _cmd_mcp(_args: argparse.Namespace) -> int:
    from memory_graph.mcp_server import main as mcp_main

    mcp_main()
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    # The labeling UI binds a LOOPBACK-only server socket to serve photos to a local
    # browser — the sanctioned local-serve surface (like the 127.0.0.1 dispatch
    # fallback). It is not outbound network, so netguard is not applied to this command.
    from memory_graph.labeling.server import serve
    from memory_graph.labeling.store import LabelStore

    if args.reset_base_rate:
        s = LabelStore("base_rate")
        if s.path.is_file():
            archived = s.path.with_suffix(".jsonl.ambiguous-bak")
            s.path.rename(archived)
            print(f"archived old base-rate labels to {archived.name}; starting fresh.")

    serve(port=args.port)
    return 0


def _cmd_backfill_context(args: argparse.Namespace) -> int:
    """Backfill lens and place metadata without opening pixels or Photos masters."""
    from memory_graph.db import connect, init_db, update_photo_fields

    conn = connect()
    init_db(conn)
    updated = 0
    records_path = args.archive_records
    if records_path:
        try:
            with open(records_path, encoding="utf-8") as handle:
                records = json.load(handle).get("records", [])
        except (OSError, json.JSONDecodeError) as exc:
            print(f"context backfill: cannot read archive records: {exc}")
            conn.close()
            return 1
        for record in records:
            rid = record.get("uuid")
            if not rid:
                continue
            photo_id = f"hdd:{rid}"
            fields = {
                "lens_model": record.get("lens_model"),
                "latitude": record.get("latitude"),
                "longitude": record.get("longitude"),
                "city": record.get("city"),
                "country": record.get("country"),
                "state": record.get("state"),
            }
            if any(value is not None for value in fields.values()):
                update_photo_fields(conn, photo_id, fields)
                updated += 1
        conn.commit()
    if not args.hdd_only:
        try:
            import osxphotos
        except ImportError:
            print("context backfill: osxphotos unavailable; HDD metadata was still applied")
        else:
            db = osxphotos.PhotosDB(dbfile=args.dbfile)
            for photo in db.photos():
                photo_id = f"lib:{photo.uuid}"
                exif = getattr(photo, "exif_info", None)
                fields = {
                    "lens_model": getattr(exif, "lens_model", None),
                    "latitude": getattr(photo, "latitude", None),
                    "longitude": getattr(photo, "longitude", None),
                    "city": getattr(photo, "city", None),
                    "country": getattr(photo, "country", None),
                    "state": getattr(photo, "state", None),
                }
                if any(value is not None for value in fields.values()):
                    update_photo_fields(conn, photo_id, fields)
                    updated += 1
            conn.commit()
    conn.close()
    print(f"context backfill: updated {updated} photo rows")
    return 0


def _cmd_doctor(_args: argparse.Namespace) -> int:
    from memory_graph.config import load_config
    from memory_graph.paths import DB_PATH, LOCAL_CONFIG

    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — doctor reports, does not crash
        print(f"config: ERROR — {exc}")
        return 1

    print(f"memory_graph {__version__}")
    print(f"config: ok (backend={cfg.narration.backend}, thumb={cfg.ingest.thumb_long_edge}px)")
    print(f"local overlay: {'present' if LOCAL_CONFIG.is_file() else 'none'}")
    print(f"database: {'exists' if DB_PATH.exists() else 'not yet created (run `init`)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memory-graph", description="Memory Graph Engine")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("init", help="create the database and data directories")
    sub.add_parser("doctor", help="report config and environment health")

    lib = sub.add_parser("ingest-library", help="probe: read the Photos library (no-master guard)")
    lib.add_argument("--limit", type=int, default=0, help="stop after N candidates (0 = all)")
    lib.add_argument("--dbfile", default=None, help="path to a specific Photos database")

    h = sub.add_parser("ingest-hdd", help="probe: read a read-only external archive")
    h.add_argument("root", help="archive root directory (must be a read-only mount)")
    h.add_argument("--limit", type=int, default=0, help="stop after N candidates (0 = all)")

    ch = sub.add_parser("cluster-hdd", help="probe: ingest + reject + cluster an HDD folder")
    ch.add_argument("root", help="archive folder (must be a read-only mount)")
    ch.add_argument("--limit", type=int, default=0, help="cap candidates before clustering")

    ah = sub.add_parser("analyze-hdd", help="full pipeline on an HDD folder, persisted to the db")
    ah.add_argument("root", help="archive folder (must be a read-only mount)")
    ah.add_argument("--limit", type=int, default=0, help="cap candidates before analysis")
    ah.add_argument("--no-vision", action="store_true", help="skip Apple Vision (faster)")

    dp = sub.add_parser("dispatch", help="show today's resonance-ranked dispatch")
    dp.add_argument("-n", type=int, default=0, help="number of photos (0 = config default)")

    tt = sub.add_parser("train-taste", help="train the taste model on your Library favorites")
    tt.add_argument("--limit", type=int, default=0, help="cap photos scanned (0 = all)")

    cr = sub.add_parser("critique", help="longitudinal style critique (insights per era)")
    cr.add_argument("--narrate", action="store_true", help="add grounded LLM narration")

    sub.add_parser("rescore", help="recompute quadrants+resonance from stored signals "
                                    "(after a crop-factor/optics fix; no re-decode)")

    ri = sub.add_parser(
        "reidentify", help="match local face derivatives and cluster recurring people"
    )
    ri.add_argument("--limit", type=int, default=0, help="cap photos processed (0 = all)")
    ri.add_argument("--skip-gallery", action="store_true",
                    help="do not scan Photos; load an existing named gallery if present")
    ri.add_argument("--cluster-only", action="store_true",
                    help="skip Vision and cluster persisted face embeddings only")
    ri.add_argument("--recognition-min-similarity", type=float, default=None,
                    help="temporary named-match cosine threshold")
    ri.add_argument("--anonymous-threshold", type=float, default=None,
                    help="temporary anonymous-cluster cosine threshold")
    ri.add_argument("--anonymous-min-cluster-size", type=int, default=None,
                    help="temporary anonymous recurrence floor")

    sub.add_parser("graph", help="generate and persist the insight graph (for `ask`)")

    for name, help_text in (
        ("holdout-build", "choose the blinded items for one holdout round (one-shot)"),
        ("holdout-label", "open the blinded labelling page for one round (resumable)"),
    ):
        hp = sub.add_parser(name, help=help_text)
        hp.add_argument("--round", type=int, default=1, choices=[1, 2, 3])
    sub.add_parser("holdout-score", help="pool all holdout rounds and apply the rule")

    bg = sub.add_parser("backfill-gps",
                        help="set has_gps on existing rows from source metadata")
    bg.add_argument("--archive-records", default=None,
                    help="prototype archive_records.json (HDD GPS); defaults to the "
                         "MEMORY_GRAPH_ARCHIVE_CACHE location if set")

    bc = sub.add_parser("backfill-context",
                        help="backfill lens and place metadata without reading photo pixels")
    bc.add_argument("--archive-records", default=None,
                    help="archive_records.json for HDD lens/place metadata")
    bc.add_argument("--dbfile", default=None, help="specific Photos database")
    bc.add_argument("--hdd-only", action="store_true", help="skip Photos Library metadata")

    ask = sub.add_parser("ask", help="ask a free-form question about your photography")
    ask.add_argument("question", help="e.g. 'how has my photography changed?'")

    sr = sub.add_parser("sample-run", help="analyze a stratified sample or the full canonical pool")
    sr.add_argument("--target", type=int, default=10000, help="sample size (default 10000)")
    sr.add_argument("--all", action="store_true",
                    help="analyze every canonical photo in the local pool")
    sr.add_argument("--dbfile", default=None,
                    help="Photos database path when the library is not the default one")
    sr.add_argument("--hdd-only", action="store_true", help="skip the Library source")
    sr.add_argument("--no-vision", action="store_true", help="skip Apple Vision (faster)")

    lbl = sub.add_parser("label", help="local labeling UI (loopback) to tune the engine")
    lbl.add_argument("--port", type=int, default=8737, help="loopback port")
    lbl.add_argument("--reset-base-rate", action="store_true",
                     help="archive old base-rate labels and re-ask with the strict question")

    sub.add_parser("mcp", help="run the MCP dispatch server over stdio")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    # Block outbound network for every command EXCEPT the two that bind a loopback
    # server socket themselves: `label` (the local labeling UI) and `mcp` (which calls
    # block_network internally before the SDK loads). Neither makes outbound connections.
    # `holdout-label` binds a 127.0.0.1 page for the browser; it makes no outbound calls.
    if args.command not in ("label", "mcp", "holdout-label"):
        block_network()

    handlers = {
        "init": _cmd_init,
        "doctor": _cmd_doctor,
        "ingest-library": _cmd_ingest_library,
        "ingest-hdd": _cmd_ingest_hdd,
        "cluster-hdd": _cmd_cluster_hdd,
        "analyze-hdd": _cmd_analyze_hdd,
        "dispatch": _cmd_dispatch,
        "train-taste": _cmd_train_taste,
        "critique": _cmd_critique,
        "rescore": _cmd_rescore,
        "reidentify": _cmd_reidentify,
        "graph": _cmd_graph,
        "backfill-gps": _cmd_backfill_gps,
        "backfill-context": _cmd_backfill_context,
        "holdout-build": _cmd_holdout_build,
        "holdout-label": _cmd_holdout_label,
        "holdout-score": _cmd_holdout_score,
        "ask": _cmd_ask,
        "sample-run": _cmd_sample_run,
        "label": _cmd_label,
        "mcp": _cmd_mcp,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
