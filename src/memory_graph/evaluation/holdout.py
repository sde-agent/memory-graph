"""Holdout evaluation of the daily memory ranking (evaluation.md Parts U and V, pre-registered).

Each round is three steps, each its own command:
  1. build:  choose 100 blinded items from the untouched pool (25 simulated days x top 3 from the
             engine's own dispatch selector, plus 25 random controls). Refuses to overwrite a round.
  2. label:  a local page (`memory-graph holdout-label --round N`). Answering No saves at once; the
             feeling is asked only after Yes.
  3. score:  pools every round that has labels. lift = P(worthy | top-3) - P(worthy | control),
             pass rule applied unchanged, quadrant agreement reported on yes-answered items only.

The pass rule lives in `decide()` and is tested directly. Do not edit it after labels exist.
"""

from __future__ import annotations

import json
import math
import random

from memory_graph.paths import DATA_DIR

# ----- frozen parameters (evaluation.md Parts U and V).
ROUND_SEEDS = {1: 20261026, 2: 20261201, 3: 20261215}
N_DAYS = 25
PER_DAY = 3
N_CONTROL = 25
PASS_LIFT = 0.25          # lift at or above this (with a positive lower bound) passes
INCONCLUSIVE_LIFT = 0.10  # lift below this fails
INFORMATIVE_LIFT = 0.10   # Part W: lift needed to call the feelings scheme informative
Z95 = 1.96

LABELS_DIR = DATA_DIR / "labels"
# Round 1 keeps its original file names; later rounds get numbered files.
SET_FILE = LABELS_DIR / "holdout_set.json"
LABEL_FILE = LABELS_DIR / "holdout.jsonl"
THUMB_DIR = LABELS_DIR / "holdout_thumbs"


def round_files(round_no: int):
    """(set file, label file) for a round. Round 1 reads the module globals at call time."""
    if round_no == 1:
        return SET_FILE, LABEL_FILE
    return (LABELS_DIR / f"holdout_set_r{round_no}.json",
            LABELS_DIR / f"holdout_r{round_no}.jsonl")


def decide(lift: float, lower: float) -> str:
    """The pre-registered pass rule. Returns PASS, INCONCLUSIVE or FAIL."""
    if lift < INCONCLUSIVE_LIFT:
        return "FAIL"
    if lift >= PASS_LIFT and lower > 0:
        return "PASS"
    return "INCONCLUSIVE"


def lift_stats(k_top: int, n_top: int, k_ctrl: int, n_ctrl: int) -> dict:
    """Difference of proportions with a normal approximation (Part U)."""
    p1 = k_top / n_top
    p2 = k_ctrl / n_ctrl
    se = math.sqrt(p1 * (1 - p1) / n_top + p2 * (1 - p2) / n_ctrl)
    lift = p1 - p2
    return {"p_top": p1, "p_control": p2, "lift": lift, "se": se,
            "lower": lift - Z95 * se, "upper": lift + Z95 * se}


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a proportion (used for the quadrant report)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + Z95 ** 2 / n
    centre = (p + Z95 ** 2 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _read_jsonl_ids(path) -> set[str]:
    out: set[str] = set()
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id"):
                out.add(str(rec["id"]))
    return out


def _excluded_ids() -> set[str]:
    """Everything already used: tuning labels, every earlier holdout set and every answer."""
    used: set[str] = set()
    label_set = LABELS_DIR / "label_set.json"
    if label_set.is_file():
        used |= set(json.loads(label_set.read_text()).get("ids", []))
    for name in ("base_rate", "quadrant", "pairwise"):
        used |= _read_jsonl_ids(LABELS_DIR / f"{name}.jsonl")
    for path in LABELS_DIR.glob("holdout_set*.json"):
        used |= {it["id"] for it in json.loads(path.read_text())["items"]}
    for path in LABELS_DIR.glob("holdout*.jsonl"):
        used |= _read_jsonl_ids(path)
    return used


def build_set(conn, cfg, *, now: float, round_no: int = 1, force: bool = False) -> dict:
    """Select the blinded items for one round. Refuses to overwrite an existing round."""
    from memory_graph.scoring.dispatch import select_dispatch

    set_file, _ = round_files(round_no)
    if set_file.is_file() and not force:
        raise FileExistsError(
            f"{set_file} already exists. A holdout round is one-shot; do not re-select after "
            "labels exist. Pass force=True only before any labels are collected.")
    seed = ROUND_SEEDS[round_no]

    used = _excluded_ids()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM photos WHERE is_canonical = 1 AND resonance IS NOT NULL "
        "AND last_dispatched IS NULL AND private = 0")]
    pool = [r for r in rows if r["id"] not in used]

    remaining = list(pool)
    items: list[dict] = []
    for day in range(N_DAYS):
        picked = select_dispatch(remaining, cfg.dispatch, now=now, n=PER_DAY)
        ids = [it.id for it in picked][:PER_DAY]
        if len(ids) < PER_DAY:   # the selector may return fewer; top up by resonance
            extra = sorted((r for r in remaining if r["id"] not in ids),
                           key=lambda r: (-(r["resonance"] or 0), r["id"]))
            ids += [r["id"] for r in extra[: PER_DAY - len(ids)]]
        for pid in ids:
            items.append({"id": pid, "kind": "dispatch", "day": day})
        remaining = [r for r in remaining if r["id"] not in ids]

    rng = random.Random(seed)
    controls = rng.sample(sorted(r["id"] for r in remaining), N_CONTROL)
    items += [{"id": pid, "kind": "control", "day": None} for pid in controls]

    # Blind the order: a fixed-seed shuffle so the labeller cannot tell top-3 from control.
    random.Random(seed).shuffle(items)
    spec = {"round": round_no, "seed": seed, "days": N_DAYS, "per_day": PER_DAY,
            "controls": N_CONTROL, "pass_lift": PASS_LIFT,
            "inconclusive_lift": INCONCLUSIVE_LIFT, "pool_size": len(pool),
            "excluded_used": len(used), "reference_time": now}
    out = {"spec": spec, "items": items}
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    set_file.write_text(json.dumps(out, indent=2))
    return out


def _load_labels(label_path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                labels[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return labels


def score_rounds(rounds: list[tuple[dict, object]], rows_by_id: dict[str, dict]) -> dict:
    """Pool every round's labels and apply the pre-registered rule (Parts U and V)."""
    k_top = n_top = k_ctrl = n_ctrl = 0
    per_round = []
    yes_pairs: list[tuple[int, int]] = []
    for round_no, (set_obj, label_path) in enumerate(rounds, start=1):
        labels = _load_labels(label_path)
        items = set_obj["items"]
        top = [it for it in items if it["kind"] == "dispatch" and it["id"] in labels]
        ctrl = [it for it in items if it["kind"] == "control" and it["id"] in labels]
        kt = sum(1 for i in top if labels[i["id"]]["worthy"])
        kc = sum(1 for i in ctrl if labels[i["id"]]["worthy"])
        k_top += kt
        n_top += len(top)
        k_ctrl += kc
        n_ctrl += len(ctrl)
        per_round.append({"round": round_no, "top": len(top), "control": len(ctrl),
                          "expected_top": sum(1 for i in items if i["kind"] == "dispatch"),
                          "expected_control": sum(1 for i in items if i["kind"] == "control"),
                          "k_top": kt, "k_control": kc})
        # Quadrant agreement is report-only and uses yes-answered items only (Part V).
        for it in items:
            rec = labels.get(it["id"])
            if rec and rec["worthy"] and rec.get("quadrant") and it["id"] in rows_by_id:
                yes_pairs.append((rows_by_id[it["id"]]["dominant_quadrant"], rec["quadrant"]))

    report: dict = {"rounds": per_round, "labelled_top": n_top, "labelled_control": n_ctrl}
    if n_top == 0 or n_ctrl == 0:
        report["decision"] = "INCOMPLETE"
        return report

    st = lift_stats(k_top, n_top, k_ctrl, n_ctrl)
    report.update(st)
    report["decision"] = decide(st["lift"], st["lower"])
    agree = sum(1 for eng, felt in yes_pairs if eng == felt)
    lo, hi = wilson(agree, len(yes_pairs)) if yes_pairs else (0.0, 0.0)
    report["quadrant"] = {"agree": agree, "n": len(yes_pairs),
                          "rate": agree / len(yes_pairs) if yes_pairs else None,
                          "wilson95": [lo, hi]}
    return report


def score(set_obj: dict, rows_by_id: dict[str, dict]) -> dict:
    """Round 1 only (kept for the original round-1 report)."""
    return score_rounds([(set_obj, LABEL_FILE)], rows_by_id)


def score_feelings(set_obj: dict, label_path, rows_by_id: dict[str, dict]) -> dict:
    """Multiple-feelings test (evaluation.md Part W). Chance-corrected, report-style rule.

    A hit is the engine's dominant quadrant being among the feelings picked. The chance rate for an
    item is the sum, over the picked feelings, of how often each quadrant is the engine's dominant
    one in this round. The lift is observed hits minus expected chance.
    """
    labels = _load_labels(label_path)
    items = [it for it in set_obj["items"]
             if it["id"] in labels and labels[it["id"]].get("quadrants")
             and it["id"] in rows_by_id]
    report: dict = {"n": len(items)}
    if not items:
        report["decision"] = "INCOMPLETE"
        return report

    dom = {it["id"]: rows_by_id[it["id"]]["dominant_quadrant"] for it in items}
    counts: dict[int, int] = {}
    for q in dom.values():
        counts[q] = counts.get(q, 0) + 1
    total = len(items)
    p = {q: counts.get(q, 0) / total for q in range(1, 6)}

    hits = 0
    expected = 0.0
    variance = 0.0
    sizes = []
    all_five = 0
    for it in items:
        picked = set(labels[it["id"]]["quadrants"])
        sizes.append(len(picked))
        all_five += len(picked) == 5
        hits += dom[it["id"]] in picked
        e = sum(p[q] for q in picked)
        expected += e
        variance += e * (1 - e)

    observed = hits / total
    chance = expected / total
    lift = observed - chance
    se = math.sqrt(variance) / total
    lower = lift - Z95 * se
    report.update({
        "observed": observed, "chance": chance, "lift": lift, "se": se,
        "lower": lower, "upper": lift + Z95 * se,
        "avg_feelings": sum(sizes) / total, "all_five_share": all_five / total,
    })
    report["padding_flag"] = report["avg_feelings"] >= 4
    report["decision"] = ("INFORMATIVE" if lift >= INFORMATIVE_LIFT and lower > 0
                          else "NOT INFORMATIVE")
    return report
