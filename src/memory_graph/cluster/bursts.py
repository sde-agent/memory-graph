"""Burst grouping + canonical keyframe election (spec.md §3, evaluation.md B.3).

Two candidates join the same burst when ALL hold:
  - they are time-adjacent: the gap to the running burst's start is <= window;
  - they share a "shot key": same source, same framing bucket, same orientation — so a
    wide shot and a tele shot of the same scene are never merged;
  - they are near-duplicate by perceptual hash: Hamming(pHash) <= max distance.
Library bursts that Apple already grouped (a shared `burst_group`) are honored directly.

Canonical election picks the ONE frame to keep as the moment's representative:
  score = sharpness (Laplacian variance, weighted inside the largest face box when
          faces are known) + an eyes-open bonus + a favorited-member preference.
Sharpness is the dominant term; ties break toward a favorited / earliest frame.
Face boxes and eyes-open land in P5 — until then election is sharpness + favorite,
with the hooks already in place so P5 slots in without reshaping this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory_graph.config import Config
from memory_graph.contracts import Candidate
from memory_graph.ingest.imaging import hamming
from memory_graph.metrics.sharpness import region_variance, sharpness

# Framing bucket edges (35mm-equiv mm). A burst must stay within one bucket.
# Mirrors config framing.bounds conceptually; recomputed from the candidate's focal.
_FRAMING_BOUNDS = (20.0, 35.0, 71.0, 135.0, 300.0)


@dataclass
class Shot:
    """A single moment: one elected representative plus the frames it collapsed."""

    burst_id: str
    representative: Candidate
    members: list[Candidate] = field(default_factory=list)

    @property
    def favorite(self) -> bool:
        return any((m.library_meta or {}).get("favorite") for m in self.members)

    @property
    def size(self) -> int:
        return len(self.members)


def _framing_bucket(focal35: float | None) -> int:
    if focal35 is None:
        return -1
    bucket = 0
    for edge in _FRAMING_BOUNDS:
        if focal35 >= edge:
            bucket += 1
    return bucket


def _shot_key(c: Candidate) -> tuple:
    """What must match for two frames to belong to the same burst."""
    focal = c.exif.get("focal35") or c.exif.get("focal")
    orient = c.exif.get("orientation")
    # Normalize orientation to portrait/landscape rather than exact EXIF code.
    w, h = c.exif.get("width"), c.exif.get("height")
    portrait = (h or 0) > (w or 0) if (w and h) else None
    return (c.source, _framing_bucket(focal), portrait, orient if portrait is None else None)


def _apple_burst(c: Candidate) -> str | None:
    return (c.library_meta or {}).get("burst_group")


def _near_duplicate(a: Candidate, b: Candidate, max_distance: int) -> bool:
    pa, pb = a.exif.get("phash"), b.exif.get("phash")
    if not pa or not pb:
        return False
    try:
        return hamming(pa, pb) <= max_distance
    except ValueError:
        return False


def _sort_key(c: Candidate) -> tuple:
    return (c.timestamp if c.timestamp is not None else float("inf"), c.id)


def group_bursts(candidates: list[Candidate], cfg: Config) -> list[list[Candidate]]:
    """Group candidates into bursts. Returns a list of member-lists (chronological)."""
    ordered = sorted(candidates, key=_sort_key)
    window = cfg.dedup.burst_window_seconds
    max_dist = cfg.dedup.phash_max_distance

    # Apple-grouped library bursts first: all members of one burst_group stay together.
    apple: dict[str, list[Candidate]] = {}
    loose: list[Candidate] = []
    for c in ordered:
        bg = _apple_burst(c)
        if bg:
            apple.setdefault(bg, []).append(c)
        else:
            loose.append(c)

    groups: list[list[Candidate]] = list(apple.values())

    # Time + shot-key + pHash grouping for the rest.
    current: list[Candidate] = []
    start_ts: float | None = None
    key = None
    for c in loose:
        ts = c.timestamp
        k = _shot_key(c)
        joins = (
            current
            and k == key
            and ts is not None
            and start_ts is not None
            and (ts - start_ts) <= window
            and _near_duplicate(current[-1], c, max_dist)
        )
        if joins:
            current.append(c)
        else:
            if current:
                groups.append(current)
            current = [c]
            start_ts = ts
            key = k
    if current:
        groups.append(current)
    return groups


def _largest_face_box(c: Candidate) -> tuple[float, float, float, float] | None:
    """Largest face bounding box (normalized x,y,w,h) from library_meta, or None.

    P5 will populate faces for both sources; today only library face centers/sizes may
    be present. Returns None when no usable box exists (election falls back to whole-frame).
    """
    faces = (c.library_meta or {}).get("faces") or []
    best = None
    for f in faces:
        size = f.get("size")
        cx, cy = f.get("center_x"), f.get("center_y")
        if size and cx is not None and cy is not None:
            # osxphotos face center/size are fractions of the image; build a square box.
            x = max(0.0, cx - size / 2)
            y = max(0.0, cy - size / 2)
            if best is None or size > best[0]:
                best = (size, (x, y, size, size))
    return best[1] if best else None


def _eyes_open_bonus(c: Candidate) -> float:
    """Eyes-open bonus for election. Hook for P5 face landmarks; neutral (0) until then."""
    return 0.0


def election_score(c: Candidate, cfg: Config) -> float:
    """Score a candidate as a burst representative. Higher wins.

    Sharpness dominates (weighted inside the largest face box if known); a favorited
    member and an eyes-open bonus act as tie-breakers / nudges.
    """
    if c.thumb is None:
        # No pixels to measure (thumbnail already released): fall back to a cached value.
        base = float(c.exif.get("sharpness") or 0.0)
    else:
        box = _largest_face_box(c)
        base = region_variance(c.thumb, box) if box else sharpness(c.thumb)
    fav = 1.0 if (c.library_meta or {}).get("favorite") else 0.0
    # Favorite nudge is proportional to the field so it breaks near-ties without
    # letting a favorited-but-blurry frame beat a clearly sharper one.
    return base * (1.0 + 0.05 * fav) + _eyes_open_bonus(c)


def elect_canonical(members: list[Candidate], cfg: Config) -> Candidate:
    """Pick the representative of a burst by election score; earliest breaks ties."""
    best = max(
        members,
        key=lambda c: (election_score(c, cfg), -(_sort_key(c)[0] if c.timestamp else 0)),
    )
    return best


def _burst_id(members: list[Candidate]) -> str:
    """A stable burst id: the smallest member id (deterministic, order-independent)."""
    return "burst:" + min(m.id for m in members)


def cluster_and_elect(candidates: list[Candidate], cfg: Config) -> list[Shot]:
    """Group candidates into bursts and elect a canonical representative for each."""
    shots: list[Shot] = []
    for members in group_bursts(candidates, cfg):
        rep = elect_canonical(members, cfg)
        shots.append(Shot(burst_id=_burst_id(members), representative=rep, members=members))
    return shots


def global_dedup(shots: list[Shot], cfg: Config) -> list[Shot]:
    """Collapse near-duplicate SHOTS across the whole set, regardless of time/shot-key.

    Burst clustering only merges time-adjacent same-moment frames, so near-identical
    photos taken minutes apart — or the SAME photo present on both the HDD and in the
    Library — survive as separate shots (finding: 56 dupe pairs leaked into a 276-photo
    labeled set). This second pass groups shot representatives by perceptual-hash
    similarity across everything and keeps the best of each duplicate group.

    To avoid an O(n^2) scan on 100k+ shots, representatives are bucketed by a pHash
    prefix (a cheap locality key); only within-bucket pairs are compared by Hamming
    distance. Members of merged shots are folded together so nothing is lost.
    """
    max_dist = cfg.dedup.phash_max_distance
    hashed = [s for s in shots if s.representative.exif.get("phash")]
    no_hash = [s for s in shots if not s.representative.exif.get("phash")]

    # LSH-style banding: to find pairs within Hamming <= max_dist without an O(n^2) scan,
    # split each hash into B bands. Two hashes within max_dist differ in at most max_dist
    # bits, so by the pigeonhole principle they share at least one band when B > max_dist.
    # Candidate pairs are those colliding in any band; we then confirm with full Hamming.
    # (A single short prefix missed dupes differing in early bits — the finding.)
    phash_len = len(hashed[0].representative.exif["phash"]) if hashed else 0
    bands = max(max_dist + 1, 4)
    band_size = max(1, phash_len // bands)

    parent = list(range(len(hashed)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    phashes = [s.representative.exif["phash"] for s in hashed]

    # Union-find directly WITHIN each band bucket — never materialize the full pair set
    # (a large bucket is O(bucket^2) pairs; on 100k+ items that exploded). Within a
    # bucket we only Hamming-check pairs not already in the same component, so once a
    # cluster forms, further members join cheaply.
    for b in range(bands):
        lo = b * band_size
        hi = lo + band_size if b < bands - 1 else phash_len
        buckets: dict[str, list[int]] = {}
        for idx in range(len(hashed)):
            buckets.setdefault(phashes[idx][lo:hi], []).append(idx)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for a in range(len(members)):
                ia = members[a]
                ra = find(ia)
                for c in range(a + 1, len(members)):
                    ic = members[c]
                    if find(ic) == ra:
                        continue   # already same component; skip the Hamming check
                    if _safe_hamming(phashes[ia], phashes[ic]) <= max_dist:
                        union(ia, ic)
                        ra = find(ia)

    groups: dict[int, list[Shot]] = {}
    for idx, s in enumerate(hashed):
        groups.setdefault(find(idx), []).append(s)

    merged: list[Shot] = [
        _merge_shots(g, cfg) if len(g) > 1 else g[0] for g in groups.values()
    ]
    merged.extend(no_hash)
    return merged


def _safe_hamming(a: str, b: str) -> int:
    try:
        return hamming(a, b)
    except ValueError:
        return 64


def _merge_shots(cluster: list[Shot], cfg: Config) -> Shot:
    """Fold several near-duplicate shots into one, re-electing across all their members."""
    all_members: list[Candidate] = []
    for s in cluster:
        all_members.extend(s.members)
    rep = elect_canonical(all_members, cfg)
    return Shot(burst_id=_burst_id(all_members), representative=rep, members=all_members)
