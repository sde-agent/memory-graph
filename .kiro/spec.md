# Memory Graph Engine — Technical Specification (Living Contract)

This spec is the SOURCE OF TRUTH for implementation. It supersedes the original
module-contract draft and reflects the design decided with the owner. The sibling
`../photo-themes` prototype is a PARTS BIN we copy/adapt from; it is not authoritative.

Companion docs:
- `steering/intent.md` — the emotional vision (why).
- `steering/product.md` — product decisions, guardrails, roadmap.
- `evaluation.md` — how every phase and the end product are measured and guarded.

Language/runtime: Python 3.11+. Package root: `memory_graph/`. Config-driven (no
personal data in code). Apple on-device frameworks (Vision, Foundation Models) are
preferred over PyTorch/CLIP wherever a V3/iOS-portable choice exists; exceptions (the
V1 CLIP taste model) are noted.

Reuse legend per module:  [COPY] adapt from photo-themes  ·  [NEW] build fresh  ·
[REF] reference-only, reshape to our contract.

---

## 0. Core Data Model

### `Candidate` (in-memory, flows through the pipeline)
```
id: str                      # stable, source-prefixed ("lib:<uuid>" / "hdd:<sha1>")
source: str                  # "library" | "hdd"
original_ref: str            # uuid (library) or path (hdd); NEVER a master path to open
timestamp: float             # capture time (epoch); None if unknown
thumb: PIL.Image             # in-memory RGB, long edge ~512-768px (ephemeral)
exif: dict                   # normalized optics (see §4)
library_meta: dict | None    # favorite, persons[], face attrs, albums, keywords (library only)
```
Thumbnails are ephemeral: decoded once, used, released. No full-resolution photo is
ever written to internal storage.

### Persisted record
See §7 for the SQLite schema. We persist the full soft quadrant vector and all
resonance inputs, not a single class integer.

---

## 1. Ingest Adapters (`memory_graph/ingest/`)

### `hdd.py`  [COPY from archive.py + exif.py + previews.py]
- `stream_hdd(root: str) -> Iterator[Candidate]`
- Sequential `os.scandir` traversal (generators; not `os.walk`).
- Extension filter (config): `.jpg .jpeg .heic .cr2 .nef .arw .tiff .dng`.
- Read-only enforced: assert source mount is read-only; never write to the tree.
- Single-pass: open each file ONCE — read EXIF (cached, resumable, exiftool) and
  downsample to the in-memory thumbnail together. RAW uses the embedded JPEG where
  present, else a decode fallback.
- Stable id = source-prefixed content hash so re-runs upsert rather than duplicate.
- Corrupt/unreadable files: skip and COUNT (never fatal).

### `library.py`  [COPY from library.py — WITH THE CRITICAL INVERSION]
- `stream_library() -> Iterator[Candidate]`
- `osxphotos.PhotosDB()`; read-only.
- MUST prefer `photo.path_derivatives` and MUST NOT access `photo.path` / masters.
  (The prototype reads `photo.path` first — we invert this. Enforced by test +
  runtime assertion per evaluation.md B.1.)
- Items with empty derivatives (iCloud-only, not downloaded): skip and COUNT.
- `library_meta` carries: favorite flag, named persons + face attributes (bbox,
  quality, gaze/smile where available), albums, keywords, screenshot flag,
  35mm-equiv focal (via read-only SQL on osxphotos' own connection when EXIF lacks it).

Both adapters yield the uniform `Candidate`. No memory-ceiling contract (dropped).

---

## 2. Rejection (`memory_graph/filters/rejection.py`)  [COPY from content_flags.py]

- `classify_clutter(c: Candidate) -> ClutterFlags`  → subset of
  `{screenshot, document, flat_nothing}` (empty set = keep).
- Signals: Apple Vision scene classification (document/whiteboard/receipt/text ids
  ≥ threshold), screenshot = PNG or Photos screenshot flag or exact device-screen
  dimensions or Vision `screenshot` score; flat = very low contrast + saturation.
- **Asymmetric loss:** a false reject of a cherished photo is far worse than a false
  keep. Tuned for high recall of real photos (evaluation.md B.2).
- **Never auto-deletes.** Flagged items are quarantined for owner review on first run
  and merely removed from the resonance candidate pool, not from disk.

---

## 3. Burst Clustering & Canonical Election (`memory_graph/cluster/bursts.py`)
[COPY burst grouping from dedup.py; NEW keyframe election]

- `cluster_and_elect(candidates: list[Candidate]) -> list[Shot]`
- Group into a burst when time-adjacent (config `burst_window_seconds`) AND same
  framing/camera "shot key" AND near-duplicate (pHash distance ≤ config). Uses Apple
  `burst_group` where present (library), pHash+time otherwise (hdd).
- **Canonical election [NEW]:** elect the keyframe by sharpness (Laplacian variance),
  weighted inside the largest face box when faces exist, with an eyes-open bonus from
  face landmarks and a favorited-member preference. (Prototype picks earliest/favorited
  only — insufficient.)
- Output `Shot{ representative: Candidate, members: list[Candidate], burst_id: str }`.
  Only representatives flow downstream; members are retained for grouping/audit.

---

## 4. Metrics (`memory_graph/metrics/`)

### `optics.py`  [COPY/ADAPT exif.py — ADD aperture/shutter/ISO tags]
- `extract_optics(exif: dict) -> Optics`
- 35mm-equiv focal (EXIF `FocalLengthIn35mmFilm` or crop-factor from config body table),
  aperture (F-number), shutter (seconds), ISO, derived EV. (Prototype extracts focal
  only — we add `-FNumber -ExposureTime -ISO`.)

### `geometry.py`  [NEW + COPY saliency from vision.py/composition.py]
- `horizon(thumb) -> (angle_deg, horizon_y_ratio)` — Canny + HoughLinesP, median of
  near-horizontal segments; Dutch-angle flag from |angle|. [NEW]
- `negative_space(thumb) -> ratio` — derived from Apple Vision saliency box area
  (1 − salient-area fraction). [COPY saliency, NEW ratio]
- `composition(saliency_boxes) -> {centered, thirds, off_center}` [COPY].

### `light.py`  [COPY — directly reusable]
- `solar(lat, lon, utc) -> band` — NOAA solar elevation → night/blue/golden/daylight/
  high-sun. Golden-hour is the Q1/warmth signal.
- `color_temp(thumb) -> {warm_cool, saturation, contrast}` — warm-cool = meanR−meanB.

### `faces.py`  [NEW — the biggest new build]
- `detect_faces(thumb) -> list[Face]` — Apple Vision `VNDetectFaceRectangles` +
  landmarks: bbox, count, `face_ratio` (largest bbox area / frame), gaze (frontal?),
  smile estimate, eyes-open.
- `candidness(faces) -> score` — multi-face + frontal gaze + high smile → "posed"
  (down-weight); unposed/off-gaze → "candid" (elevate). Rough heuristic for v1.
- `recognize(faces, gallery) -> list[NamedFace]` — embed detected faces; nearest-
  neighbor match against the gallery (§5). Must clear a pre-registered precision bar
  before asserting a name (evaluation.md B.4); below bar → "a person is present" only.

### `router.py`  [NEW — soft scores over ranges, NOT hard thresholds]
- `score_quadrants(optics, geometry, light, faces) -> QuadrantVector` → `[q1..q5]`
  floats in [0,1] via smooth ramps over config ranges. No winner-take-all cliff.
- `dominant_quadrant = argmax(vector)` — convenience only; the FULL vector persists.
- Range intent (authoritative source = intent.md matrix; the old draft's crisp
  numbers are discarded as cliffs):
  - Q1 Nostalgic: normal focal (~35–50mm), wide aperture, eye-level, golden/warm.
  - Q2 Visceral: ultra-wide, low horizon, fast shutter — "warm-Q2" when warm-lit.
  - Q3 Empathetic: tele focal, high `face_ratio`, candid — the top resonance quadrant.
  - Q4 Solitary: large negative space OR long exposure.
  - Q5 Provocative: Dutch tilt OR shutter-drag range OR frame occlusion.

---

## 5. Identity Gallery (`memory_graph/identity/gallery.py`)  [NEW]

- `build_gallery() -> Gallery` — from the Library's already-NAMED faces, extract face
  embeddings and store per-person reference vectors (zero training; the Library is the
  labeled set). Rationale: the cast of people is small and stable over ~20 years.
- `match(face_embedding, gallery) -> (name|None, confidence)` — cosine NN; applied to
  BOTH library and hdd faces so named identity works everywhere.
- `bridge_by_phash(hdd_candidate, library_index)` — if an HDD photo is a pHash
  near-duplicate of a Library photo, propagate that Library photo's person tags
  directly (cheap, exact).
- Guardrail: identity never leaves the device except within a chosen dispatch; a match
  below the precision bar may not assert a name in narration.

---

## 6. Scoring & Dispatch (`memory_graph/scoring/`)

### `resonance.py`  [NEW]  — see evaluation.md Part A for the full contract
```
resonance = base_quadrant_affinity      # Σ w[q]*q_score[q] with per-term modifiers:
                                        #   age_boost applied INSIDE the Q1 term only,
                                        #   warm_gate applied INSIDE the Q2 term only
          * candid_multiplier
          * favorite_multiplier          # a signal, never a dominator (DR-3 floor)
          * known_person_boost           # FLAT in V1; per-person config in V1.5
          * taste_model_multiplier       # V1.5
```
- `age_boost` is NOT a global multiplier — it amplifies only nostalgia (Q1), so an old
  clinical ultra-wide gets no age bonus. `warm_gate` makes Q2 count as "warm aliveness"
  only when warm-lit, not raw awe.
- Every multiplier defaults to 1.0 when its signal is unavailable (graceful degrade;
  a photo is never penalized for an uncomputable signal).
- Used in TWO ways: (1) ranking the daily dispatch, and (2) as a first-class input to
  the LLM critique/insights. Dispatch is where resonance is EVALUATED, not the only
  place it is CONSUMED. Weights/magnitudes are config. (See evaluation.md Part A.)

### `taste.py` / `embeddings.py`  [COPY from taste.py/taste_v2.py/embeddings.py]
- V1.5: per-user logistic regression on CLIP image embeddings, trained on the owner's
  Favorite flag (later direct thumbs). Episode-split CV + blind-eval. Output is a
  resonance INPUT, not a label oracle. (V3 note: CLIP → Apple Vision feature prints.)

### `dispatch.py`  [NEW]
- `daily_dispatch(date, n=1..3) -> list[DispatchItem]` — top-N by resonance with
  repeat-avoidance, diversity floor, seeded/reproducible selection (evaluation.md A.4).
- Each `DispatchItem` carries the thumbnail, template-fact narration, and provenance
  for audit.

### `critique.py`  [COPY pattern from era.py/era_analyst.py — reshaped]
- Longitudinal style answers grouped by life eras (config-defined). Reports the
  UNWEIGHTED true distribution (your actual shooting history), AND layers resonance on
  top as a distinct lens — the gap between what you shot and what you resonate with is
  a first-class insight. Resonance enriches the critique; it never silently filters the
  true distribution. Hybrid narration per §8; all resonance values the LLM cites are
  code-recomputed from the DB (grounding, §8 / evaluation.md Part C).

---

## 7. Persistence (`memory_graph/db.py`)  [NEW schema]

SQLite, target < 100 MB for ~100k canonical photos. Stores the full quadrant vector
and all resonance inputs.

```sql
CREATE TABLE IF NOT EXISTS photos (
    id                 TEXT PRIMARY KEY,
    source             TEXT NOT NULL,          -- 'library' | 'hdd'
    original_ref       TEXT NOT NULL,          -- uuid or path; never an openable master
    timestamp          REAL,
    phash              TEXT NOT NULL,
    sharpness          REAL NOT NULL,
    is_canonical       INTEGER NOT NULL,
    burst_id           TEXT,
    -- optics
    camera_make        TEXT,
    camera_model       TEXT,
    focal_length_35mm  REAL,
    aperture           REAL,
    shutter_speed      REAL,
    iso                INTEGER,
    ev                 REAL,
    -- geometry & light
    horizon_tilt_deg   REAL,
    horizon_y_ratio    REAL,
    negative_space     REAL,
    golden_hour        INTEGER,                -- 0/1 derived from solar band
    warm_cool          REAL,
    -- faces & identity
    face_count         INTEGER NOT NULL DEFAULT 0,
    face_ratio         REAL,
    candid_score       REAL,
    known_person       INTEGER NOT NULL DEFAULT 0,  -- any match above precision bar
    person_names       TEXT,                   -- JSON array, above-bar matches only
    -- quadrant soft vector + convenience argmax
    q1_score           REAL NOT NULL DEFAULT 0,
    q2_score           REAL NOT NULL DEFAULT 0,
    q3_score           REAL NOT NULL DEFAULT 0,
    q4_score           REAL NOT NULL DEFAULT 0,
    q5_score           REAL NOT NULL DEFAULT 0,
    dominant_quadrant  INTEGER,
    -- resonance inputs / cache
    favorite           INTEGER NOT NULL DEFAULT 0,
    taste_score        REAL,
    resonance          REAL,                   -- last computed; recomputable from above
    era                TEXT,                   -- life-era label (config-derived)
    -- dispatch bookkeeping
    last_dispatched    REAL,
    private            INTEGER NOT NULL DEFAULT 0   -- never sent off-device if 1
);
CREATE INDEX IF NOT EXISTS idx_canonical_ts  ON photos(is_canonical, timestamp);
CREATE INDEX IF NOT EXISTS idx_burst         ON photos(burst_id);
CREATE INDEX IF NOT EXISTS idx_resonance     ON photos(is_canonical, resonance);
CREATE INDEX IF NOT EXISTS idx_era           ON photos(era);

-- The insight graph: engine-authored structured claims. Photos are evidence.
CREATE TABLE IF NOT EXISTS insights (
    id                TEXT PRIMARY KEY,
    kind              TEXT NOT NULL,          -- e.g. 'era_style', 'resonance_gap'
    claim             TEXT NOT NULL,          -- plain machine statement
    direction         TEXT,                   -- committed by code (e.g. 'higher')
    magnitude         REAL,                   -- committed by code
    evidence_ids      TEXT NOT NULL,          -- JSON array of photo ids
    resonance_angle   TEXT,                   -- how resonance figures in
    allowed_entities  TEXT,                   -- JSON: names (above bar) + places permitted
    era               TEXT,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_insight_era  ON insights(era);
```

---

## 8. Insight Graph & Narration (`memory_graph/narration/`)  — see evaluation.md Part C

**Architecture:** the memory graph is an INSIGHT graph. Insights are engine-authored,
structured records derived by code from deterministic metrics + resonance. Photos are
EVIDENCE hanging off insights. The LLM narrates an insight — free to interpret, fenced
by the insight's committed claim — it does NOT author insights.

### `insights.py`  [NEW]  — the insight record + builder
- `Insight{ claim, direction, magnitude, evidence_ids, resonance_angle,
  allowed_entities, era }` committed to the DB (§7). Direction/magnitude are fixed by
  CODE before any LLM runs, so narration cannot contradict them. `allowed_entities` is
  the exact permitted set of names (above the recognition precision bar) and places.
- Insights are the frozen-eval unit (evaluation.md D.1).

### `templates.py`  [NEW]  — deterministic, metric-driven facts ("50mm, f/1.8, eye-
level, golden hour, 12 years ago"). Exact-match unit-tested. The grounding floor and
the fall-back when the LLM cannot be trusted for an insight.

### `llm.py`  [COPY the arm's-length runner pattern from the prototype's
`insights.py::ClaudeRunner` — NOTE: that logic lives in OUR `llm.py`, not our
`insights.py` (ours builds insight records; the LLM runner is separate)]
- Input: ONE insight + its template facts + evidence photo(s). Nothing else.
- Primary: Apple Foundation Models via an arm's-length helper process (portable to
  V2/V3). Optional author-only upgrade: headless `claude -p`. BOTH held to the SAME
  groundedness checks: prose may not emit a setting/name/place/date outside the insight
  and may not contradict its committed direction/magnitude; interpretation is allowed
  but labeled and must be consistent with the insight.
- **Failure is honest, never silent:** verify → fail → regenerate once → fail again →
  TELL the user it could not generate anything meaningful and fall back to the template.
- **Privacy (fails closed):** every string scanned before send; anything in the Photos
  **Hidden** album is off-limit (V1); per-person off-limit marking is V1.5; undetermined
  privacy = private.

---

## 9. MCP Dispatch Surface (`memory_graph/mcp_server.py`)  [COPY from mcp_server.py]

- stdio MCP server; network blocked before the SDK is imported; strictly read-only
  over the engine DB. Serves no photo masters.
- Tool `daily_dispatch(n=1..3)` → 1–3 resonance-ranked photos as template-fact
  narration TEXT + a `http://127.0.0.1` view link each. NEVER image bytes (ZERO EGRESS:
  showing a photo to a hosted LLM sends it; the link renders the photo locally in the
  owner's browser). Owner-`private` photos are excluded and the viewer refuses them.
  Default-deny on undetermined privacy.
- Tools `style_over_time` / `what_changed(era)` → the unweighted longitudinal critique.
- Staleness: report `stale:true` if the DB predates the last ingest.

---

## 10. Cross-Cutting (`memory_graph/guards/`)  [COPY netguard/pseudonyms/publish_check/
home/cache]

- `netguard` (block outbound by default; sockets off in tests), `cache` (per-photo
  keyed, resumable, integrity guard halts a full run on any mid-run recompute),
  `pseudonyms` + `publish_check` + `home` (V2 privacy: no personal data in the repo or
  git history; derived home, pseudonymized places, fail-closed scan).

---

## Build Phasing (evaluation built alongside each phase — see evaluation.md)

- **P0** Harness & contracts: `Candidate`, a minimal `Insight` dataclass + the
  `insights` table (defined early so P7/P8/P10 share ONE definition, not a retrofit),
  config loader, DB, test/guard scaffolding.
- **P1** Ingest (hdd + library, with the master-path guard).
- **P2** Rejection.
- **P3** Burst + canonical election.
- **P4** Metrics: optics, geometry, light.
- **P5** Face stack + identity gallery.
- **P6** Quadrant soft scorer.
- **P7** Resonance + dispatch + MCP surface.   ← first user-visible payoff
- **P8** Narration (templates + grounded LLM).
- **P9** Taste model (V1.5).
- **P10** Longitudinal critique + eras.

Nothing is implemented yet. This is the contract the build must satisfy.
