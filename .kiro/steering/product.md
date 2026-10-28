# Product Intent: Memory Graph Engine (Local-First Photography Intelligence)

## Objective
Build a Python 3.11+ system to ingest, filter, cluster, and semantically critique
large photo archives across an external HDD and the macOS Photos Library. The engine
maps photographic technical execution into a 5-Quadrant Emotional Matrix, then ranks
by personal emotional resonance to power daily evocative memory dispatches and
longitudinal stylistic critique ("What was my style in the 2010s?"). It is an
emotionally intelligent companion for a lifetime of photographs — not an asset
manager, browser, or search tool.

## Hardware & Storage Target (Apple Silicon M-Series, 32GB RAM)
- **Local NVMe Headroom:** ~128 GB free space.
  - Zero disk replication: NEVER write raw full-resolution photos to the internal drive.
  - Ephemeral thumbnails only: decode once to an in-memory thumbnail (long edge
    ~512–768px), extract attributes/pHash/metrics, and release it. (There is no fixed
    RAM ceiling — the earlier 1.5 GB cap was dropped as a made-up constraint.)
  - Max SQLite footprint for ~100k canonical photos: < 100 MB.
- **Photos Library ("Optimize Mac Storage"):**
  - Zero cloud egress: NEVER call `photo.path` or open master files (triggers blocking iCloud downloads).
  - Use `osxphotos` exclusively to query metadata from `photos.db` and read pre-rendered local derivative caches (`photo.path_derivatives`).
- **External Spinning HDD (High Latency, ~120 MB/s Sequential):**
  - Single-threaded sequential traversal using `os.scandir` generators.
  - Single-pass read: read EXIF and downsample in memory once. Never re-open files on the external drive.

## The 5-Quadrant Emotional & Technical Matrix
- **Quadrant 1 (Nostalgic & Reflective):** 35mm–50mm normal lens, wide aperture (f/1.4–f/2.8), eye-level, golden hour. -> *Nostalgia, Comfort, Domestic Preservation.*
- **Quadrant 2 (Visceral & High-Arousal):** 14mm–24mm ultra-wide, low horizon / worm's eye, fast shutter (>1/1000s). -> *Awe, Wonder, Insignificance.*
- **Quadrant 3 (Empathetic & Connected):** 85mm–135mm telephoto, tight face crop (>20%), candid unposed shutter (>1/250s). -> *Tenderness, Intimacy, Belonging.*
- **Quadrant 4 (Solitary & Introspective):** Standard/Telephoto with vast negative space (>60%) OR long exposure (>2s smooth water/clouds). -> *Serenity, Tranquility, Solitude.*
- **Quadrant 5 (Provocative & Dynamic):** Dutch angle horizon tilt (>3°), intentional shutter drag (1/10s–1/2s), frame occlusions. -> *Tension, Mystery, Disorientation.*

## Face & Identity (v1 Decision)
People are a top resonance signal (a known face, people together). The cast of
relevant people is small and stable over ~20 years, so the macOS Photos Library's
existing named-face data serves as a ready-made, zero-training identity gallery.

- **Face detection:** Runs on every candidate photo (both sources) to produce face
  count, bounding boxes, and `face_ratio` (largest-face area / frame area).
- **Named recognition (Tier 2, v1 scope):** Face embeddings from the Library's
  already-named faces form a reference gallery. Faces found on HDD photos are
  embedded and matched by nearest-neighbor against that gallery, so named-person
  identity works across BOTH the HDD and the Library.
- **Candidness proxy:** Multiple faces + frontal gaze + high smile confidence ->
  "likely posed" (down-weighted). Unposed gaze / candid -> elevated. Library face
  attributes feed this cheaply; HDD photos use detector-derived gaze where available.
- **Identity bridging:** Where an HDD photo is a near-duplicate (pHash) of a Library
  photo, Library person tags propagate directly.

## Quadrant Scoring Model (Ranges, Not Hard Thresholds)
Quadrant membership is a soft score vector `[q1..q5]` computed over RANGES with
smooth ramps, not winner-take-all integer thresholds. This removes boundary cliffs,
resolves overlap/precedence, and produces the score vector the resonance ranker needs.
`dominant_quadrant` is the argmax of this vector, retained for convenience.

## Resonance Ranking (Two Consumers)
Resonance is a per-photo signal used in TWO ways: (1) it ranks the daily dispatch,
and (2) it is a first-class input to the LLM critique/insights. Dispatch is where
resonance is EVALUATED (a clean yes/no), not the only place it is CONSUMED.

The longitudinal critique STILL reports the true, UNWEIGHTED distribution (actual
shooting history) AND layers resonance on top as a distinct lens — the gap between
what was shot and what resonates is itself a first-class insight. Resonance enriches
the critique; it never silently filters the true distribution.

Personal quadrant priority (weights, tunable): Q3 (tenderness/belonging) highest,
then warm-Q2 (aliveness/energy, gated by warm color-temp / golden light), then Q4
(calm/awe), then Q1 (nostalgia), with Q5 (tension) lowest for dispatch.

Resonance model (full contract in `evaluation.md` Part A):
`resonance = base_quadrant_affinity * candid * favorite * known_person * taste`, where
`age_boost` amplifies ONLY the Q1 term (not every old photo) and `warm_gate` applies
ONLY to the Q2 term. Every multiplier defaults to 1.0 when its signal is unavailable.
`known_person` is a FLAT boost in V1 (per-person weights are V1.5). Favorites and age
are signals, never dominators (a diversity floor enforces this).

## The Memory Graph is an INSIGHT Graph
The valuable node is not a photo with metrics — it is an **Insight**: an
engine-authored, structured claim derived by CODE from deterministic metrics +
resonance (claim, direction, magnitude, evidence photo ids, resonance angle, allowed
entities, era). Photos are EVIDENCE hanging off insights. Direction and magnitude are
committed by code BEFORE any LLM runs, so narration can never contradict them. The LLM
narrates the graph; it does not author it. (Full contract in `evaluation.md` Part C.)

## Narration Engine (Hybrid, Insight-Grounded)
- **Template layer:** deterministic, metric-driven facts ("50mm, f/1.8, eye-level,
  12 years ago"). Always on; the layer we EVALUATE to exact values and the floor we
  fall back to.
- **LLM layer:** a LOCAL model (zero cloud egress) narrates ONE committed Insight —
  free to interpret the human meaning, fenced by the insight's claim. It may not emit
  a setting/name/place/date outside the insight, nor contradict its committed
  direction/magnitude; code verifies every output. Interpretation is allowed but
  labeled as a guess. Runs as a separate on-demand phase, out of the ingest path.
- **Failure is honest, never silent:** on a double verification failure the system
  TELLS the user it could not generate anything meaningful and falls back to the plain
  template — boring-but-true-and-labeled, never fluent-but-wrong.
- **Privacy (fails closed):** anything in the Photos Hidden album is off-limit in V1
  (never narrated, never dispatched off-device); per-person off-limit marking is V1.5.
## Dispatch Surface (v1 Decision) — ZERO EGRESS, photos never leave the Mac
The daily dispatch is delivered through a local **MCP server** (stdio). Claude or
ChatGPT connects to it and becomes the delivery UI. A `daily_dispatch` tool returns,
for each of the 1–3 resonance-ranked photos, the deterministic template-fact narration
as TEXT plus a `http://127.0.0.1` view link; the connected LLM adds depth on the text.

- **HARD RULE (supersedes the earlier "conscious trade"):** photo PIXELS never leave
  this Mac. Showing a photo to a hosted LLM IS sending it to that service, so the MCP
  server NEVER returns image bytes — only text and a loopback link. The owner clicks
  the link and the photo renders in their own browser (served by the local viewer,
  from the local derivative, never a master); the pixels transit only localhost. The
  connected LLM only ever sees text. Enforced by code and a regression test
  (`test_mcp_egress.py`) that fails if any image content re-appears.
- The MCP server is strictly read-only over the engine's own database and never
  touches photo masters or the Photos Library bundle. Private (Hidden-album) photos are
  excluded from dispatch and the viewer refuses to serve them.

## Roadmap & Architectural North Stars
The engine is built in versions. Decisions made NOW must not foreclose later ones.

- **V1 (current):** Works on the author's Mac. Dual-source ingest, faces, 5-quadrant
  soft scoring, resonance ranking, daily dispatch via MCP, hybrid narration.
- **V1.5 — Personal "memory model":** A per-user model trained on the user's own
  Favorite flag (and later, direct thumbs) that predicts "would I want to see this
  again." Each user trains their own; the model is a resonance INPUT, not a label
  oracle. CRITICAL LESSON (inherited): set the evaluation success bar *after*
  measuring the base rate, never before.
- **V1.5 — Feeling labels that learn from use (NOT hand-tuned).** The V1 feeling
  model is a hand-tuned rule set in `metrics/router.py`: EXIF + scene labels ->
  a 5-feeling score vector, argmax = dominant quadrant. The pre-registered holdout
  (evaluation.md Parts U/V/W) measured this against the owner's own felt labels and
  found it matches at CHANCE (round 3: engine's top feeling in the owner's picked
  set 29% vs 26% chance, lift +3, not informative; rounds 1-2 single-choice 17% vs
  24% chance, below chance). The owner felt Nostalgic/Close on almost everything he'd
  want back and almost never Quiet, while the engine keeps calling things Quiet/
  Solitary. This is a VOCABULARY + CALIBRATION mismatch, not a scoring bug, and it did
  NOT affect the yes/no dispatch result (which passed the control cleanly, +21 pts).
  DO NOT hand-tune the ramps to fit ~170 one-rater labels — that would fit noise and
  fail the next round (the exact trap the holdout exists to catch). The right fix is a
  CONTINUOUS feedback loop: collect the owner's reactions to dispatched photos over
  months (a simple thumbs / "this made me feel ___" on each daily memory), then learn
  the feeling mapping from that accumulated signal. "Tune as we go" = learn from real
  use, not a one-time fit. Any mapping change needs a fresh pre-registered round before
  any claim. The five feeling names stay provisional until this loop exists.
- **V2 — Downloadable by anyone (public GitHub), runs on any Mac supporting Apple
  Foundation Models.** Implications enforced from day one:
  - No personal data in the repo, ever (homes, eras, people, keywords, archive
    paths live only in gitignored config/data; a publish-check scans tracked files
    and git history and fails closed on any leak).
  - Everything a new user must change is CONFIG, not code (eras, cameras,
    people-of-interest, archive roots, home position derived — never typed).
  - A first-run setup flow: mark cameras, identify YOURSELF among the named faces
    (`[owner] self_names`), review flagged clutter before any analysis. Identifying the
    owner is required: photos the owner appears in were taken by someone else, so without
    it the engine mis-attributes others' framing of the owner TO the owner (a real bug
    found in V1 — a "signature" for the owner described how others shot him). With it, the
    owner is never a subject-they-chose, pairings read as "you appear alongside X", and a
    separate insight surfaces being photographed by loved ones ("seen rather than seeing").
  - **Camera focal-length backfill (REQUIRED before first analysis).** Many cameras —
    all Canon DSLRs/compacts, most small-sensor point-and-shoots, phones — omit the
    35mm-equivalent focal from EXIF and record only the raw physical/lens focal. Left
    uncorrected, every such frame is misclassified as ultra-wide, which corrupts the
    framing insight AND skews the Q1/Q2/Q3 quadrant vector (ultra-wide biases Q2
    visceral). V1 hit this on the author's own archive (Kodak DC5000 at 6.3mm, Canon
    Rebel XTi lens focals, several PowerShots). The public tool MUST:
      1. Scan the user's whole library (Photos Library INCLUDED — read focal/focal35/
         model from `photos.db` via osxphotos, never opening a master) and the HDD,
      2. list every camera model that lacks a 35mm-equivalent focal,
      3. propose a crop factor per model (from a shipped lookup of common bodies AND a
         derivation from the model's own raw-focal range vs its documented equivalent),
      4. let the user confirm/edit, write the factors to their gitignored
         `config.local.toml [framing.crop]`,
      5. then (re-)analyze with correct optics. Full-frame bodies must map to 1.0 so
         they are never wrongly scaled.
    This backfill is what makes older-era insights (the 2000s point-and-shoot years)
    trustworthy rather than silently wrong. See the re-scorable-metrics investment
    below (persisted scene_labels/horizon_dutch) so this correction is a cheap re-score,
    not a full multi-thousand-photo Vision re-decode, every time.
  - Prefer Apple Foundation Models for any narration the PRODUCT depends on (a Claude
    subscription cannot be shipped to users).
- **V3 — iOS app (ultra-long-term).** The single biggest V3-driven choice made now:
  **prefer Apple on-device frameworks (Vision, Foundation Models) over PyTorch/CLIP
  wherever possible**, because torch does not port to iOS gracefully. Keep the engine
  (ingest → metrics → scoring → persistence) as a pure library with no CLI/web
  assumptions so a future Swift/iOS core can call or reimplement it.
  - Known tension to revisit at V3: the V1 taste model uses CLIP embeddings; iOS
    needs an Apple-native embedding (Vision feature prints). CLIP is accepted for
    V1/V1.5; the swap is flagged for V3.

## Local LLM Strategy
- **The narrator is a SWAPPABLE component; the insight graph is the product.** The
  reflection tools (`ask`, `style_over_time`) return grounded observation NOTES + a
  writing instruction; the CONNECTED model (Claude Desktop / ChatGPT over MCP, or an
  on-device model) composes the prose. The engine must NEVER pre-write the answer on the
  product surface — doing so hides whether the graph is rich enough and breaks for anyone
  without that local model. Grounding (no invented names, no reversed directions) travels
  with the notes, so honesty is narrator-independent.
- **Portable default narrator:** Apple Foundation Models on-device, invoked via an arm's-length
  helper process. It requires no API key and keeps the text transformation local.
- **Optional Claude narrator:** users who already have the Claude CLI installed and
  authenticated can select it in the local reflection app or use it through the CLI
  adapter. It receives grounded text notes only; photo pixels never leave the Mac through
  the engine. Claude is optional, not a required dependency.
- **Heavier local fallback:** an MLX-served small model if Foundation Models prove
  too constrained for synthesis.

## Relationship to the `photo-themes` Prototype (Parts Bin, Not Blueprint)
A prior prototype at `../photo-themes` solves ~70% of the ingest/metrics/taste/MCP/
guardrail surface with strong rigor. We REUSE its code by copying and adapting
modules into THIS self-contained repo (V2 ships as one repo). It is a parts bin:
- The SOURCE OF TRUTH is this project's specs, not the prototype.
- The prototype's statistics-first philosophy and module boundaries do NOT carry over
  as defaults; code is reshaped to fit OUR contracts, or dropped.
- Empirical lessons inherited: "moments over craft" (loved photos are not better-made
  by aesthetic score; a single child face-to-camera was the strongest signal) — so
  faces and resonance outrank EXIF-geometry craft in ranking weight; a fluent, wrong
  narration is the top failure mode — so narration is lead-with-facts + code-grounded.
