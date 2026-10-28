# Evaluation & Guardrails Design

This document is the source of truth for how we measure whether the Memory Graph
Engine actually works, and the guardrails that keep it honest and safe. It is
organized from the hardest, most central thing outward: **resonance / dispatch
first**, then the pipeline phases that feed it, then narration, then end-to-end.

Guiding principles (inherited and made explicit):

1. **Set the success bar AFTER measuring the base rate, never before.** The prior
   taste model failed because a 2x bar was set when 47% of random old photos were
   already liked. We measure base rates first, then set targets.
2. **Pre-register every quantitative test** (who is compared, the metric, the
   pass rule) BEFORE running it, and record failures honestly as they come.
3. **Deterministic layers are unit-tested to exact values; probabilistic layers are
   measured, not asserted.** Template facts must match the DB exactly; LLM prose is
   measured for groundedness, not pinned.
4. **Guardrails fail closed.** A privacy or egress guard that is unsure blocks.
5. **Never auto-destroy.** Nothing cherished is deleted without owner review.

A phase is not "done" until its evaluation passes its pre-registered bar AND its
guardrails are in place.

---

## PART A — RESONANCE & DAILY DISPATCH (the heart)

Resonance is the whole product. Everything upstream exists to feed a ranking that
makes the user stop and feel. It is also the hardest thing to measure, because
"made me feel something" has no EXIF field.

### A.1 What resonance is, formally

Resonance is a per-photo signal used in TWO ways:

1. **Dispatch ranking** — it orders the daily dispatch. Dispatch is also where
   resonance is EVALUATED (A.2–A.4), because a dispatch yields a clean yes/no signal
   to measure against.
2. **Insight / critique input** — resonance is a first-class signal for the LLM
   critique too. "The photos you keep returning to are quiet-intimacy portraits" is
   itself a resonance-driven finding. Resonance is not confined to dispatch.

Important nuance on the critique: the longitudinal critique STILL reports the
unweighted TRUE distribution (your actual shooting history, un-filtered by
resonance). Resonance is layered on TOP as a distinct lens — the GAP between what you
shot and what you resonate with is one of the richest insights ("your most-shot
quadrant was Q2, but the photos you are drawn back to are Q3"). Resonance enriches the
critique; it does not replace the unweighted truth, and it must never silently filter
the true distribution.

Grounding still applies everywhere: when the LLM uses a resonance value in an insight,
that value is recomputed from the DB by code, never invented (Part C).

The model is multiplicative so each factor degrades gracefully to 1.0 when
unavailable:

```
resonance(photo) =
    base_quadrant_affinity          # personal weighting over the soft [q1..q5] vector,
                                    #   with age applied INSIDE the Q1 term (see below)
  * candid_multiplier               # candid > posed
  * favorite_multiplier             # Library favorite/star (a signal, never a dominator)
  * known_person_boost              # FLAT in V1: any above-bar named person present
                                    #   (per-person weights are a V1.5 config feature)
  * taste_model_multiplier          # V1.5 learned "would I want to see this again"
```

Age is NOT a blanket multiplier on the whole score — an old clinical ultra-wide is not
made resonant by age alone. Age amplifies ONLY the nostalgia (Q1) component, inside
`base_quadrant_affinity`:

```
base_quadrant_affinity =
      w[Q1] * q_score[Q1] * age_boost(photo_age)     # age amplifies nostalgia here only
    + w[Q2_warm] * q_score[Q2] * warm_gate           # "warm aliveness", not raw awe:
                                                     #   Q2 gated by warm color-temp /
                                                     #   golden light (cold ultra-wides
                                                     #   do not read as warm aliveness)
    + w[Q3] * q_score[Q3]
    + w[Q4] * q_score[Q4]
    + w[Q5] * q_score[Q5]
```

- `w` is the personal priority vector. Starting weights (TUNABLE, not final):
  Q3 ≈ 1.0, warm-Q2 ≈ 0.8, Q4 ≈ 0.65, Q1 ≈ 0.5, Q5 ≈ 0.2. The two per-term modifiers
  (`age_boost` on Q1, `warm_gate` on Q2) are shown in the expansion above; all weights
  and modifier shapes are config.
- Every multiplier defaults to 1.0 when its signal is missing (e.g. HDD photo with
  no Library favorite, no known-person match). A photo is never penalized for a
  signal we simply could not compute.

The weights and multiplier magnitudes are CONFIG, not code (V2 portability), and are
the primary tuning surface.

### A.2 The base-rate measurement (do this FIRST, before setting any bar)

Before we can claim the ranker is good, we must know how good "random" already is.

- **Metric:** Draw N random canonical photos (stratified by era so old photos are not
  over-represented). Present them blind (no scores shown) and the owner marks each
  "would I be glad to see this in a daily dispatch? yes / no."
- **Output:** the base acceptance rate `p_random`. This is the number every later
  dispatch bar is set relative to. If `p_random` is 0.45, a "2x" target is
  impossible; the bar must be expressed as a realistic lift (e.g. precision@3 of the
  ranked dispatch must exceed `p_random` by a pre-registered margin we only choose
  AFTER seeing `p_random`).

### A.3 The resonance validation set (owner-labeled)

- A small, FIXED labeling set (~400 photos) is pre-chosen and stratified across
  quadrants/sources. The owner does NOT label the whole archive — the targets below are
  a few hundred clicks TOTAL, drawn from that set, done once (resumable):
  - `dispatch_worthy` (base rate): ~100 photos, yes/no "would I want this surfaced?"
  - `felt_quadrant`: ~150 photos, which emotional quadrant it evokes (1–5).
  - **pairwise preferences**: ~100 pairs, "which would you rather be shown?" — the
    primary signal for ranking quality (easier and less fatiguing than absolute scores),
    and exactly what DR-2 measures the ranker against.
- Delivered via the local loopback labeling UI (`memory-graph label`), which
  pre-renders the set's thumbnails once so it is fast, and guides the three tasks in
  sequence with a per-task progress bar toward these targets.

### A.4 Pre-registered dispatch tests (pass rules frozen before running)

- **DR-1 (precision@3):** Over a held-out set of days, the top-3 ranked dispatch must
  be `dispatch_worthy` at a rate exceeding `p_random` by a margin chosen after A.2.
  Pre-register the margin and the day count before running.
- **DR-2 (pairwise agreement):** The ranker's ordering of owner-labeled pairs must
  agree with the owner at a rate pre-registered after seeing a random-ordering
  baseline (random agreement is 50%).
- **DR-3 (no-collapse):** A single factor must not dominate pathologically — e.g. the
  dispatch must not become "only favorites" or "only the oldest photos." Favorites and
  age are SIGNALS, not dominators (owner-confirmed). Pre-register a diversity floor
  (e.g. across a month of dispatches, ≥K distinct quadrants and both sources appear,
  and favorites do not exceed a pre-registered share). Guards against the Q1 age_boost
  or the favorite_multiplier swamping the other signals.
- **DR-4 (repeat-avoidance):** The same photo (or its burst siblings) must not recur
  within a pre-registered window. Deterministic; unit-tested.

Each test records its result verbatim, including failures (PR-1/PR-2 in the prototype
both failed and were kept as-is; we do the same).

### A.5 Dispatch guardrails

- **Egress (HARD RULE): photo pixels NEVER leave the Mac.** The `daily_dispatch` MCP
  tool returns TEXT + a `http://127.0.0.1` view link only — never image bytes — because
  showing a photo to a hosted LLM sends it to that service. The owner clicks the link
  and the photo renders locally in their browser; the connected LLM only ever sees
  text. A regression test (`test_mcp_egress.py`) fails if image content re-appears.
  Private (Hidden-album) photos are excluded from dispatch AND refused by the viewer;
  default-deny on undetermined privacy.
- **No master access:** The dispatch path reads only the engine DB and the small
  derived thumbnail, never a Photos master or an HDD original re-open.
- **Determinism of selection:** Given the same DB state and date, dispatch selection
  is reproducible (seeded), so a dispatch can be explained and audited after the fact.
- **Freshness:** If the DB is stale relative to the last ingest, the tool says so
  rather than silently serving old picks (inherited `stale:true` discipline).

### A.6 Resolved resonance-model decisions (owner-confirmed)

- **`age_boost` is Q1-scoped, NOT global.** Age amplifies only the nostalgia (Q1)
  term; it does not blanket-boost every old photo. An old clinical ultra-wide gets no
  age bonus. (Formula in A.1.)
- **`known_person_boost` is FLAT in V1.** Any above-precision-bar named person present
  gives the same boost. Per-person weighting (e.g. a child > an acquaintance) is a
  **V1.5** feature delivered as a config file, leaning on the Library's identities.
- **"warm-Q2" is gated by warm color-temp / golden light** (`warm_gate`), so cold,
  clinical ultra-wides do not read as "warm aliveness." The exact gate (color-temp
  threshold vs golden-hour flag vs a blend) is a tuning parameter, set during A.4.
- **DR-3 enforces that favorites and age are signals, not dominators** (diversity +
  favorite-share floors).

Remaining as pure tuning knobs (shapes chosen during A.2–A.4, not open design
questions): the numeric weight values, the `age_boost` curve shape, and the precise
`warm_gate` cutoff.

### A.7 Tuning results (first real labeling pass — recorded honestly)

From 100 base-rate + 150 quadrant + 100 pairwise owner labels on the real 9.4k graph:
- **Base rate `p_random` = 44%** (worthy). Every dispatch bar is set against this.
- **Strongest preference signals** (pairwise winner-higher-rate): known_person 68%,
  Q4 63%, candid 62%, taste 61%; face_ratio was 39% (tight crops are NOT preferred).
  Base-rate lift: known_person present in 32% of worthy vs 7% of not-worthy (4.5x).
- **Resonance retuned to this data (Part C):** quadrant demoted to a soft minor input
  (`quadrant_weight`), known_person/candid/taste multipliers raised, Q4 weight lifted.
- **Validated ranking AUC = 0.68** (worthy > not-worthy; 0.5 = random). Worthy mean
  resonance 1.08 vs not-worthy 0.82. The ranker reliably surfaces what the owner would
  be glad to see — the dispatch earns its keep over random (44%).
- Honest headroom: 0.68 is good, not spectacular. Levers left: retrain taste on the
  full clean set, per-person weights (V1.5), more labels.

### A.8 Base-rate question ambiguity + the blur signal (correction in progress)

- **The base-rate question was ambiguous.** "Glad to see this in a daily memory?" was
  read as EITHER "nice to see it again" (low bar, ~44% yes) OR "would I want the app to
  send me THIS unprompted as the one memory today" (high bar, the dispatch target).
  These are different targets; the AUC 0.68 was measured against the ambiguous one.
  FIX: re-ask base-rate with the strict dispatch question ("Yes, send me this" vs "No,
  skip it", with a "most should be No" hint). Quadrant and pairwise labels are
  unambiguous and are KEPT; only the ~100 base-rate labels are redone (old ones archived
  for comparison). Then re-measure p_random and the AUC against the correct target.
- **Blur de-weight: DISABLED after the strict relabel (a surprising, honest finding).**
  Against the STRICT target, the owner's dispatch-worthy photos are actually LESS sharp
  (median 170) than the ones they'd skip (median 351) — cherished moments skew
  soft/candid/shallow-DOF, crisp shots skew mundane. Sharpness-alone AUC was 0.507
  (chance). A blur de-weight would rank the owner's taste BACKWARDS, so it is neutralized
  (`blur_deweight = 1.0`). Accidental-shake junk is still handled where it works: burst
  keyframe election picks the sharpest of near-identical frames.

### A.9 Strict-label results (the CORRECTED ground truth)

- **Strict base rate `p_random` = 23%** (was 44% ambiguous). About 1 in 4 photos is
  something the owner would want PUSHED to them — a selective, honest dispatch target.
- **Measured lifts vs strict worthiness** (worthy minus not-worthy mean): known_person
  +0.26, faces-present +0.22, Q3 +0.12; candid was NEGATIVE (-0.05) and Q4 near-flat
  (+0.03) — correcting the earlier ambiguous-label tuning (candid had looked positive).
- **Resonance retuned to this:** known_person_bonus up (2.0), a new face_present_bonus
  (1.3), candid neutralized (1.05), Q3 the top quadrant weight.
- **Validated resonance AUC vs STRICT target: 0.633 before retune -> 0.718 after.**
  0.718 is in the "genuinely good" band, measured against the CORRECT target. The
  dispatch reliably concentrates the 23% worthy photos near the top.
- Lesson restated: the question you measure against IS the model you get. An ambiguous
  label question tuned us to the wrong, easier target; fixing it (and honestly following
  the data even when it contradicted a prior like "blur = bad") produced both a truer
  score and a better one.

---

## PART B — PIPELINE PHASE EVALUATION (feeds resonance)

### B.1 Ingest

- **Eval:** On a known HDD subtree and a known Library slice, assert the expected
  count of candidates is produced; corrupt/truncated files are skipped and COUNTED,
  not fatal; derivative-miss (iCloud-only Library items with empty `path_derivatives`)
  are skipped and counted.
- **Guardrails (hard):**
  - Library adapter must NEVER access `photo.path` / a master. Enforced by a test that
    fails if the master-path accessor is called, plus a runtime assertion.
  - HDD is read-only: assert the mount is read-only; never write to the source tree.
  - Single-pass: a file on the HDD is opened once (EXIF + downsample together).
  - Resumability: a crash mid-run resumes from a checkpoint (upsert by stable id);
    a cache-integrity guard halts a "full run" if any per-photo result was recomputed
    mid-run (inherited — this caught a half-populated index before).

### B.2 Rejection (clutter / documents)

- **Base rate first:** measure what fraction of the archive the filter flags.
- **Eval:** owner-labeled set of true-clutter vs true-photo. The asymmetric loss is
  explicit: a FALSE REJECT of a cherished photo is far worse than a false keep. Tune
  for high recall of real photos; report the confusion matrix, not a single accuracy.
- **Guardrail:** never auto-delete. Flagged clutter is quarantined to a review surface
  the owner approves on first run (inherited D12 first-run review). Rejection only
  removes from the resonance/dispatch candidate pool, not from disk.
- **FINDINGS (real-archive labeling pass, recorded honestly):**
  1. **Dupes were the biggest leak.** Burst clustering only merges TIME-ADJACENT frames,
     so near-identical photos minutes apart — or the same photo on both HDD and Library —
     survived as separate canonicals (56 dupe pairs in a 276-photo labeled set). Fixed
     with a GLOBAL near-duplicate pass (`global_dedup`) using LSH banding over the pHash
     (pigeonhole: bands > max_dist guarantee a shared band for true near-dups), confirmed
     to catch 56/56 pairs efficiently.
  2. **A blanket text-coverage reject wrongly rejected a cherished child photo** (text
     0.27 from a background sign). Fixed: text-doc reject fires only when NO real-photo
     content is present (people/child/face/outdoor/nature). Validated: 0 of 44 worthy
     photos rejected after the fix.
  3. Document/blank thresholds were slightly too loose (blanks at contrast 0.044-0.056,
     docs at text 0.08-0.12); tightened with the protective guard so nothing cherished
     is lost.
  4. **"Not worthy" != "clutter".** 54 of 56 not-worthy photos were real but mundane
     photos, not clutter — that is a RESONANCE signal (low taste), not a rejection one.
     The rejection filter was doing better than the raw complaint implied; the true leak
     was dupes + a handful of docs/blanks.

### B.3 Burst clustering & canonical keyframe election

- **Eval:** hand-label N bursts with the owner's preferred frame; measure how often
  the elected canonical matches (sharpness via Laplacian variance, eyes-open via face
  landmarks, favorited-member preference). Report top-1 match and "within owner's
  acceptable set" rates.
- **Guardrail:** deterministic and seeded; burst grouping parameters are config and
  pre-registered before the election eval runs.

### B.4 Face stack (NEW — the biggest new build)

Three sub-capabilities, each with its own bar:

- **Detection:** recall of faces on a labeled set (missing a face loses a resonance
  signal). Report recall at a fixed precision.
- **Named recognition.** CRITICAL: a wrong name is worse than no name (a misattributed
  "known person" corrupts resonance). A match must clear a PRE-REGISTERED precision bar
  before it feeds resonance; below the bar it contributes "a person is present" but not
  identity.
  - **Library photos (the common case): identity is READ, not inferred.** P5 found that
    osxphotos exposes the Photos Library's OWN per-face `name` (Photos already did
    high-quality recognition), plus `has_smile` and `yaw`. So Library photos get
    authoritative names and real candidness for free — no embedding, no matching, no
    precision bar needed. This is the reliable path for the vast majority of photos.
  - **HDD-only photos (the hard case): embedding match, gated hard.** For HDD photos
    with no Library twin, we embed the face crop (Apple feature print) and match against
    a gallery built from Library named faces.
    FINDING (P5 live probe, recorded honestly): generic `VNGenerateImageFeaturePrint`
    on face crops is WEAK for identity — a probe scored 2/4 correct at a 0.55 bar, with
    wrong matches as confident as 0.88 and right/wrong similarities OVERLAPPING. So:
    (a) the 0.55 default bar is too low and must be tuned UP against a labeled set;
    (b) at a safely high bar, HDD-only recognition will name few faces, and that is the
    correct conservative behavior (present-but-unnamed beats wrongly-named);
    (c) a dedicated face-embedding model (vs. a general image feature print) is the
    likely upgrade if HDD-only naming needs to improve — flagged, not built in v1.
  - pHash bridging still gives HDD photos that duplicate a Library photo the Library's
    authoritative names directly (no embedding risk).
- **Candidness proxy (gaze + smile + face count):** labeled posed-vs-candid set;
  measure accuracy. This is a rough heuristic for v1; its job is only to nudge, so the
  bar is modest and its multiplier magnitude is kept conservative until validated.
- **Guardrail:** identity never leaves the device except as part of a chosen dispatch;
  a face match below the precision bar may not assert a name in narration.

### B.5 Metrics (optics, geometry, light)

- **Eval:** against synthetic/known inputs where ground truth is exact — e.g. a photo
  with known EXIF yields the exact focal/aperture/shutter; a programmatically tilted
  image yields the known horizon angle; a known-location-and-time yields the correct
  golden-hour flag (NOAA solar elevation is deterministic). Negative-space ratio from
  saliency is checked against hand-marked masks on a small set.
- **Guardrail:** metrics are pure functions of (image, exif) — no hidden state — so
  they are fully unit-testable and reproducible.

### B.6 Quadrant soft scorer

- **Pre-registration:** the range/ramp parameters for each quadrant are frozen in
  config BEFORE the accuracy eval, so we cannot fit to the labels post hoc.
- **Eval:** against the owner's `felt_quadrant` labels (A.3). Because scores are soft,
  measure (a) top-1 argmax agreement, and (b) rank correlation between the score
  vector and the owner's sense (a photo can be "mostly Q1, a bit Q3"). Report the
  confusion matrix across quadrants and inspect systematic overlaps.
- **Guardrail:** no boundary cliffs — a 1mm focal change cannot flip a hard class
  (smooth ramps). `dominant_quadrant` is a convenience argmax, never the only stored
  value; the full vector persists.
- **FINDINGS (P6 live probe on real 2007 photos, recorded honestly):**
  1. A quadrant must not fire strongly from a single cue. First `_blend` let one
     cue (e.g. a 50mm focal) saturate Q1 to 1.0; fixed to a corroboration-weighted
     combine with a `core`-cue cap, so a quadrant needs its DEFINING cue present.
  2. Q5 (provocative/tension) over-fired badly — 5/12 real photos, several at 1.00 —
     because the P4 horizon detector reported confident tilt on TELEPHOTO shots with no
     true horizon (latching onto a branch/slope). Fixed by adding a line-fit CONFIDENCE
     to `horizon()` (fraction of columns agreeing with the fitted line) and gating the
     Dutch flag — and thus the Q5 tilt cue — on it. Re-run on the same photos: Q5
     dropped to 0 dominant (scores 0.03–0.17), and the shots correctly read Q3/Q4.
     Lesson: a geometry signal must be trustworthy before an emotion is inferred from it.
  3. **The owner's felt quadrant vs the engine (real labeling pass, 150 quadrant
     labels), THE central B.6 finding:**
     - Pure EXIF/geometry router: top-1 agreement 25%. Barely above chance (20% for 5
       classes). The router systematically missed Q1 (nostalgia) and Q3 (tenderness),
       which are CONTENT/memory feelings, not technical-execution signatures — the
       "moments over craft" lesson, confirmed on the owner's own labels.
     - Adding CONTENT cues (Vision scene labels + faces drive Q1/Q3/Q2/Q4): top-1 rose
       to 29% — modest, because the deeper issue is that the owner's quadrants OVERLAP
       (a warm photo of people at home is nostalgic AND tender AND domestic at once).
     - **But the SOFT VECTOR is right even when the argmax is not:** felt quadrant is in
       the engine's top-2 scores 69% of the time, top-3 90%. The single
       `dominant_quadrant` is a lossy, arbitrary tiebreak between genuinely-close
       quadrants; the vector carries the truth. CONCLUSION: do not chase single-label
       accuracy; treat the quadrant vector as a soft input and lean on resonance
       (faces, favorites, taste, known-people) for ranking — the owner's actual
       priorities. This is the Part C direction, now empirically justified.

---

## PART C — NARRATION & THE INSIGHT GRAPH (the scariest failure mode)

The top risk, learned the hard way by the prototype: a **fluent, confident, WRONG
narration.** A connected LLM twice told a plausible wrong story from stale memory. The
prototype also could not read *direction* in prose — it once said "more" where the
evidence said "less" (its documented limitation #8). Our design must make both
failures structurally impossible, WITHOUT turning the prose into a stat-salad.

### C.0 Core architecture: the memory graph is an INSIGHT graph

The LLM does NOT consume raw metrics and invent a story. The engine first derives and
COMMITS an **Insight** — a structured, first-class record authored by code from
deterministic metrics + resonance. The LLM then narrates THAT insight: free to find
the human meaning, fenced by the insight's own committed claim.

```
deterministic metrics + resonance ─► INSIGHT (engine-authored, structured, committed)
                                          claim, direction, magnitude, evidence ids,
                                          resonance angle, allowed entities (names/places)
                                                 │
                                                 ▼
                                   LLM narrates THIS insight (voice + human meaning)
                                                 ▼
                              verify prose stays INSIDE the insight's fence
                              (no fact/name/place outside it; no contradicted direction)
                              fail twice ─► SAY SO + fall back to the plain template
```

Why this is better than "LLM dresses up a FactSheet":
- The valuable node is the insight, not the photo. Photos are EVIDENCE hanging off
  insights. The graph of insights is the product; the LLM learns from and narrates it,
  it does not AUTHOR it.
- **Direction and magnitude are decided by CODE and frozen in the insight** before the
  LLM runs. The "said more where data said less" failure cannot occur — the LLM may
  only narrate the committed direction.
- We never ask the LLM to emit structured claims (which would stilt the prose and
  defeat its purpose). The structure lives in the insight record; the LLM writes
  only prose. The real value is interpretation, which is exactly what we free it to do.

### C.1 The Insight record (engine-authored, the ground truth)

An Insight is committed to the DB and carries at minimum:
- `claim`: the plain machine statement ("most-shot quadrant Q2; most-resonant Q3").
- `direction` / `magnitude`: fixed by code from the data (e.g. Q3 resonance share is
  HIGHER than its shooting share, by X points).
- `evidence`: the photo ids (and metric values) that support it, recomputed on demand.
- `resonance_angle`: how resonance figures in (this is a first-class insight arm, per
  Part A — not confined to dispatch).
- `allowed_entities`: the exact set of names (above the recognition precision bar) and
  places the narration is permitted to use. Nothing outside this set may appear.
- `era`: the life-era context.

Insights are the frozen-eval unit (see D.1): an owner-approved insight must keep its
direction and qualifiers and stay within a tolerance band, or a change is a deliberate
version bump.

### C.2 Template layer (deterministic facts)

- Every template fact ("50mm, f/1.8, eye-level, golden hour, 12 years ago") must match
  the DB exactly. Unit-tested to exact strings/values. This is the layer we TRUST and
  the floor we fall back to when the LLM cannot be trusted for a given insight.

### C.3 LLM layer — narrates an insight, within its fence

- Input to the LLM: ONE insight record + its template facts + the evidence photo(s).
  Nothing else — no raw EXIF dump, no free-text memory, no filesystem/network/tools.
- **Groundedness verification (code, not trust)** of the returned prose against the
  insight:
  - No numeric setting (focal/aperture/shutter/ISO/age/count) that is not in the
    insight's facts (within tolerance). An unlisted number → reject.
  - No person name outside `allowed_entities` (so no name below the precision bar, and
    no invented person) → reject.
  - No place/date outside the insight's facts → reject.
  - **Direction/magnitude consistency:** the prose may not assert a comparison that
    contradicts the insight's committed `direction`/`magnitude`. Because direction is
    fixed in the record, this is checkable without parsing English for direction — we
    check the prose does not contain the OPPOSITE committed terms/claims, not that it
    re-derives the number.
- **Interpretation is allowed and wanted**, but must be (a) LABELED as interpretation
  (a guess), and (b) consistent with the insight (a Q3 tight-face-crop insight may be
  read as "tenderness"; a Q2 ultra-wide landscape may not). Interpretation lives
  INSIDE the fence the insight draws. (Tightening this is a V1.5 tuning item.)

### C.4 Failure behavior — honest, never silent

- On a verification failure, regenerate ONCE with a sharper instruction. If it fails
  AGAIN (the "twice" rule — stop patching, change approach), the system does NOT
  quietly ship the plain template. It **explicitly tells the user it could not generate
  anything meaningful and is falling back to the plain facts**, then shows the template
  narration. Honest over seamless. The worst case is "boring but true AND labeled as
  such," never "fluent but wrong."

### C.5 Privacy harness (pre-send, fails closed)

- Every string in an insight + facts is scanned BEFORE it reaches the LLM: home
  location, real place names where pseudonyms are required, and anything owner-private.
  A hit BLOCKS the call (no silent strip-and-retry) and falls back to the template.
- **Owner-private source (V1):** any photo in the Photos **Hidden** album is off-limits
  — never narrated, never dispatched off-device. Per-person off-limit marking (mark a
  specific person as never-narrate) is a **V1.5** feature. Default-deny: undetermined
  privacy status is treated as private.

### C.6 Arm's-length invocation

- Primary: Apple Foundation Models (portable to V2/V3). Optional author-only upgrade:
  headless `claude -p`. BOTH run as a separate process, prompt via stdin (not argv, so
  it never hits the process list), empty working directory, minimal environment, no
  tools, no network, no session persistence. The LLM is a pure text transform over one
  insight — it cannot reach the filesystem, the Photos Library, or the network to pull
  in a stray "fact." Both paths are held to the SAME C.3 groundedness checks.

### C.7 What we measure

- **Groundedness is hard-enforced** (C.3), so the metric is the raw-output failure
  RATE: how often does the LLM produce something that fails verification? A high rate
  means the insight/prompt needs work, never that we loosen the gate.
- **Eloquence is measured, not pinned:** a small owner-rated sample
  ("True / Partly / Wrong") tracked as a quality signal, with regressions logged.
- **Template facts** are unit-tested to exact values — the ground we stand on.

### C.8 Critique distributional accuracy (distinct from narration)

- Separate from narration prose: the longitudinal critique's UNWEIGHTED distributions
  (e.g. "quadrant shares per era") and its resonance lens must match the DB exactly —
  they are code-computed aggregates, unit-tested against the records. An insight may
  only be committed if its `direction`/`magnitude` are reproducible from these
  aggregates (ties back to C.1 and the D.1 frozen-eval suite). This guarantees the
  critique's FACTS are correct before the LLM ever narrates them.

---

## PART D — END-TO-END & CROSS-CUTTING GUARDRAILS

### D.1 End-to-end eval

- The frozen-eval unit is the **Insight** (C.1). A set of owner-approved insights and
  "known-good dispatches" becomes a regression suite with tolerance: an approved
  insight must keep its committed `direction` and qualifiers and stay within a
  pre-registered `magnitude` band, and its evidence ids must still support it. A
  deliberate change to the method is a written version bump, not a silent drift.

### D.2 Cross-cutting guardrails (always on)

- **No-network guard:** outbound connections are blocked by default in all commands;
  tests run with sockets disabled. The only sanctioned egress is the explicit MCP
  dispatch (Part A.5) and the optional LLM helper, each arm's-length.
- **No cloud master download:** the Library master path is never touched (B.1).
- **Publish-check (V2):** before sharing the repo, scan all tracked files AND git
  history for home/coords/real place names/emails/keys; report kind/file/line, never
  the value; fail closed on any hit. Personal config/data are gitignored.
- **Cache integrity:** a full run halts if any per-photo result was recomputed mid-run.
- **Reproducibility:** seeded selection; given the same DB + date, dispatch and
  critique are reproducible and auditable.

### D.3 What we explicitly do NOT claim

- We do not claim the quadrant mapping is "correct" in any absolute sense; it is
  measured against the owner's felt labels and tuned, nothing more.
- We do not claim the LLM's interpretations are true; they are labeled as
  interpretation and are only constrained to be GROUNDED in the facts, not CORRECT in
  their psychology.

---

### B.7 Taste model (P9 — FINDING, recorded honestly)

- **Embedding choice validated:** the taste model uses APPLE VISION FEATURE PRINTS
  (768-dim), not CLIP — keeping the engine torch-free and iOS-portable (V3 north star).
- **Live result on the real Library** (1428 photos): favorite base rate **1.8%**
  (26 favorites), held-out **AUC 0.831**. The low base rate is exactly why the bar is
  set AFTER measuring it, not before (the inherited lesson). AUC 0.83 on an Apple-native
  embedding BEAT the prototype's CLIP-based taste model (AUC 0.79) — portable AND better.
- The score feeds resonance as the `taste_score` multiplier; it is an input, not an
  oracle. A dedicated collected-thumbs retrain (prototype's v2) is the V1.5 follow-up.

### C.9 Richer insight derivations (post-launch, owner feedback)

The owner found the critique "flat" — it only emitted "dominant quadrant per era" and
read like a bare `style_over_time` dump. The dispatch, by contrast, felt great (all 3
daily memories were ones they'd love back). So the fix was to EXPAND the insight
library, not the architecture: added code-committed cross-era TREND insights derived
from data already in the DB — people-centrality (known-person share over time),
focal-length drift, the person who became most-photographed (names gated to the
recognition bar via `allowed_entities`), and group-vs-intimacy shift. On the real
archive these surface the genuine arc: recognized-loved-one share 0% -> 52% across two
decades, and the owner's partner as the heart of the 2020s. Direction/magnitude are
still code-fixed; the LLM only narrates.
- BUG caught: a missing `import json` in critique.py made `json.loads` raise NameError,
  silently swallowed by a bare `except`, dropping EVERY people insight. Fixed the import
  AND narrowed the except to (JSONDecodeError, TypeError) so a coding error can't hide
  again. Lesson: broad excepts hide real bugs — narrow them.

### C.10 Critique VOICE, not more stats (owner feedback, second miss corrected)

Adding more trend stats made the critique WORSE — "more stat salad, less insight." The
output read like an analyst's table because that is what the engine fed the LLM (bare
aggregates) and, over MCP, the tool returned bare claims so Claude tabulated them. Root
cause was framing, not missing data. Fixes:
- The committed insight `claim` now leads with FEELING/POSTURE (meaning.py maps each
  quadrant to a lived intent — "preserving a memory the way the eye lived it"), with the
  number as incidental evidence.
- A separate critique narration (`narrate_critique`) with the INTENT-DOC VOICE: one warm
  second-person paragraph, tables/bullets/percentages forbidden, and — per the owner —
  a GROUNDED INTERPRETIVE LEAP is explicitly ALLOWED (the insight must feel personal,
  which pure data cannot). Verified loosely but honestly: no invented named person
  (hard-checked), directions not reversed (evidence block is the backstop).
- Output is **A + B**: A = the interpretive narrative (the experience); B = the evidence
  block (the eval guard — the owner can trace every move in the prose to a real
  observation). B doubles as the honesty check on the leap.
- Over MCP the tool now PRE-WRITES the vetted narrative (local `claude -p`) so the
  connected client gets finished prose, not stats to re-tabulate. Result on real data:
  a genuine reflection ("your camera quietly turned toward the people you love ...
  belonging doesn't need a crowd"), not a table.

### C.8 realized (P10): critique facts are code-computed and verified

- The longitudinal critique derives Insights from DB aggregates (per-era unweighted
  shot distribution AND resonance-weighted distribution; the GAP is a first-class
  insight). Direction/magnitude are committed by CODE before any LLM runs. Aggregation
  is pure and unit-tested. Live validation produced a correct 2000s insight (29% of 146
  photos Solitary) and a grounded narration that PASSED verification — rich feeling,
  zero invented facts.

---

## Status & sequencing

Evaluation is built ALONGSIDE each phase (as designed). Implemented state:

1. **Resonance / dispatch (Part A)** — implemented (P7); weights are the pre-registered
   first draft, awaiting the owner-labeled tuning pass (A.2–A.4).
2. **Narration groundedness (Part C)** — implemented (P8/P10); the code-verified fence
   and honest fallback are live and tested; validated against real `claude` output.
3. **Phase evals (Part B)** — each phase shipped with tests; live findings recorded in
   B.4 (faces), B.6 (quadrants), B.7 (taste).
4. **End-to-end (Part D)** — the pipeline runs end to end on real data; the owner-labeled
   validation sets (A.2/A.3) and the frozen-insight regression suite (D.1) are the
   remaining tuning/validation work, to be done with real labels.

The MACHINERY is built and tested; the TUNING (weights, bars, thresholds) is the
deliberate next step, and by design it waits for the owner-labeled sets so it is done
against a target, not blind.

---
## E. Zero-egress guarantee (hard rule: photo pixels never leave the Mac)

The owner's rule is absolute: showing a photo to a hosted LLM IS sending it to that
service's servers, so the MCP surface must never return image bytes.

- `daily_dispatch` returns TEXT (grounded facts) + a `http://127.0.0.1` link served by a
  local loopback viewer (`viewer.py`). The owner clicks it; the photo renders in their
  own browser, pixels transiting only localhost. The connected LLM only ever sees text.
- Verification is structural, not trust: `tests/test_mcp_egress.py` scans the entire
  `mcp_server` source and fails if `ImageContent`, `b64`, or `base64` ever appear — so a
  future edit cannot silently reintroduce pixel egress. The viewer is asserted to bind
  `127.0.0.1` only (never `0.0.0.0`), and private (Hidden-album) photos stay excluded.

## F. The insight graph realized + free-form `ask` (depth = node count)

The intent doc says the insights ARE the graph — "a big set of nodes for the LLM to pick
from, maybe hundreds." Implemented as:

- **Generation (`generate.py`).** The engine commits a POPULATION of tagged `Insight`
  nodes across dimensions (color, framing, people, light, solitude, groups, style) for
  the whole archive AND each era, plus cross-era `change` nodes. Each node leads with a
  feeling-framed claim (the number is incidental evidence) and carries retrieval `tags`
  (now a first-class field on `Insight`, persisted in a migrated `insights.tags` column).
  Direction/magnitude are committed by CODE before any LLM runs — the same contract as
  the critique. Pure and unit-tested (`tests/test_generate.py`).
- **Selection (`ask.py`).** `select_nodes(question, insights)` tokenizes the question
  (light stemming so "colours"/"warmer" hit the "color"/"warm" tags), scores nodes by
  tag overlap + era targeting (a decade mention like "the 2010s" surfaces BOTH
  "early 2010s" and "late 2010s" nodes; a different named era is pushed out), and widens
  for longitudinal questions. **Depth is node count:** a NARROW ask keeps only
  on-topic nodes, deduped to one representative per theme (a sentence); an ERA-TARGETED
  ask keeps that era's several dimensions (mid-depth); a BROAD ask pulls the change +
  per-era population (a paragraph). Pure and unit-tested (`tests/test_ask.py`).
- **Narration (`narrate_answer`).** Same grounded voice and honesty fence as the
  critique (no invented name, no reversed direction; A narrative + B evidence block),
  with a LENGTH instruction derived from the selection's target sentence range so
  verbosity tracks the question's scope. Honest fallback to the evidence block when no
  backend or on a failed grounding check.
- **Surfaces.** `memory-graph graph` builds/persists the graph (replacing prior graph
  nodes so stale observations never linger); `memory-graph ask "..."` and the MCP
  `ask(question)` tool answer live. The MCP tool is TEXT-ONLY (covered by the E guard).

**Live validation (real 9,639-photo DB).** A narrow ask ("warmer or cooler colors?")
answered in two grounded sentences. A broad ask ("how has my photography changed over
two decades?") produced a genuine arc — the move toward the people loved, a stable warm/
cool palette, a framing loop back to wide — and honestly flagged what it could NOT say
("I can't say from this how that inner tone has shifted since"). Not stat salad: feeling,
grounded, personal. 264 tests passing, lint clean.

---
## G. Camera focal correction + re-scorable metrics (ground-truth fix)

A real data-integrity bug found by the owner's instinct about his 2000s Kodak era.

**The bug.** Many cameras omit the 35mm-equivalent focal from EXIF and record only the
raw physical/lens focal: all Canon bodies in this archive (DSLRs and PowerShots), the
Kodak DC5000, a Sony DSC-W1, early iPhones. With an empty `[framing.crop]` table the
engine stored the raw focal verbatim — a DC5000's 6.3mm, a Canon XTi's 22mm lens focal —
so every such frame was bucketed as ultra-wide. This corrupted the framing insight AND
skewed the quadrant vector (ultra-wide feeds Q2 visceral). The 2000s read "wide /
visceral" when it was really "normal, eye-level / nostalgic."

**The fix (config).** Per-camera crop factors in the gitignored `config.local.toml`
`[framing.crop]`, each cross-checked: raw-focal-range x factor == the camera's documented
35mm-equivalent range (e.g. DC5000 6.3-12.6mm x 4.762 == its documented 30-60mm;
full-frame 5D bodies -> 1.0 so they are never wrongly scaled). After correction the 2000s
framing median moved 28mm -> 45mm (wide -> normal) and the 2000s dominant quadrant moved
Q2 -> Q1 (visceral -> nostalgic) — the true character of point-and-shoot family years.

**The investment (B2: re-scorable metrics).** The root structural problem was that a
quadrant re-score needs inputs that were NOT persisted — the Vision scene_labels (the
content cues that fixed Q1/Q3) and the horizon dutch gate (Q5). So any optics fix forced
a full Vision re-decode. We now persist `scene_labels`, `horizon_dutch`, `focal_raw`, and
`exif_focal35`, and added `pipeline.rescore_row` (+ `memory-graph rescore`) that rebuilds
the router inputs from the DB and recomputes the quadrant vector + resonance with NO
pixels. `recompute_focal_35mm` always derives from the RAW focal (never a previously
stored equivalent), so a re-score is idempotent and immune to prior state.
  - FAITHFULNESS is tested, not asserted: `test_rescore_matches_full_analysis_when_
    nothing_changes` proves `rescore_row` reproduces the full pixel analysis's q-vector
    and resonance to 1e-9. Live on the real 9,639-photo DB, `rescore` ran in 0.68s with
    0 quadrant changes (vs a ~40-minute Vision re-decode) — confirming both speed and
    fidelity. Pre-B2 rows are flagged non-rescorable (their Vision cues were never saved)
    and must be re-analyzed from pixels.
  - One-time migration: the existing archive was re-analyzed fresh (full Vision pass,
    crop table in place) to populate the new columns; from here, optics/config fixes are
    a sub-second re-score. The DB was backed up first (`memory_graph.db.pre-cropfix-*`).

**Public-release requirement (recorded in product.md V2).** A first-run tool MUST scan
the user's whole library (Photos Library included — read focal/model from photos.db via
osxphotos, never opening a master), list cameras lacking a 35mm-equiv focal, propose +
confirm crop factors into their local config, then analyze. Without it, every new user's
older-camera eras are silently mis-insighted exactly as this archive was. 273 tests
passing, lint clean.

---
## H. Owner identity (the photographer is sometimes the subject)

Found by the owner testing the signature nodes: a "signature" for the owner themselves
was really describing how OTHERS photograph them — attributing their technical choices
to the owner. The engine assumed "you" == the photographer, but when the owner is in the
frame, someone else held the camera.

**The model.** A configured `[owner] self_names` (gitignored, personal; empty default)
names the owner's own identity in the Library face-naming. The insight generators then
treat owner-present photos correctly:
  - **No signature** for the owner — you do not choose how you are framed by others.
  - **Excluded from "people you photograph most"** — the owner appears because others
    shot them, not as a subject the owner chose.
  - **Pairings with the owner are reframed** from "you photographed X and Y together" to
    "you appear alongside X" (you are IN the frame, not behind it).
  - **No arrival/fade** for the owner in their own archive.
  - **New owner-aware nodes** (`self:presence`, `self:beside`): "you are a frequent
    subject yourself, photographed by others... a record of being seen rather than
    seeing" and who is most often beside you. This is the emotionally true mirror of the
    authored-people nodes, and a DIFFERENT claim than authorship.

Retrieval wrinkle fixed: a self-question ("how often am I in my own photos?") strips the
pronouns I/me (stopwords), so the self nodes are tagged with surviving CONCEPT words
(own, myself, self, photographed, subject) plus the owner's own name — tested so it does
not silently dead-end. Live result: a grounded, correctly-framed reflection on being the
subject. 288 tests passing, lint clean.

**Public-release requirement (recorded in product.md V2):** the first-run setup must ask
the user to identify themselves among the named faces, exactly as it asks them to mark
cameras — without it, every user's self-portraits-by-others are mis-attributed.

---
## I. Per-person life trajectories (the right way to grow the graph)

The owner noticed the node count had DROPPED (92 -> 85) and asked why. The honest answer:
an earlier over-firing arrival rule had emitted 8 bogus "arrived" nodes (everyone who
post-dated the sparsely-named 2000s), and tightening it to fire only on a genuine shift
collapsed those to 1 — a correct quality fix, but it exposed that the arrival/fade
dimension was now UNDER-producing. The right response was to add TRUE nodes, not relax
back to noise.

`_trajectory_nodes` now classifies each recurring person's per-era SHARE sequence into a
life-story shape instead of a single first-vs-last comparison:
  - **arrived** (absent early, central late — e.g. the family pet, two newer friends),
  - **faded** (present early, gone late — e.g. an older relative, an extended-family
    member),
  - **peaked** (rises to a middle-era peak then recedes — e.g. a relative tied to "the
    face of one particular season"),
  - **rising** (steady growth — a friend growing more present),
  - **steady** (prominent throughout — the partner and a close family member, "never
    really came or went").
Shares are only trusted in eras with >= 40 named photos (face-naming is sparse early, so
a 56%-of-9 reading is noise). A low presence floor (3%) lets secondary faces earn a shape
node — a relative prominent only in the late 2010s IS a real chapter — while 'steady'
additionally requires real dominance so a minor face is never called a 'constant'.

Result: the graph grew 85 -> 93 with 9 trajectory nodes, each a distinct, accurate shape
(vs the 8 bogus arrivals removed earlier). A life-story question now weaves them into a
grounded arc: "[an older relative] drifted out... [a relative] rose in the late 2010s
then receded... [the pet] and [two friends] weren't in your early pictures and are now
faces you keep... [the partner] and [a family member] never really came or went."

Retrieval: each shape carries the words people ASK with ("came and went", "constant",
"new", "left", "gone"), so a life-story question pulls the right shapes rather than any
node that merely mentions people — covered by a regression test. 291 tests, lint clean.

---
## J. Unknown-face presence (project A: honest aggregates, no false identity)

The owner noted the graph only reasons about NAMED people, ignoring a large unnamed
population (measured: 10,821 detected faces, only 2,543 named — ~76% unnamed; 3,967 of
5,612 people-photos have faces but zero names). Two kinds of "unknown face" exist, and
only one is honestly buildable in V1:
  - **Per-unknown-person** ("the same unknown person across N photos") — NOT possible:
    the engine stores only a per-photo face COUNT, no per-face embedding/cluster/UUID,
    and discards Apple's `_UNKNOWN_` clusters at ingest. Deferred to project B (wire the
    dormant identity Gallery: per-face embeddings + clustering + a recurrence threshold).
  - **Aggregate unnamed presence** — buildable NOW, and shipped here.

`_unknown_presence_nodes` + `_unknown_trend_nodes` emit, per scope and across eras, how
large the unnamed world is: a share-gated (>=25%) node for people-photos whose faces were
never named, a node for named-people-AMONG-others, and a trend node for the shift. On the
real archive the arc is striking — unnamed share 99% (2000s) -> 94% -> 58% -> 18% (2020s)
-> a 'narrowed from a wide field of faces toward the specific people you name' trend.

Truthfulness guard: the CLAIM always says "unnamed", never "strangers" — an untagged face
may be an unlabeled loved one, especially in older eras where Apple's naming is sparse. A
test asserts "stranger" never appears in an unknown_people claim. (The LLM MAY interpret
"never named" as strangers in its narrative, but that leap rests on the true fact and the
evidence block shows "never named".) Graph grew to 99 nodes; 295 tests, lint clean.

Project B (next): per-face embeddings via Apple Vision + clustering of unnamed faces,
surfacing RECURRING unknown people ("someone you haven't named appears in 60 photos,
often beside [the partner] — worth naming?"), gated by a recurrence threshold so one-off
faces never become nodes. Needs a faces table, an embedding pass, and tuning to avoid
merging distinct strangers — its own careful build.

---
## K. The narrator is swappable; the insight graph is the product (MCP voice fix)

A core architecture clarification from the owner, triggered by the MCP `ask` returning
"word salad" (tables) while the console read beautifully. The diagnosis exposed a design
mistake AND the right product thesis.

**What was actually happening.** Every `ask` goes: (1) `select_nodes` retrieves relevant
graph nodes [pure Python — the real work], (2) `build_answer_prompt` wraps them in the
voice directive, (3) a narrator writes the prose. The CONSOLE answers the owner loved
were written by a LOCAL `claude -p` at step 3 — not "the engine." Over MCP, Claude
Desktop launches the server with a stripped PATH, so local `claude` wasn't found, the
tool fell back to dumping RAW evidence with a weak note, and the hosted model tabulated
it. Compounding it, the tool DESCRIPTION framed the tool as one that "answers", so the
host model treated our output as data to re-present.

**The thesis (owner-stated).** The product is the INSIGHT GRAPH. The connected model
(Claude/ChatGPT) must be able to construct the reflection from the nodes ALONE —
pre-writing the answer with a local CLI defeats the point (it hides whether the nodes are
rich enough, and no cloned-repo user has that CLI). So:
  - The MCP `ask` / `style_over_time` tools now return ONLY the grounded notes + a WRITING
    INSTRUCTION ("these are NOTES to write FROM, not a dataset to display"); they NEVER
    pre-write prose. No local-backend dependency on the reflection path.
  - The tool DESCRIPTIONS now instruct the host assistant to WEAVE the notes into one
    flowing reflection and explicitly not to tabulate — the description is where a host
    model's behavior is actually shaped.
  - Grounding travels with the notes (the directive fences invented names and reversed
    directions), so honesty does not depend on which model narrates.
  - The real product test becomes: can a capable connected model write a good reflection
    from the selected nodes? If not, improve the GRAPH (coverage/wording), not the prose.

**On node count.** The owner repeatedly asked if too few nodes were sent. For a broad
question the payload is already rich (12 grounded, feeling-framed notes for the two-decade
question) — the salad was a protocol/framing failure, not a quantity failure. Node COUNT
matters for COVERAGE (an unusual question finding something), not for whether a
well-covered question can be narrated.

**Future direction (owner flagged as useful).** Narrator-as-swappable-component is the
V2/V3 north star: ship Apple Foundation Models as the zero-dependency default narrator,
but ANY MCP client's model can narrate the same notes. The graph is the moat; two users
with the same photos and different narrators should get the same INSIGHTS, worded
differently. (The local `claude` backend remains only a dev-console convenience, not part
of the product surface.) 296 tests, lint clean.

---
## L. Local reflection app + on-device narrator (owning the loop)

Over MCP, Claude Desktop kept tabulating the grounded notes and even stated it "ignored
the writing instructions" — because the HOST owns its system prompt and treats tool
output as data to present, not instructions to obey. No amount of in-payload framing
fixes that. The owner's call (option 4): build a local app where WE own the loop —
question -> `select_nodes` -> call an LLM with our voice as the actual SYSTEM PROMPT ->
render. This does not abandon the thesis; it proves it (retrieve = the product; narrate =
a swappable component we now drive directly).

**Apple Foundation Models narrator — built and working on-device.** A Swift helper
(`helpers/foundation_narrate.swift`, compiled on first use, cached under data/) reads a
prompt on stdin and returns the on-device model's reply. The `foundation_models` backend
(previously a stub) now shells out to it. Verified on macOS 27 / Swift 6.4: grounded,
passes the name/direction fence, NO api key, NO network — the V2 standalone ideal made
real. It reads a touch more lyrical than Claude; Claude is tighter. Both produce real
reflections, never stat-salad — which is the point.

**The app (`app.py`, `make app`).** A Streamlit search bar. Picks a narrator (Foundation
Models or local `claude`) or compares BOTH side by side on the same selected nodes — so
the head-to-head "is Foundation Models good enough to ship standalone?" question is
answerable by eye. Evidence shown in an expander. Zero-egress preserved: only the text
notes reach a narrator; Foundation Models is on-device, `claude` is a local process.

**Why this matters for the product.** If Foundation Models reads as well as Claude, the
app ships with ZERO third-party dependency — a fully standalone, local-first, zero-egress
reflection tool, no Claude/OpenAI/key required. That is the strongest possible form of the
V2 promise. `streamlit` is an optional extra ('app'); the core engine and MCP stay lean.
296 tests, lint clean, publish-check clean.

---
## M. The constants must not be filtered out of a change story (wife-missing bug)

The owner caught a damning gap: his WIFE (second-most-photographed, 661 photos, married
since 2008) appeared in NEITHER narrator's two-decade answer, while friends and in-laws
did. Not a narration problem — a selection bug with a real lesson.

**Root cause (two layers).** (1) Her trajectory classified correctly as 'steady' and the
node WAS created. (2) But `select_nodes` dropped it: a "how did it CHANGE" question scores
change-flavored nodes (arrived/faded/peaked/rising, all tagged 'changed'/'over time')
highest, and the 'steady' nodes — tagged 'constant'/'throughout', with no era and no
'changed' tag — scored so low they never even passed the topical filter into the ranked
list. The system was literally reasoning "you asked what changed, and she didn't change,
so I'll omit her" — dropping the two people who matter most (the owner's wife, and
nearly his partner of record) from a question about the owner's own life.

**The fix (principled).** A life-change story is told AGAINST its constants — "who
changed" is only meaningful relative to "who stayed". So on a BROAD question, 'steady'
relationship nodes are now made topical and given a gentle boost, guaranteeing the
anchor people survive among the change nodes. Verified: the two-decade selection now
includes both steady anchors, and the narrated answer names the wife (a "[partner] and
[spouse] remain steady, returning like familiar breaths"-style line). Regression test
added. 297 tests, lint clean.

**Lesson for the graph/selection design.** Relevance-by-keyword-overlap has a blind spot:
the most important signal in a "change" story is often the thing that did NOT change.
Selection must reserve room for the stable backdrop, or it amplifies novelty and erases
permanence — exactly backwards for an emotionally intelligent memory tool.

---
## N. Weighting: minor faces must not read with spouse-weight

After the steady-anchor fix, the owner noticed a friend's spouse (a minor recurring face,
~73 photos, peaking around 5% of an era) was getting "weird weight" — a full "a bond
deepening over the years" sentence, the same prominence as people who are actually
central. Two compounding causes:
  1. A trajectory shape fired for anyone who recurred, so a 2%->5% blip became an
     individual "rising" node — overstating a friend-of-a-friend as a life chapter.
  2. Every node was handed to the narrator as a FLAT, equal "note" with no signal of
     relative weight, so a magnitude-0.04 node got the same airtime as a 0.48 one.

**Fix A — a presence bar for individual sentences.** A person earns their OWN trajectory
sentence only if they reach a real presence (>=15% of an era) or are a dominant 'steady'
anchor. Everyone below the bar is collapsed into ONE "supporting cast" aside per shape
family (newer faces vs faces that thinned out) — acknowledged as real without inflation.
The minor faces went from one inflated sentence each to a single grouped mention.

**Fix B — order notes by weight + an EMPHASIS instruction.** `build_answer_prompt` now
sorts observations by magnitude (strongest first; the 'cast' aside always last) and tells
the narrator the notes are importance-ordered — give the earlier ones center stage, the
later minor ones only a light touch, and never give a friend-of-a-friend the same weight
as a spouse or child. Verified on both narrators: the minor face dropped out of the
two-decade answer entirely, and the spouse/partner read as the anchors.

**Lesson.** Recurrence is not importance. An emotionally intelligent memory tool must
weight a spouse differently from an acquaintance who happens to appear a few times —
both in WHICH nodes earn a voice (the presence bar) and in HOW MUCH voice each gets
(weight-ordering + emphasis). 299 tests, lint + publish-check clean.

---
## O. Tuning the on-device narrator (how close can Foundation Models get to Claude?)

The owner found Claude "way superior" on the two-decade reflection and asked whether
Apple Foundation Models (on-device, ~3B) can be tuned to match. Answer: the MECHANICAL
gap is largely closable by prompt; the INTERPRETIVE-LEAP gap is a model-size ceiling.

**What a small model got wrong (observed).** With the default prompt (written to TRUST a
capable narrator), FM: (1) name-dumped — listed all 9 allowed names in one breath; (2)
over-wrote — "presence hums beneath the surface", "names etched into a growing rhythm";
(3) restated notes instead of building an arc.

**The fix — a model-tier prompt.** `build_answer_prompt(..., tier=)`: "default" trusts a
large model to weave; "compact" (new `_ASK_SYSTEM_COMPACT`) gives a small model the
scaffolding it needs — explicit one-paragraph length, an ARC to follow (how it was ->
how it changed -> what it means), a NAME BUDGET (2-3, only the top-weighted; the rest are
"a few friends"), and concrete anti-patterns to avoid (named the exact purple words).
A subtle small-model bug surfaced and was fixed: the first draft used dash-bullets to
FORMAT the instructions, and FM echoed a leading "- " into its output — so the compact
prompt forbids starting any line with a dash/bullet and uses prose instructions only.
The app sends 'compact' to Foundation Models and 'default' to Claude automatically.

**Result (same question, same nodes).** Tuned FM is now clean and genuinely good: proper
arc, no name-dump (just the partner/spouse/pet), plain sincere language, correct
weighting. But
it ends at REPORTING the arc ("your attention has settled on the people who matter
most"), where Claude MAKES MEANING ("you stopped trying to collect people and started
trying to hold a whole moment of life with them in it"). That interpretive synthesis is
the ~3B-vs-hundreds-of-B gap; no prompt closes it fully.

**Product takeaway.** Tuned FM is "good enough to ship standalone" — a warm, grounded,
zero-dependency/zero-egress reflection. Claude is a premium narrator for those who have
it. The swappable-narrator architecture offers BOTH: FM as the honest default, Claude as
an opt-in upgrade — a stronger story than either alone. 300 tests, lint + publish clean.

---
## P. Narrator scope for V1

V1 supports two narrator paths:

1. **Apple Foundation Models on-device** — the privacy-first default, invoked through
   the local Swift helper.
2. **Claude CLI** — an optional narrator for users who already have the Claude CLI
   installed and authenticated. It receives grounded text notes only; photo pixels never
   leave the Mac through this app.

A larger Apple server-hosted narrator was investigated but is not part of the shipped
code or product surface. Keeping the supported choices small makes setup clearer and
avoids advertising an unavailable entitlement-gated path. The narrator remains swappable:
the insight graph and grounding checks are the product, while Apple on-device and Claude
are two available writing engines.

---
## Q. The content dimension — using Vision's scene labels (a critical early miss)

The owner's sharpest structural catch: every content dead-end we'd patched (landscapes,
night/day) was the engine REVERSE-ENGINEERING subject from technical proxies (EV for
"night", negative-space for "landscape") — when Vision had ALREADY classified the actual
content of every photo. We persist those labels (`scene_labels`, 1,025-label vocabulary
over all 9,639 photos) and used them ONLY to compute the quadrant vector, then discarded
the semantic content. So "do I photograph children / the outdoors / water" dead-ended
even though the answer sat in the DB.

**Fix — a curated SUBJECT dimension.** `_subject_nodes` mines `scene_labels` for the
recurring THINGS the owner shoots. Curated groups (children, nature/outdoor, water,
animals, food, travel, celebration), share-gated (>=12%), whole-archive + per-era, plus a
'dominant' node for "what do I shoot most". Curation is the whole point: raw Vision labels
are full of junk ("structure" 7.6k, "material", "textile" — background stuff, not a
subject), so an uncurated pass would emit "you love architecture" from wall-and-furniture
noise. On the real archive it surfaced the true themes: children (36%) and the outdoors
(38%, the top subject); water/animals/food stayed below the bar (honest — real but not
themes). Graph grew 103 -> 123 nodes.

**A tokenizer bug it exposed.** "what do I photograph most?" is ALL stopwords
(what/do/i/photograph/most — 'photograph'/'shoot' are stopwords so they don't match the
ubiquitous word in every claim), so it tokenized to EMPTY and dead-ended. Added an intent
fallback: when a question has no content tokens but signals the "what do I shoot" intent,
seed the subject vocabulary so the subject nodes match.

**Lesson.** We had the richest signal (Vision's semantic labels) and were using it only as
a feature for a downstream score, never as content the owner could ASK about. Technique is
half the story; WHAT is in the frame is the other half, and it was already computed. A
memory tool must expose the content it already understands, not just the geometry. 307
tests, lint + publish clean.

---
## R. "Which photos do I keep more" — tagging critique nodes + an always-on keep node

The question dead-ended, exposing two bugs:

1. **Critique nodes shipped with EMPTY tags.** `era_style`, `resonance_gap`, and the
   `trend_*` nodes predate the tagging system, so `ask` could never reach them BY TOPIC —
   only by era or the broad path. The resonance_gap node is literally the "which photos
   do I keep / return to" answer, yet was unretrievable. Fixed: all critique nodes now
   carry retrieval tags (the gap node tagged keep/kept/return/drawn/resonance/cherish).

2. **A stale-node bug in the rebuild.** `graph` cleared only the critique kinds PRODUCED
   this run. After resonance tuning, no era's top quadrant FLIPS anymore, so no per-era
   gap node is produced — and the old, empty-tagged gap nodes from a prior build were
   never deleted (their kind wasn't in the produced set). Fixed with a fixed
   `CRITIQUE_KINDS` constant: the rebuild clears ALL known critique kinds, so a kind that
   stops being emitted leaves no stale rows.

3. **The gap concept was too brittle to answer the question.** The flip gap ("you shot X
   most but return to Y") needs the top to differ, which no longer happens here. But
   "what do I keep" is still answerable from what the owner is DRAWN BACK TO. Added an
   always-on whole-archive `_resonance_preference` node: the quadrant with the highest
   pooled resonance share across eras. On the real archive it answers truthfully — the
   kept photos are the Solitary & Introspective ones ("the images that pull you inward...
   you return to it again and again"), the intent doc's core thesis (what you return to
   != what you shot) finally surfacing on demand.

**Lesson.** Two systemic issues: (a) a whole node family was unretrievable because it
predated tagging — any new retrieval dimension must backfill tags on ALL node producers,
not just the newest; (b) a rebuild that clears only "kinds produced this run" silently
keeps stale rows of a kind that stopped being produced — clear by a fixed known set.
308 tests, lint + publish clean.

---
## S. Cross-dimension lift: "what do I shoot at golden hour?"

The owner's test question was wrong on both narrators. The real answer is that golden hour
pulls toward the outdoors and water, and that the pet and the partner recur in golden
frames. The graph had no node that crosses time-of-day with subject or person, so neither
narrator had anything to say. Both models failed because the graph was missing the fact.

**Data check (before building).** Golden hour is rare (167 of 9,639 photos, 1.7%). Against
the archive, nature is over-indexed in golden frames (49% vs 37%), water too (14% vs 8%),
and two people recur in golden frames far above their share of the archive.

**Fix.** A `conjunction` kind (`_golden_conjunction_nodes`) emits a node when a subject or
person is over-represented in golden frames: a lift of at least 1.3x for subjects and 1.5x
for people, with at least 10 golden frames and at least 30 golden frames overall. The
owner is never a golden-person node.

Two retrieval bugs surfaced and were fixed:
- The narrow-question dedupe collapsed every conjunction node into one, because they share
  kind and direction. Conjunction nodes now dedupe on their own id.
- Subject vocabulary was on the shared conjunction tags, so person nodes matched a
  subject question. Subject vocabulary now sits only on subject-family nodes.

**Known limit (not fixed).** Golden hour is computed from GPS plus timestamp. The prototype
cache has 0% GPS for the 2000s and 27% for the 2010s, so the engine cannot establish golden
hour for early photographs. The owner's recollection of early golden-hour landscapes is
therefore not checkable from the data. Options, to decide later: an owner-supplied home or
location fallback, or a rough local-clock heuristic (approximate, lower confidence).

**Also noted.** The engine does not know that one recurring person is a pet, so the
narration calls it a person. Recording roles (pet, family, friend) needs owner-supplied
config. Deferred.

312 tests, lint and publish-check clean.

---
## T. Golden hour needs location: per-era nodes gated on GPS coverage

Golden hour is computed from sun position, which needs GPS. The database stored
`golden_hour=0` for photos without GPS, which reads as "not golden" when it means
"unknown". Two consequences, both corrected here:

1. **A false daylight claim.** The 2000s "mostly ordinary daylight" node was built from
   photos with no location data at all (1 of 1,919 has GPS). It is removed. A period with
   no location data now produces no golden or daylight claim.

2. **A false growth trend.** Golden share among ALL photos appeared to rise (1.1%, 2.2%,
   3.4%). Among photos WITH location it is flat: 4.3%, 4.7%, 4.5% (early 2010s, late 2010s,
   2020s). The rise was GPS coverage growing (25% -> 47% -> 75%), not shooting behaviour.

**Change.** `has_gps` is now stored (migration + pipeline), backfilled from source
metadata (HDD: prototype records; Library: osxphotos latitude, metadata only, no masters)
via `memory-graph backfill-gps`. Golden nodes read the golden share on located photos only,
and need at least 30 located photos in a period. A share of at least 8% is a "reach for
golden hour" node; 3% to 8% is a "small but steady thread" node; below 3% is a daylight
node. Claims carry no numbers; magnitude holds the share.

**Conjunction lifts recomputed on the located population.** The earlier nature lift
(1.33x) was inflated: nature is more common among located photos, and the old comparison
used all photos. On the honest base, nature is 1.14x (below the 1.3x bar, so it no longer
emits). Robust golden-hour signals now: water 1.55x, and the partner 1.57x. Food (1.67x) and
celebration (1.32x) are real per the lift but rest on very few golden frames, so they are
noise-prone and should be read with caution.

**Known limit.** Early photos have no GPS, so the engine cannot establish golden hour
before the 2010s. Options (not built): owner-supplied home location fallback, or a local
clock heuristic (approximate, lower confidence). Owner decision.

312+ tests, lint and publish-check clean.

---
## U. Holdout pre-registration (frozen before any photo is selected or labelled)

**Question.** Does the daily memory ranking choose photos the owner would want to see,
on photos the engine was never tuned on?

**Holdout pool.** Every canonical photo not in the tuning label set (`label_set.json`), not in
any label store (`base_rate`, `quadrant`, `pairwise`), and never dispatched before
(`last_dispatched` is null). The pool size is recorded when the set is built.

**Design (100 labels, blinded).**
- 25 simulated held-out days. Each day's top 3 come from the engine's own dispatch
  selector (`select_dispatch`), run on the remaining pool with a fixed reference time.
  Each photo is used at most once across days (repeat-avoidance is off across simulated days,
  and no photo appears twice). That gives 75 dispatch items.
- 25 control photos drawn uniformly at random from the pool, excluding all dispatch items.
- Items are shuffled into one list with a fixed seed (20261026). The labeller is shown only the
  photo. Dispatch and control membership are hidden.

**Labels per item.** (1) "Would you want the app to send you THIS, unprompted, as your one
memory today?" yes/no, the same wording as the tuning base-rate task. (2) Felt quadrant
(1 to 5), used for the secondary quadrant-agreement report.

**Primary metric.** `lift = P(worthy | top-3) - P(worthy | control)`.

**Pass rule, fixed now.**
- PASS: lift is at least 25 points AND the 95% lower bound of the lift is above zero.
- INCONCLUSIVE: lift is at least 10 and below 25 points, OR lift is at least 25 points but the
  lower bound is at or below zero. Collect more labels from the unused pool. Do not change weights.
- FAIL: lift is below 10 points, including negative. The ranking is not meaningfully better than
  chance on unseen photos. Revisit the resonance weights. This holdout is then spent.

Statistic: difference of two proportions, normal approximation,
`SE = sqrt(p1(1-p1)/75 + p2(1-p2)/25)`, 95% bound = lift +/- 1.96 SE.

**Secondary metric (report only, no gate).** Quadrant agreement, felt quadrant equals
the engine's dominant quadrant, over all 100 items, with a Wilson 95% interval. Reference:
the earlier EXIF-only router agreed with felt labels about 25% of the time.

**What this round does not test.** Insight narratives and node selection are not graded here.
They are judged by reading, and need a separate rubric.

**Discipline.** Once labels are collected, the pass or fail call is made by the rule above,
not by inspecting which photos scored well. Any change to weights after seeing results means the
next test uses a new holdout sample from the unused pool.

**Status (set built, not yet labelled).** The holdout was built with `memory-graph
holdout-build` after this pre-registration was written. Pool: 9,567 untouched photos (the
exclusions were the 500 tuning and labelled ids and any previously dispatched photo). Items:
75 dispatch (25 simulated days x top 3) and 25 controls, shuffled with seed 20261026. Every
item's preview was rendered and checked (100/100). The set file lives in the gitignored
`data/labels/holdout_set.json` and cannot be rebuilt without `force`. Labelling is
blinded and resumable via `memory-graph holdout-label`. Score via `memory-graph holdout-score`
once all 100 are labelled. Decision under the rule above, with no changes before the call.

**Amendment (recorded before any holdout label exists).** The terminal labeller showed the
photo in Preview while the question stayed in a Terminal window, so the owner could not see the
question and the photo together. The labelling is now a local web page (`memory-graph
holdout-label`, 127.0.0.1 only) that shows the photo, the question, Yes/No and the five
feelings on one screen. Wording changed for plain language. The meaning is the same as the
tuning question:
- Old: "Would you want the app to send you THIS, unprompted, as your one memory today?"
- New: "Imagine this photo arrives on your phone today as your one memory. Would you be glad
  to see it?"
- The five feelings now carry one-line plain descriptions. The quadrant names are unchanged.
The pass rule, the design, the seed and the pool are unchanged. Storage is the same file
(`data/labels/holdout.jsonl`), so the scorer is unchanged.

**Result (round 1, scored once, all 100 labelled, rule applied unchanged).**
- P(worthy | engine top-3) = 43% (32 of 75).
- P(worthy | random control) = 24% (6 of 25).
- Lift = +19 points, 95% bound -1 to +39.
- **Decision: INCONCLUSIVE.** The lift is above the 10-point floor but below the 25-point bar,
  and the lower bound touches zero. The rule says: do not change weights; collect more labels
  from the unused pool.
- Control rate (24%) matches the 23% baseline from the earlier strict relabel, which suggests the
  labelling is calibrated.
- Quadrant agreement (report only): 22 of 100 = 22% (95% 15% to 31%). This is close to the
  earlier EXIF-only router (about 25%). The felt-feeling categories do not match the engine's
  dominant quadrant well. That is a finding about the quadrant scheme, not a gate.
- No weights were changed after seeing this result.

---
## V. Holdout round 2 pre-registration (written before the round-2 sample is drawn)

**Why a second round.** Round 1 was inconclusive: lift +19 points, bound -1 to +39, against a 25-point bar.
The round-1 lower bound touched zero, so more evidence is needed. Round 1 was scored once, and the result is
recorded in Part U. Round 2 is pooled with round 1 under the rule below.

**Changes from round 1 (decided by the owner).**
1. **The yes/no question stays the only gate.** Answering No saves at once and shows the next photo.
   The feeling step is asked only after Yes. The owner's argument holds: a feeling for a memory you
   don't want back doesn't bear on the decision.
2. **The quadrant check uses yes answers only.** It is report-only, so it does not gate the decision.
   In round 1 the feeling was recorded for every item; round 1 feelings on No answers are excluded from
   the quadrant report, so both rounds are analysed the same way.

**Sample.** Same design as round 1: 25 simulated days, top 3 from the engine's dispatch selector, and
25 random controls (75 + 25). Seed **20261201**. The pool excludes every photo in the tuning label set,
every tuning label store, every previous holdout set, and every previous holdout answer file. It also
excludes any photo already dispatched (`last_dispatched` set) and any private photo.

**Decision rule (pooled, fixed now).** The decision uses the pooled yes/no result from rounds 1 and 2
together: 150 top-3 items and 50 controls. Thresholds are unchanged:
- PASS: lift at least 25 points AND 95% lower bound above zero.
- INCONCLUSIVE: lift at least 10 and below 25, OR lift at least 25 with lower bound at or below zero.
- FAIL: lift below 10 points.
Round 2 on its own is reported for information and does not decide.

**Honest limit.** Round 1 was already seen before this pre-registration, so the pooled test is not a
fully fresh test. It is a pre-registered rule applied to more data, and the report will say so.

**Report only.** Quadrant agreement on yes-answered items, pooled across rounds, with a Wilson 95%
interval. Per-round figures are also shown.

**Unchanged.** Wording of the questions, the pass rule thresholds, the design, the seed for round 1.

**Status (round 2 drawn, not yet labelled).** Drawn with seed 20261201 after Part V was written.
Pool 9,467 untouched photos. Items: 75 dispatch and 25 controls, 100 distinct, with zero overlap
with round 1. All 100 previews rendered and checked. Files: `data/labels/holdout_set_r2.json`
(items) and `data/labels/holdout_r2.jsonl` (answers, created on the first answer). Labelling:
`memory-graph holdout-label --round 2`. Scoring: `memory-graph holdout-score` pools every round
with labels, and applies the Part V rule unchanged.

**Result (round 2, pooled with round 1, scored once under the Part V rule).**
- Pooled P(worthy | engine top-3) = 41% (62 of 150). Pooled P(worthy | random control) = 18% (9 of 50).
- Pooled lift = +23 points, 95% bound +10 to +37.
- **Decision: INCONCLUSIVE.** The lower bound is above zero, but the lift is 2 points short of the 25-point bar.
- Round 2 on its own (reported for information only, not used for the decision): top-3 30 of 75 (40%),
  controls 3 of 25 (12%). Its figures are noisier, since 25 controls give a wide range.
- Quadrant agreement, yes answers only (report only): 12 of 71 = 17% (95% 10% to 27%). This is lower than
  round 1 and points to the feeling scheme, not to the ranking.
- No weights were changed after seeing this result.

**Finding (single-choice feeling, rounds 1 and 2, 71 yes answers with a feeling).** The single-feeling
match rate (17%) is below the rate expected if the engine's mood calls and the owner's feelings were
unrelated (24%). Felt feelings: Nostalgic 39, Close 26, Excited 5, Quiet 1. Engine dominant quadrant on
the same photos: Quiet 29, Nostalgic 20, Close 14, Excited 8. The engine's "Quiet" calls are mostly felt as
Nostalgic or Close, and the owner almost never feels Quiet on a photo he'd want back. This also contradicts the
resonance-preference (keep) insight, which says kept photos are Solitary and Introspective. That insight
is built from engine scores, not the owner's felt labels. Next test: multiple feelings, scored against
chance (see the next pre-registration before it is drawn).

---
## W. Holdout round 3 pre-registration: multiple feelings (written before the round-3 sample is drawn)

**Why.** The single-feeling test forced one answer per photo. Photos carry several feelings, and the
engine scores all five feelings as a soft vector. Round 1 and round 2 single-choice agreement (17%)
was below the chance rate (24%). Round 3 tests whether letting the owner pick several feelings gives a
fair read of the engine's mood scores.

**Decisions (agreed by the owner).**
1. **Round 3 is the final yes/no round.** The yes/no decision is the pooled result across rounds 1, 2 and
   3. No round 4 is drawn for the yes/no question, whatever the result.
2. **Feelings are asked for every photo, Yes or No.** This test is about how the engine reads mood, not
   about the daily memory decision, so the feelings are not gated on Yes.

**Sample.** Same design as earlier rounds: 25 simulated days x top 3, plus 25 random controls (75 + 25).
Seed **20261215**. The pool excludes everything excluded in Part V, plus every round 1 and round 2 item and
answer.

**Yes/no decision (pooled, rounds 1 to 3).** Unchanged rule from Parts U and V:
- PASS: lift at least 25 points AND 95% lower bound above zero.
- INCONCLUSIVE: lift at least 10 and below 25, OR lift at least 25 with lower bound at or below zero.
- FAIL: lift below 10 points.

**Feelings test (round 3 only).** For each photo, the owner picks one or more feelings from the five. Let
F be the set picked. The engine's dominant quadrant is a hit when it is in F.
- Observed hit rate H = share of hits over the round-3 items.
- Chance rate E: for each item, the sum over q in F of p(q), where p(q) is the share of round-3 items whose
  engine dominant quadrant is q. The chance rate is the mean over items.
- Lift = H minus E. Standard error = sqrt(sum of E_i(1 - E_i)) / n. 95% lower bound = lift minus 1.96 SE.
- **Informative** if the lift is at least 10 points AND the lower bound is above zero. Otherwise **not
  informative**. This test does not change the yes/no decision.
- Guard against padding: report the average number of feelings picked, and the share of photos where all
  five are picked. If the average is 4 or more, the test is flagged as uninformative, because a broad pick
  would match by chance.

**Context, reported for information.** Rounds 1 and 2 single-choice agreement (17%) against its chance
rate (24%). Not a decision input.

**Honest limits.** Round 1 was already seen, and round 3 pools with it, so the yes/no result is not a
fully fresh test. The feelings test is a fresh test on its own sample. The owner's feelings are a single
rater's view.

**Status (round 3 drawn, not yet labelled).** Drawn with seed 20261215 after Part W was written. Pool
9,367 untouched photos. Items: 75 dispatch and 25 controls, 100 distinct, zero overlap with rounds 1 and 2.
All 100 previews rendered and checked. Files: `data/labels/holdout_set_r3.json` and, once answered,
`data/labels/holdout_r3.jsonl`. Labelling: `memory-graph holdout-label --round 3` (yes or no, then every
feeling that applies). Scoring: `memory-graph holdout-score` reports the pooled yes/no decision across
rounds 1 to 3, the single-feeling context, and the Part W feelings test for round 3. 341 tests pass.

**Result (round 3 scored, seed 20261215, 100 of 100 answered).** Scored with `memory-graph holdout-score`, no weights changed.
- Pooled yes/no, rounds 1 to 3: top 3 is 40% (89/225); random control is 19% (14/75). Lift +21 points, 95% lower bound +10.
- Decision (pre-registered rule): **INCONCLUSIVE**. The lift is below the 25-point bar, so this is not a PASS. It is above the 10-point floor, so it is not a FAIL.
- Round 3 alone, derived from the pooled totals and the earlier rounds: top 3 is 27/75 (36%), control is 5/25 (20%).
- Feelings test (Part W): engine's top feeling is among the owner's picks on 29% of photos. Chance given the picks is 26%. Lift +3 points (95% bound -5 to +11). **NOT INFORMATIVE.**
- Feelings use: average 1.3 feelings per photo. Padding flag does not trigger (threshold is 4 or more).
- Single-feeling context (rounds 1 and 2) unchanged: 17%, against 24% chance.
- Reading: the ranking is above the control, but not by the 25 points pre-registered. The engine's mood calls do not match the owner's feelings any better than chance.
- Next: owner decides whether to reword the five feeling names (step 3 of Next). Any change needs a new pre-registration and a fresh sample.
