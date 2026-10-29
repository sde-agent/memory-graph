# Stage 3 — Camera Focal-Length Backfill

## Status

Approved and narrowed. Stage 3 adds exactly one capability on top of the Stage 2
release (`17d8c25`): a camera crop-factor backfill step that runs before the existing
analysis. It does not add a setup wizard, a review surface, a second analysis
orchestrator, or a new state machine. The branch is `stage3/setup-safe-first-analysis`.

## Why this is the whole scope

The existing README already documents the complete first run:

```bash
make setup
make init
cp config.local.toml.example config.local.toml   # edit personal values
memory-graph sample-run --all
memory-graph graph
```

That path already handles Full Disk Access, optional HDD analysis, progress, resume by
stable photo ID, duplicate avoidance, Hidden-album privacy, and utility-image
filtering. None of that needs to be rebuilt.

The one real gap is optics. Many older cameras (all Canon DSLRs/compacts, most
point-and-shoots, early phones) record only the raw lens focal, never the
35mm-equivalent. Left uncorrected, every such frame is misclassified as ultra-wide,
which corrupts the framing insight and skews the Q1/Q2/Q3 quadrant vector. A user
currently has to discover this and hand-edit `[framing.crop]`. Stage 3 automates the
discovery and proposal while keeping the write explicit.

## User-visible behavior

A new command sits between configuration and analysis:

```bash
memory-graph camera-scan                 # list models missing a 35mm equivalent
memory-graph camera-scan --confirm       # write the resolvable crop factors
memory-graph sample-run --all            # existing analysis, now with correct optics
memory-graph graph
```

`camera-scan`:

- reads Photos Library metadata (and `--hdd-root` if given) through the existing
  guarded adapters — never a master, never pixels;
- lists each camera model that lacks a 35mm-equivalent focal, with its raw focal
  range and affected photo count;
- proposes a crop factor per model from the shipped lookup or from a consistent
  raw/equivalent derivation;
- reports models it cannot resolve instead of guessing;
- writes nothing unless `--confirm` is passed;
- on `--confirm`, merges the resolvable factors into the existing `[framing.crop]`
  table in the gitignored `config.local.toml` and tells the user what to run next.

Options:

- `--hdd-root PATH` — also scan a read-only external archive.
- `--dbfile PATH` — use a non-default Photos database.
- `--confirm` — write the resolvable factors.

After confirming, a user with an existing graph can run `memory-graph rescore` to
correct the quadrant vector from stored signals without a Vision re-decode; a new user
just runs `sample-run --all`.

## Guardrails

1. The Photos adapter reads metadata and `path_derivatives` only; it never touches a
   master. The existing `assert_no_master_access` guard still applies.
2. HDD scanning uses the existing read-only mount check.
3. The scan releases each thumbnail immediately; only lightweight EXIF/metadata is kept.
4. Crop factors are written only on explicit `--confirm`.
5. Full-frame bodies resolve to `1.0`; no model is silently scaled by a fallback.
6. Unresolved models are reported and left for a manual factor, never guessed.
7. Writes go only to the gitignored local overlay, are validated through `load_config`,
   are atomic, and preserve unrelated config tables.
8. No analysis, narration, network egress, or source-file mutation happens in this
   command.
9. All Stage 2 behavior and its regression suite remain intact.

## Non-goals

- A setup wizard or guided first-run flow.
- A privacy, clutter, or photo review surface (privacy and clutter stay automatic in
  the existing pipeline).
- A second resumable-analysis orchestrator or new setup state machine.
- Owner-identity onboarding (owner self-names remain a documented config edit).
- Relationship inference, per-person weights, or taste retraining.
- A new face-embedding provider or identity calibration.
- Any change to historical insight claims or unweighted distributions.

## Implementation

Package `memory_graph/setup/`:

```text
__init__.py     # exports inventory_cameras, propose_crop_factors, update_crop_factors
models.py       # CameraProposal dataclass only
cameras.py      # pure inventory + lookup/derived/unresolved crop proposals
writer.py       # validated atomic [framing.crop] write to the local overlay
```

CLI `memory_graph/cli.py`:

- `camera-scan` subcommand and `_cmd_camera_scan` handler;
- a small `_scan_camera_metadata` helper over the existing guarded streams.

No other Stage 2 module changes.

## Crop-factor proposal rules

- Existing local `[framing.crop]` entries win over the shipped lookup.
- The shipped lookup contains common bodies, with full-frame bodies pinned to `1.0`.
- A derived factor is accepted only when per-record raw/equivalent ratios agree
  closely (within 5%); otherwise the model is unresolved.
- Models with neither a lookup nor a consistent derivation are reported unresolved.

## Evaluation and acceptance

### Unit tests

- `tests/test_setup_cameras.py` — lookup, derived, unresolved, and existing-equivalent
  proposal paths; full-frame `1.0`.
- `tests/test_setup_writer.py` — atomic write, preservation of unrelated tables,
  overlay creation, and invalid-factor rejection without replacing the file.

### CLI tests

- `tests/test_cli_camera_scan.py` — proposals without `--confirm` write nothing;
  `--confirm` writes the resolvable factor; unresolved-only `--confirm` writes nothing
  and signals the gap; a missing Photos database is reported clearly; thumbnails are
  released.

### Regression gates

Before release:

- all Stage 2 tests still pass;
- new Stage 3 tests pass;
- Ruff passes;
- publish check passes;
- zero-egress tests pass;
- no personal data enters tracked files or Git history.

## Decision record

- Stage 2 remains the stable downloadable baseline and its commands are unchanged.
- Stage 3 is only the camera focal-length backfill plus its explicit write step.
- Privacy and clutter remain automatic pipeline behavior, not a review surface.
- Crop-factor confirmation is required before optics-based analysis is trustworthy for
  affected camera models.
