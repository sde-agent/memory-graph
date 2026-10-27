# Memory Graph Engine — V2 Product Brief

## Product goal

Make the local-first Photographic Memory Graph usable by another person on their own Mac, while allowing each person to tune what deserves resurfacing without changing the historical record of what they photographed.

## V1 baseline

V1 proves the core loop on a real archive:

- ingest local HDD and Photos Library derivatives;
- reject utility images and collapse bursts;
- measure optics, light, geometry, scenes, faces, identity, and recurrence;
- score resonance;
- generate a grounded insight graph;
- answer reflective questions through local narrators;
- keep pixels local.

Current V1 snapshot: 55,771 canonical photos, 192 insight nodes, 52,114 face observations, 871 anonymous recurring-face clusters, and 357 passing tests.

## V2 priorities

### 1. Personal tuning controls

Add a local tuning UI backed by `config.local.toml`:

- quadrant weights Q1–Q5;
- age and warmth behavior;
- per-person weights;
- person hard exclusions;
- camera/model weights;
- subject weights;
- place/travel weights;
- technical preferences for bokeh, sharpness, negative space, candidness, and motion;
- reset-to-defaults and preview-before-save.

Hard exclusions and soft de-emphasis must be separate concepts. A blocked person should not be dispatched or narrated, while a de-emphasized person may still appear in a direct historical question.

Tuning affects dispatch/resonance only. It must not silently rewrite unweighted historical distributions or delete insight evidence.

### 2. Friend-ready setup

Create a first-run flow that:

- detects Python/macOS/Full Disk Access/exiftool/osxphotos prerequisites;
- discovers Photos Library and external volumes;
- explains the derivative-only/no-master rule;
- identifies the archive owner among Apple Photos faces;
- discovers camera models missing 35mm-equivalent focal lengths;
- guides crop-factor confirmation;
- lets the user configure life eras;
- reviews hidden/private albums;
- reviews clutter candidates;
- creates a local data directory and sidecars;
- builds the named identity gallery;
- runs anonymous recurrence clustering;
- generates the first graph.

### 3. Safer identity

The current generic Vision feature-print matcher is experimental. V2 should add:

- a labeled calibration workflow;
- confidence and margin reporting;
- per-person precision diagnostics;
- a better face-specific on-device embedding provider if available;
- explicit “unknown” behavior below the calibrated threshold;
- anonymous clusters that remain nameless unless the user confirms them.

### 4. Better context

Complete the context layer with:

- lens-model insights;
- camera-by-era and camera-by-subject comparisons;
- GPS-region clustering without cloud reverse geocoding;
- optional user-supplied place names;
- season/month and life-event chapters;
- travel/home/relocation eras.

### 5. Distribution and packaging

V2 should provide:

- a clean install path;
- a setup wizard;
- a documented local-only default;
- an Apple Foundation Models narrator package;
- optional Claude/local narrator adapters;
- resumable ingest and identity commands;
- database/sidecar size diagnostics;
- backup and reset commands;
- a publish-safe repository with no personal data.

## Non-goals for V2

- hosted photo analysis by default;
- sending image pixels to an LLM;
- universal social graph inference;
- pretending anonymous clusters are named people;
- deleting photographs automatically;
- replacing historical truth with learned preference.

## Success criteria

A friend should be able to clone the repository, configure their own local sources, complete setup without editing Python, understand what the system knows and does not know, tune resurfacing behavior, and receive grounded answers without any photo pixels leaving their Mac.
