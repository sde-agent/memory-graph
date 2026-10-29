# Memory Graph Engine

A local-first photographic memory companion for macOS. It is not an asset manager or generic photo search tool. It analyzes the technical and human patterns in a personal archive, builds a grounded insight graph, and helps answer questions such as:

- “What was my style in the 2010s?”
- “How do my camera choices affect the people I photograph?”
- “Do I prefer bokeh or deep focus?”
- “Which people and moments keep recurring?”

Read the full story in [CASE_STUDY.md](CASE_STUDY.md) and the next-product plan in [V2_PRODUCT_BRIEF.md](V2_PRODUCT_BRIEF.md).

## Stage 2 release status

This checkout is the stable Stage 2 baseline. It is a downloadable local research product, not a hosted service or a universal psychology model.

- Apple Photos metadata and local derivatives, plus optional external-HDD ingest;
- grounded insight graph across technical, emotional, era, subject, and identity dimensions;
- resonance dispatch, personal Favorite-trained taste scoring, local narrators, MCP, and Streamlit app;
- 358 automated tests and clean Ruff lint at the release checkpoint.

The archive counts in the case study are from the author's private run and are not expected for a new installation. Generic Vision face embeddings remain experimental, place names are not reverse-geocoded, and social relationship inference plus richer per-person weighting/exclusion policy are deferred to Stage 3. Current exclusions are local dispatch filters; they do not rewrite historical facts or train a relationship graph.

## Privacy model

- Photo pixels never leave the Mac.
- Photos Library masters are never opened; only local derivatives are used.
- HDD photos are read from the external source or local derivative cache.
- Full-resolution photos are never copied into the repository or internal storage.
- Hidden/private photos are excluded from dispatch, identity gallery construction, and anonymous clustering.
- Narrators receive grounded text, never image bytes.

## Requirements

- macOS with Apple Vision available;
- Python 3.11+;
- Full Disk Access for the terminal when reading Photos Library metadata;
- `exiftool` on PATH for external-HDD EXIF;
- Apple Foundation Models support for the on-device narrator path;
- optional local `claude` CLI for development comparison.

## Quick start

Run these steps once, in order. Each step links to a section below with details and
troubleshooting. Steps 5 and 6 are optional.

```bash
# 1. Install (one time)
git clone https://github.com/sde-agent/memory-graph.git
cd memory-graph
make setup
source .venv/bin/activate
make init
memory-graph doctor

# 2. Configure your personal values (one time)
cp config.local.toml.example config.local.toml
#    edit config.local.toml — see "2. Configure" below

# 3. Fix camera crop factors (recommended, before analysis)
memory-graph camera-scan
memory-graph camera-scan --confirm

# 4. Analyze your Photos Library and build the graph
memory-graph sample-run --all
memory-graph graph

# 5. (Optional) add an external HDD, then rebuild the graph
memory-graph analyze-hdd "/Volumes/YourDrive/Photos"
memory-graph graph

# 6. (Optional) resolve recurring people, then rebuild the graph
memory-graph reidentify --recognition-min-similarity 0.95
memory-graph reidentify --cluster-only --anonymous-threshold 0.90
memory-graph graph

# 7. Explore
memory-graph dispatch -n 3
memory-graph ask "What was my style in the 2010s?"
```

Rule of thumb: run `memory-graph graph` once at the end, and again any time you add a
source (steps 5 and 6) or change crop factors. You do not need to repeat `sample-run`
unless you add new photos — if you only changed crop factors, use `memory-graph rescore`
instead.

The rest of this document explains each step.

## 1. Install

From a fresh clone:

```bash
git clone https://github.com/sde-agent/memory-graph.git
cd memory-graph
make setup
source .venv/bin/activate
make init
memory-graph doctor
```

`make setup` is the only networked step; it creates `.venv` and installs the package and
dev dependencies. It cannot activate the environment in your parent shell, so either
`source .venv/bin/activate` (as above) or prefix every command with `.venv/bin/`:

```bash
.venv/bin/memory-graph doctor
```

If you later pull new commits into an existing checkout, refresh the editable install:

```bash
git pull
make refresh
```

## 2. Configure your archive

Copy the personal overlay and edit it:

```bash
cp config.local.toml.example config.local.toml
```

Put personal values only in `config.local.toml`, which is gitignored:

- your Apple Photos face name under `[owner].self_names`;
- life-era ranges under `[eras].ranges`;
- Hidden albums under `[privacy].hidden_albums`;
- camera crop factors under `[framing.crop]` — you usually do not edit these by hand;
  step 3 fills them in for you.

## 3. Fix camera crop factors (recommended)

Many older cameras (Canon DSLRs and compacts, point-and-shoots, early phones) record
only the raw lens focal length, not the 35mm equivalent. Uncorrected, every such photo
is misread as ultra-wide, which skews the framing insight and the quadrant/resonance
scores. Do this before the first analysis.

```bash
memory-graph camera-scan              # list camera models missing a 35mm equivalent
memory-graph camera-scan --confirm    # write the resolvable factors to config.local.toml
```

The scan reads metadata only (no masters, no pixels) and finishes in seconds. Models it
cannot resolve are reported, not guessed; add those by hand under `[framing.crop]`. To
include an external archive, add `--hdd-root "/Volumes/YourDrive/Photos"`, or scan only
the drive with `--hdd-only --hdd-root "/Volumes/YourDrive/Photos"`.

If you change crop factors *after* already analyzing, you do not need to re-run
`sample-run`; just correct the existing graph in place:

```bash
memory-graph rescore
memory-graph graph
```

## 4. Analyze and build the graph

This is the main run. It uses the macOS Photos Library and its local derivatives:

```bash
memory-graph sample-run --all
memory-graph graph
```

`sample-run --all` scans the full Photos Library metadata pool, deduplicates bursts and
cross-source duplicates, rejects utility images, and analyzes every surviving canonical
photo. `--all` means all canonical survivors, not one row per burst member or duplicate.
`graph` then builds the insight graph the questions below draw on.

The run requires Full Disk Access for your terminal (see troubleshooting). It prints
progress for pool scanning, deduplication, optics, Vision analysis, and graph
generation, and reports the Library total, rejections, canonical count, and final
database count. A full library can take a while; if interrupted, just repeat the
command — the analysis is resumable and already-analyzed photos are skipped.

If the Photos Library is in a non-default location, point at its database:

```bash
memory-graph sample-run --all --dbfile "/path/to/photos.sqlite"
```

### Troubleshooting: Full Disk Access

If you see `FileNotFoundError: Could not get path to photo library database`:

1. Open the **Photos** app and confirm the library opens normally.
2. In Photos, use **Photos → Settings → General → Use as System Photo Library** if that button is available.
3. Open **System Settings → Privacy & Security → Full Disk Access**.
4. Click **+**, authenticate, and add the application that launches the command:
   - Apple Terminal: `/Applications/Utilities/Terminal.app`
   - iTerm: `/Applications/iTerm.app`
   - Kiro: `/Applications/Kiro.app`
   - VS Code: `/Applications/Visual Studio Code.app`
5. Quit and reopen that application, then test:

```bash
memory-graph ingest-library --limit 1
```

If you run commands in an IDE's integrated terminal, add the IDE itself, not only Apple
Terminal. An empty Full Disk Access list is normal — click **+** to add the app manually.
To find a non-default library database:

```bash
find "$HOME/Pictures" -name photos.sqlite -print
```

## 5. (Optional) add an external HDD

The HDD is not required for Apple Photos-only users. If you have an external archive,
analyze it and rebuild the graph:

```bash
memory-graph analyze-hdd "/Volumes/YourDrive/Photos"
memory-graph graph
```

If you maintain the optimized archive-cache, set `MEMORY_GRAPH_ARCHIVE_CACHE` before
`sample-run`. Without it (and without an HDD cache) the runner simply uses the Photos
Library source.

## 6. (Optional) identity and recurring people

Resolve named and recurring faces, then rebuild the graph:

```bash
memory-graph reidentify --recognition-min-similarity 0.95
memory-graph reidentify --cluster-only --anonymous-threshold 0.90
memory-graph graph
```

The identity pass:

- uses Apple Photos names as authoritative for Library photos;
- attempts conservative HDD matching through exact duplicate bridges and experimental local face vectors;
- clusters recurring unnamed faces without assigning invented names;
- stores vectors in ignored local sidecars, not SQLite image blobs.

## 7. Explore the graph

```bash
memory-graph dispatch -n 3
memory-graph ask "What was my style in the 2010s?"
memory-graph ask "How do my camera choices affect the people I photograph?"
memory-graph critique
```

## Reflection app

The local Streamlit app compares available narrators using the same grounded evidence:

```bash
make app
```

Then open `http://127.0.0.1:8501`.

The app supports questions about style, people, eras, cameras, lenses, depth of field, landscapes, light, and memory patterns. Expand the Evidence section when evaluating an answer: if the evidence is wrong, the issue is retrieval or graph generation; if the evidence is right but the prose is weak, the issue is the narrator.

### Narrators

The app supports two narrator choices:

- **Apple Foundation Models (on-device):** the default local narrator on supported macOS versions. No Claude account or API key is required.
- **Claude (local CLI):** optional. If the `claude` executable is installed and authenticated on your Mac, `make app` detects it and adds it to the narrator picker. The app can compare Apple and Claude on the same grounded evidence.

To use Claude, verify it is available before starting the app:

```bash
command -v claude
claude --version
make app
```

If Claude does not appear in the narrator picker:

```bash
command -v claude
claude --version
```

If the first command prints a path, launch the app from the same shell:

```bash
export MEMORY_GRAPH_CLAUDE_BIN="$(command -v claude)"
make app
```

If it prints nothing, the Claude CLI is not installed or is not on `PATH`. If Claude was installed while Streamlit was already running, stop and restart `make app`; narrator backends are cached for the app session. The app now shows this diagnostic message when Claude is not detected.

If Claude is installed at a known custom path, you can also use:

```bash
export MEMORY_GRAPH_CLAUDE_BIN="/path/to/claude"
make app
```

Claude is not required for the engine. The app sends only grounded text notes to the selected narrator; photo pixels are never sent through the Memory Graph engine. Claude may use its own configured service connection, while Apple Foundation Models stays on-device.

## Resonance weights

The shipped `config.toml` uses neutral defaults so a new install starts without any
personal tuning baked in:

| Signal | Shipped default |
|---|---:|
| Q1 Nostalgic & Reflective | 1.0 |
| Q2 Visceral & High-Arousal | 1.0 |
| Q3 Empathetic & Connected | 1.0 |
| Q4 Solitary & Introspective | 1.0 |
| Q5 Provocative & Dynamic | 1.0 |
| Known named person | 1.0× |
| Any detected face | 1.0× |
| Favorite | 1.0× |
| Taste-model influence | 0.0 (off until you train one) |

These are deliberately flat. Personal weights are archive-specific and are never
committed; tune them for yourself in the app (below) and save to the gitignored
`config.local.toml`.

Stage 2 exposes global resonance controls in the Streamlit app sidebar under **Memory tuning**. You can preview the current top three versus a proposed top three and save a personal profile to gitignored `config.local.toml`. The sidebar shows whether an archive-specific Favorite-trained taste model exists and lets you turn it off for clean manual-weight experiments. The taste model is never portable between users.

Hard exclusions are available in the app sidebar under **Hard exclusions**. Select named people or subject categories, optionally enter anonymous cluster IDs, and save. Excluded items are removed from dispatch but remain in the historical graph and database. This is a local dispatch policy, not relationship inference or a retraining label.

## Development checks

```bash
make test
make lint
.venv/bin/python scripts/publish_check.py
```

The publish check scans for personal data and denied strings before public release.

## Roadmap

### V1 — current

Local ingest, clutter rejection, burst election, technical metrics, face detection, conservative identity, anonymous recurrence, five-quadrant scoring, resonance dispatch, insight graph, local narrators, camera/lens/depth/context nodes, and Streamlit comparison app.

### Stage 3 — in progress

Camera focal-length backfill (`camera-scan`): find cameras missing a 35mm-equivalent
focal and propose crop factors before analysis.

### Deferred

Relationship inference, per-person soft weighting, friend-ready setup wizard, calibrated face identity, local place-region clustering, smaller sidecars/database, and richer privacy-policy workflows.
