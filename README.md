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

## Install

From a fresh clone:

```bash
git clone https://github.com/sde-agent/memory-graph.git
cd memory-graph
make setup
source .venv/bin/activate
make init
memory-graph doctor
```

`make setup` creates and installs into `.venv`; it cannot activate that environment in your parent shell. If you do not want to activate it, prefix every command with `.venv/bin/`, for example:

```bash
.venv/bin/memory-graph doctor
.venv/bin/memory-graph graph
```

If you pull new repository commits into an existing checkout, refresh the editable CLI installation:

```bash
git pull
make refresh
memory-graph --version
memory-graph sample-run --help
```

The `sample-run` help must show `--all`. If it does not, the command is still resolving an older environment; use the explicit path:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/memory-graph sample-run --all
```

Running `memory-graph` with no command now prints the help. The valid syntax is `memory-graph sample-run --all`; `-sample-run` is not valid argparse syntax.

The setup command is the only networked step. It installs the local Python package and development dependencies.

## Configure your archive

Copy the personal overlay and edit it:

```bash
cp config.local.toml.example config.local.toml
```

Put personal values only in `config.local.toml`, which is gitignored:

- your Apple Photos face name under `[owner].self_names`;
- crop factors under `[framing.crop]`;
- life-era ranges under `[eras].ranges`;
- Hidden albums under `[privacy].hidden_albums`.

The external HDD root is passed as a command-line argument rather than committed into configuration:

```bash
memory-graph ingest-hdd "/Volumes/YourDrive/Photos"
```

## Check camera crop factors (recommended before first analysis)

Many older cameras (Canon DSLRs and compacts, point-and-shoots, early phones) record
only the raw lens focal length, not the 35mm equivalent. Uncorrected, every such photo
is misread as ultra-wide, which skews the framing insight and the quadrant/resonance
scores. Scan your sources for affected cameras and write the proposed crop factors:

```bash
memory-graph camera-scan              # list models missing a 35mm equivalent
memory-graph camera-scan --confirm    # write the resolvable factors to config.local.toml
```

The scan reads metadata only (no masters, no pixels) and completes in seconds. Models
it cannot resolve are reported, not guessed; add those by hand under `[framing.crop]`.
Add `--hdd-root "/Volumes/YourDrive/Photos"` to include an external archive, or
`--hdd-only --hdd-root ...` to scan only the drive. If you have already built a graph,
run `memory-graph rescore` after confirming to correct it without re-decoding.

## Build from Apple Photos (default)

An external HDD is optional. The normal first run uses the macOS Photos Library and local derivatives only:

```bash
memory-graph sample-run --all
memory-graph graph
```

This scans the complete Photos Library metadata pool, deduplicates bursts and cross-source duplicates, rejects utility images, analyzes every surviving canonical photo, and writes the local graph. `--all` means all canonical survivors; it does not create one row for every burst member or duplicate asset.

The Photos Library requires Full Disk Access for the terminal. If the run is interrupted, repeat the command; the analysis is resumable and existing photo IDs are skipped.

If you see `FileNotFoundError: Could not get path to photo library database`, follow these exact steps:

1. Open the **Photos** app and confirm the library opens normally.
2. In Photos, use **Photos → Settings → General → Use as System Photo Library** if that button is available.
3. Open **System Settings → Privacy & Security → Full Disk Access**.
4. Click the **+** button and authenticate with your Mac password/Touch ID. You must add the application that launches the command:
   - Apple Terminal: `/Applications/Utilities/Terminal.app`
   - iTerm: `/Applications/iTerm.app`
   - Kiro: `/Applications/Kiro.app`
   - VS Code: `/Applications/Visual Studio Code.app`
5. Quit and reopen that application. Then test:

```bash
memory-graph ingest-library --limit 1
```

If you run the command in an IDE's integrated terminal, add the IDE itself—not only Apple Terminal. If **Full Disk Access** is empty, that is normal: click **+** to add the application manually; it does not appear automatically.

If the Photos Library is stored somewhere non-default, find its database:

```bash
find "$HOME/Pictures" -name photos.sqlite -print
```

Then pass the result explicitly:

```bash
memory-graph sample-run --all --dbfile "/path/to/photos.sqlite"
```

The HDD is optional. `--dbfile` is only needed for a non-default Photos Library location.

The commands print a progress bar/phase status for pool scanning, deduplication, optics, Vision analysis, and graph generation. A full Photos Library run can take a while; leave the terminal open and repeat the command if interrupted.

The runner reports the Library total, cheap rejections, canonical count, Vision rejections, and final database count. This makes coverage auditable instead of silently claiming that every asset became a separate graph row.

## Optional external HDD

If you also have an external archive, pass its root explicitly. The HDD is not required for Apple Photos-only users:

```bash
memory-graph analyze-hdd "/Volumes/YourDrive/Photos"
```

For the optimized archive-cache workflow, set `MEMORY_GRAPH_ARCHIVE_CACHE` before `sample-run`. Without that environment variable and without an HDD cache, the runner simply uses the Photos Library source.

After adding sources, rebuild the graph:

```bash
memory-graph graph
```

## Identity and recurring people

Build the named gallery and process local derivatives:

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

## Explore the graph

```bash
memory-graph dispatch -n 3
memory-graph ask "What was my style in the 2010s?"
memory-graph ask "How do my camera choices affect the people I photograph?"
memory-graph critique
memory-graph graph
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
