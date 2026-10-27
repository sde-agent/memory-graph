# Memory Graph Engine

A local-first photographic memory companion for macOS. It is not an asset manager or generic photo search tool. It analyzes the technical and human patterns in a personal archive, builds a grounded insight graph, and helps answer questions such as:

- “What was my style in the 2010s?”
- “How do my camera choices affect the people I photograph?”
- “Do I prefer bokeh or deep focus?”
- “Which people and moments keep recurring?”

Read the full story in [CASE_STUDY.md](CASE_STUDY.md) and the next-product plan in [V2_PRODUCT_BRIEF.md](V2_PRODUCT_BRIEF.md).

## V1 status

V1 is a working local research product validated on a real archive:

- 55,771 canonical photos: 22,737 HDD and 33,034 Photos Library;
- 192 grounded insight nodes across 33 dimensions;
- 52,114 face observations and 871 anonymous recurring-person clusters;
- Apple Vision, local derivatives, local SQLite, and local narrator adapters;
- 357 passing tests and clean Ruff lint.

The system is intentionally honest about limits. Generic Vision face embeddings are experimental, place names are not reverse-geocoded, and per-person/subject resonance tuning is a V2 feature.

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

## Build from Apple Photos (default)

An external HDD is optional. The normal first run uses the macOS Photos Library and local derivatives only:

```bash
memory-graph sample-run --all
memory-graph graph
```

This scans the complete Photos Library metadata pool, deduplicates bursts and cross-source duplicates, rejects utility images, analyzes every surviving canonical photo, and writes the local graph. `--all` means all canonical survivors; it does not create one row for every burst member or duplicate asset.

The Photos Library requires Full Disk Access for the terminal. If the run is interrupted, repeat the command; the analysis is resumable and existing photo IDs are skipped.

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

## Current resonance weights

V1 exposes global resonance settings in `config.toml`:

| Signal | Value |
|---|---:|
| Q1 Nostalgic & Reflective | 0.55 |
| Q2 Visceral & High-Arousal | 0.45 |
| Q3 Empathetic & Connected | 0.80 |
| Q4 Solitary & Introspective | 0.55 |
| Q5 Provocative & Dynamic | 0.25 |
| Known named person | 2.0× |
| Any detected face | 1.3× |
| Favorite | 1.5× |
| Taste-model influence | 1.0 |

V1 does not yet expose per-person or hard exclusion tuning. That is a deliberate V2 product milestone. Historical insight claims remain separate from dispatch preference tuning.

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

### V2 — next

Personal tuning weights, per-person hard exclusions, camera/subject/place controls, friend-ready setup, crop-factor discovery, calibrated face identity, local place-region clustering, smaller sidecars/database, and a polished Apple Foundation Models package.
