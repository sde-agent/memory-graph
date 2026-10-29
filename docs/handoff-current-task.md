# Current Task Handoff

## Context

This repository is a local-first photographic memory system for macOS. It analyzes a personal photo archive using metadata, local derivatives, Apple Vision, resonance scoring, a personal Favorite-trained taste signal, and grounded narration. It is designed to surface emotionally meaningful memories and explain recurring photographic choices without sending photo pixels to hosted models.

The current handoff is at the end of the Stage 2 release work and before Stage 3 begins.

## What we completed and why

### Restored the stable pre-relationship checkpoint

An experimental relationship-graph implementation was added, tested, and then reverted after it introduced unnecessary complexity and regressions in the product direction. The active branch was reset to:

```text
ffefbcc Keep hard exclusions separate from taste retraining
```

The relationship-graph files are absent from the active checkout:

- `src/memory_graph/scoring/relationships.py`
- `tests/test_relationships.py`
- `V2_RELATIONSHIP_GRAPH.md`

The repository no longer treats the relationship graph as part of the current release.

### Confirmed the existing taste-model path

The restored local app was refreshed with `make refresh` and launched successfully. The app loaded the personal local configuration, including the archive-specific taste weight and resonance settings. The app responded at the local Streamlit URL during testing.

The taste model remains an optional resonance input. It influences dispatch ranking but does not rewrite the historical graph or claim to be a psychological truth.

### Prepared the Stage 2 release documentation

The documentation now explains:

- what is implemented in Stage 2;
- what remains deferred to Stage 3;
- that hard exclusions are dispatch policy, not relationship inference or retraining labels;
- that private archive counts and AUC values are case-study evidence, not portable defaults;
- that the shipped taste model uses Apple Vision feature prints rather than CLIP;
- how local configuration, privacy, and release validation work;
- that the old pre-build spec language is no longer accurate.

Updated documents include:

- `README.md`
- `CASE_STUDY.md`
- `V2_PRODUCT_BRIEF.md`
- `V2_PERSONAL_MODEL.md`
- `.kiro/spec.md`
- `.kiro/evaluation.md`
- `config.toml`
- `config.local.toml.example`
- `pyproject.toml`

### Strengthened the publish guard

`scripts/publish_check.py` now checks both the current working tree and added content in branch/tag history. It fails closed if Git history cannot be scanned and does not print matched personal values.

The release guard also confirms that personal configuration and generated data remain ignored:

```text
config.local.toml
data/
```

### Validated and committed the release state

Validation completed:

```text
358 tests passed
Ruff: passed
Publish check: clean
Working tree: clean
```

The current release commits are:

```text
bb23b85 Prepare Stage 2 release baseline
17d8c25 Scan Git history in publish guard
```

The active branch is ready to push or download. No push has been performed.

## Current problem

There is no known failing Stage 2 implementation problem. The local app works, the restored taste-model path works, and the automated checks pass.

The current problem is product scope for Stage 3. The relationship and exclusions direction was becoming larger than the core product needed. It mixed several different concerns:

- social or relationship inference;
- per-person life trajectories;
- soft per-person weighting;
- hard exclusion policy;
- narration-time privacy filtering;
- taste-model retraining after exclusions.

Those concerns should not be reintroduced as one large graph feature. Stage 2 should remain the stable baseline while Stage 3 is narrowed to a specific, measurable user problem.

## What we ruled out

### The relationship graph is not required for the current product

The daily dispatch and taste-model workflow function without the experimental relationship-graph commit. Relationship inference is deferred rather than treated as a required Stage 2 dependency.

### The recent failure was not caused by the taste model

The restored app loaded the local taste configuration and returned working results. The taste model remains separate from hard exclusions and historical insight claims.

### The issue was not a broken local app installation

The editable package was refreshed successfully. The Streamlit app started and returned an HTTP 200 response on its local port.

### The issue was not a failing test or lint baseline

The stable checkout passes the complete test suite and Ruff. The final release validation produced 358 passing tests.

### Hard exclusions do not need to become a graph

Current exclusions can remain local dispatch filters. They remove eligible items from resurfacing while leaving historical evidence and stored graph facts unchanged. They do not need to be represented as relationship nodes or automatically converted into negative taste labels.

### A larger node graph is not automatically a better product

The project previously explored increasing graph richness and adding relationship-specific nodes. More nodes can reduce retrieval quality, inflate minor people, and make the narrator overstate weak evidence. Stage 3 must improve a defined user outcome, not simply increase node count.

## Open questions for the owner

Answer these before implementing Stage 3:

1. **What exact user problem should Stage 3 solve?** Choose one primary outcome, such as better local privacy controls, better identity accuracy, better setup for a new user, or recurring unnamed-person discovery.
2. **Should Stage 3 include relationship inference at all?** If yes, define the narrow claim the system is allowed to make. For example, observed co-occurrence is materially different from inferred relationship or emotional bond.
3. **What should an exclusion mean?** Decide whether it affects dispatch only, direct historical questions, narration, identity-gallery construction, taste-model training, or some explicitly separated combination.
4. **Should excluded photos remain in the taste-model training set?** The current Stage 2 policy keeps exclusions separate from retraining. Changing that requires a new provenance and evaluation rule.
5. **What privacy behavior is required for a downloadable product?** Decide whether the next release needs a setup wizard, owner identification, crop-factor discovery, hidden-album review, or a simpler documented manual setup.
6. **What is the acceptance dataset for Stage 3?** Identify the labeled photos, questions, or privacy scenarios that will determine whether the change improves the product.
7. **Should the old stash be kept?** The local stash contains the files that were uncommitted before restoring the stable checkpoint. It is not part of the branch or a fresh clone. It can be retained as a safety copy or deleted after confirming it is no longer needed.

## Next three concrete steps

### 1. Publish and smoke-test the Stage 2 baseline

Push or tag commit `17d8c25`, then test it from a fresh clone on the same Mac:

```bash
git clone <repository-url> memory-graph-stage2
cd memory-graph-stage2
make setup
cp config.local.toml.example config.local.toml
make init
make doctor
make test
make lint
.venv/bin/python scripts/publish_check.py
make app
```

Confirm that the fresh clone contains no personal data, no relationship-graph files, and no generated local database or sidecar files.

### 2. Choose one Stage 3 problem and write its acceptance criteria

Use the open questions above to select one feature. Write down:

- the user-visible behavior;
- the privacy boundary;
- the data required;
- the failure behavior;
- the unit and live evaluation cases;
- the explicit non-goals.

Do not implement a combined relationship-and-exclusions graph.

### 3. Create a narrow Stage 3 spec and branch

Create a new branch from `17d8c25` and add a focused spec with requirements, design, tasks, and tests before changing the Stage 2 code. Keep the Stage 2 release immutable while the new experiment is evaluated. Any Stage 3 change should preserve the existing zero-egress, groundedness, local-storage, and publish-check guardrails.

## Current release boundary

Stage 2 is the stable downloadable baseline. It includes local photo ingest, utility-image rejection, burst canonicalization, technical and quadrant metrics, conservative identity and aggregate recurrence, resonance dispatch, the personal taste signal, grounded narration, the insight graph, MCP text-only surfaces, and the local reflection app.

Stage 3 is not started. Relationship inference, richer per-person policy, automatic setup discovery, calibrated identity, and other new personalization remain deferred until the owner answers the open questions and defines a measurable target.
