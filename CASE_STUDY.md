# Building a Photographic Memory Graph

I built this because I wanted to understand my own photographs—not just find them.

A normal photo search can tell me when a picture was taken, where it was taken, or what appears in it. I wanted something closer to a personal visual memory: a system that could surface a photograph that made me pause, and explain the technical choices and human context behind it.

This is a technical account of what I built, what surprised me, what failed, and what the archive taught me.

## The question

A lifetime archive contains more than files. It contains changes in attention:

- who I kept photographing;
- how close I stood;
- which cameras changed my habits;
- whether I preferred open space or crowded frames;
- whether I used shallow focus, motion, or sharpness to create a feeling;
- which people and periods became important in retrospect.

I wanted to ask questions such as:

- “What was my style in the 2010s?”
- “How do my camera choices affect the people I photograph?”
- “Do I prefer bokeh or deep focus?”
- “Do I prefer night or day?”
- “Which memories should return to me today?”

The central constraint was privacy. The photographs stay on my Mac. Pixels never go to a hosted model. A narrator can receive grounded text, but never image bytes.

## The idea

I treated photographic technique as a kind of emotional posture.

- A normal lens at eye level can feel like preservation.
- A telephoto lens isolating a face can feel like tenderness and attention.
- An ultra-wide lens from a low angle can express scale and awe.
- Negative space and long exposure can express solitude.
- A canted horizon or shutter drag can create tension.

The system turns those signals into five soft emotional quadrants. They are not presented as psychological truth. They are a vocabulary for describing recurring visual choices.

## How the system works

### Local ingest

The engine reads an external HDD sequentially and reads macOS Photos through `osxphotos`. It uses local Photos derivatives and never opens Photos masters, which avoids triggering iCloud downloads.

Thumbnails are temporary. The system extracts measurements, releases the image, and stores metadata and derived signals rather than full-resolution copies.

The final Stage 2 archive run contains **55,771 canonical photographs**. These numbers describe my private archive and are included as a reproducibility snapshot, not as a promise about another user's installation:

- 22,737 from the HDD;
- 33,034 from the Photos Library.

The project began with a 9,639-photo stratified sample. Expanding to the full available archive made the evidence denser and increased the Favorite signal from 72 to 649 examples.

### Measurements

Each photo can carry:

- camera make, body, and lens metadata;
- raw and 35mm-equivalent focal length;
- aperture, shutter speed, ISO, and derived EV;
- horizon angle and height;
- negative-space estimate;
- golden-hour and warmth signals;
- sharpness;
- Vision scene labels;
- face count, face size, gaze, eyes, and candidness;
- named people when identity is trustworthy;
- anonymous recurring-face clusters;
- quadrant scores and resonance inputs.

### Memory graph

The system does not hand raw rows to the narrator and ask it to discover the story. Code promotes observations into structured insight nodes first. Each node has a claim, tags, an optional direction and magnitude, and evidence context.

The final Stage 2 graph contains **192 nodes across 33 dimensions** in this archive. It includes aggregate people presence, trusted named identity, and anonymous recurrence alongside technical and emotional dimensions. It does not claim to infer a universal social graph. Relationship inference, life-trajectory interpretation, and richer per-person policy controls are deferred to Stage 3.

The language model narrates selected nodes. It does not create the underlying claims.

### Resonance

The daily memory score is configurable:

```text
resonance = quadrant_affinity
          × candidness
          × favorite
          × known_person
          × taste_model
          × blur_adjustment
```

The current V1 settings are:

| Signal | Current value |
|---|---:|
| Q1 Nostalgic & Reflective | 0.55 |
| Q2 Visceral & High-Arousal | 0.45 |
| Q3 Empathetic & Connected | 0.80 |
| Q4 Solitary & Introspective | 0.55 |
| Q5 Provocative & Dynamic | 0.25 |
| Quadrant contribution scale | 0.60 |
| Known named person | 2.0× |
| Any detected face | 1.3× |
| Favorite | 1.5× |
| Candidness range | 0.95×–1.05× |
| Taste-model influence | 1.0 |
| Blur adjustment | neutral at 1.0× |

These are current engineering settings, not universal truths about photography. The Stage 2 app exposes global tuning and dispatch-only hard exclusions. Richer per-person weighting and narration-time exclusions are deferred to Stage 3.

## What went wrong, and what I learned

### More photos did not automatically mean more nodes

The first graph had 126 nodes. A finer time slicing experiment produced 461 nodes from the same photos, but the answers became worse because the selection budget filled with repetitive yearly fragments.

The useful lesson was that graph richness comes from meaningful dimensions, not from generating an enormous pile of nodes.

### Several bad answers were retrieval bugs

A color question once pulled a golden-hour person because both the color node and the person node contained the word “warm.” A camera-and-people question pulled a golden-hour acquaintance because it shared the generic tag “people.”

The fix was not more eloquent prompting. I added topic boundaries and dimension-aware retrieval. A color question now retrieves color evidence. A camera-and-people question retrieves camera, relationship, and person-specific evidence without importing unrelated golden-hour facts.

### The D700 years were partially identity-blind

The D700 years contain many detected faces but almost no named identities:

- 15,251 HDD photographs with detected faces;
- zero native HDD name labels before identity bridging;
- only sparse high-confidence cross-source name matches using experimental generic feature prints.

I added face-level persistence and anonymous recurrence clustering. The current pass contains **871 anonymous recurring-person clusters** and **10,384 assigned face observations**.

That lets the system say:

> People were central during this period, and recurring unnamed people appear throughout it.

It does not pretend that every unnamed face is a known family member.

### Golden-hour lift was not the same as frequency

The first golden-hour person rule promoted a person with a high relative lift even when that person appeared in only a handful of frames. That made a one-trip connection compete with people who appeared hundreds of times.

I changed the graph to promote frequency separately from relative lift. The current golden-hour frequency observation names the people who actually occur most often, while low-support lift effects no longer become the headline.

### Camera data existed before camera insights

Camera make and model were stored from the beginning, but the graph had no camera dimension. That produced a false narrator disclaimer: the system knew the camera data, but had not promoted it into retrievable evidence.

The graph now compares iPhone and DSLR behavior, promotes common camera bodies, and connects camera family to subjects and people.

### Depth of field was stored indirectly before it became visible

Aperture and focal length were already stored and were used by the quadrant router, but there were no explicit depth-of-field or bokeh nodes. I added a labeled optical proxy:

> wide aperture plus normal/long focal length can create bokeh-friendly conditions.

That is deliberately not described as a direct measurement of rendered background blur.

## Identity and recurrence

The named gallery uses local Apple Photos derivatives and authoritative Apple Photos names. Generic Vision feature prints are treated conservatively because they are not a dedicated face-recognition model.

Unmatched face vectors are clustered locally. Anonymous clusters never become invented human names. Hidden photos are excluded before gallery and clustering work.

This is one of the most valuable parts of the system: even when identity is unavailable, recurrence can still reveal that the same human presence appears repeatedly.

## Narration

The reflection app compares Apple Foundation Models and the local Claude CLI using the same selected evidence. This made the difference between retrieval quality and narrator quality visible.

A stronger narrator cannot repair missing evidence. Conversely, once the evidence is correct, the difference between models becomes a question of synthesis, tone, and emotional precision.

The narrator also has a grounding verifier. It rejects prose that invents names or introduces unsupported terms. One bug surfaced when camera names such as “Nikon,” “iPhone,” and “Pro” were mistaken for invented people; the verifier was updated to allow terms present in grounded camera claims.

## What V1 contains

- local HDD and Photos Library ingest;
- utility-image rejection;
- burst grouping and canonical keyframe election;
- optics, light, geometry, scene, sharpness, and face measurements;
- named identity attempts and anonymous recurrence clustering;
- five-quadrant soft scoring;
- resonance ranking and local daily dispatch;
- a persisted insight graph;
- camera, lens, depth, travel, light, landscape, people, and relationship insights;
- Apple Foundation Models and local Claude narrator adapters;
- a Streamlit reflection app;
- loopback-only viewing and local privacy guardrails.

The release checkpoint has **358 passing tests** and clean Ruff lint. The publish check must pass before distribution. The private archive counts above are illustrative evidence from one run, not defaults or acceptance criteria for another user's archive.

## Current limits

- Generic Vision face embeddings remain experimental.
- Stage 2 global tuning and dispatch-only hard exclusions are available in the local app; richer per-person weighting and narration-time exclusions are deferred to Stage 3.
- Per-camera and per-subject tuning is not yet implemented.
- Place names are not reverse-geocoded; the system only reports local geographic coverage unless names are already available.
- Face-level persistence expanded the database to roughly 120 MB, above the original 100 MB target.
- The system works on my Mac but is not yet a polished first-run experience for another person.

## Stage 3 boundary

Stage 3 may add:

- local controls for per-person weights and exclusions;
- camera, subject, place, and technical-construct weights;
- a setup wizard for Photos permissions, owner identity, cameras, eras, and privacy;
- better face-specific identity calibration;
- friend-ready source discovery and resumable setup;
- smaller database and sidecar formats;
- clearer explanations of uncertainty and data coverage.

The guiding rule stays the same: the archive records what happened; personal tuning decides what deserves attention today.
