# Building a Photographic Memory Graph

## An AI product-management case study

This project started with a simple question:

> Can a local-first system surface the photographs that make someone stop and feel something—and explain why those photographs matter?

The answer required more than a photo browser, a search box, or a generic vision model. It required a product thesis, a data model, a ranking system, an insight graph, local privacy guardrails, and repeated correction when the system sounded plausible but was wrong.

## The problem

A lifetime photo archive contains far more than files. It contains eras, relationships, creative habits, technical choices, and memories that are not expressed by filenames or dates.

A useful companion needed to answer two kinds of questions:

1. **Daily memory:** Which one to three photographs would I be glad to see today?
2. **Creative reflection:** What was my photographic style in the 2010s? How did my camera choices change what I photographed? Did I prefer night or day? Did I favor bokeh, open space, or close human moments?

The product constraint was strict: photographs must stay on the Mac. Pixels cannot be sent to a hosted model. The system may expose grounded text and a loopback link, but not image bytes.

## The product thesis

Technical choices are emotional posture.

- A normal lens at eye level can feel like preservation and memory.
- A telephoto lens isolating a face can feel like tenderness and attention.
- An ultra-wide lens from a low angle can express scale and awe.
- Negative space and long exposure can express solitude.
- A canted horizon or shutter drag can create tension.

The system therefore maps photographs into five soft emotional quadrants, but it does not treat the quadrant as a final truth. It combines optics, light, geometry, faces, content, identity, favorites, age, and learned taste.

## Architecture

The system has four important layers:

### 1. Local ingest and measurement

The engine reads:

- external HDD archives sequentially;
- macOS Photos metadata through `osxphotos`;
- local Photos derivatives only, never Photos masters;
- in-memory thumbnails, EXIF, Vision signals, face attributes, scene labels, optics, light, geometry, and sharpness.

The current V1 database contains **55,771 canonical photographs**:

- 22,737 from the HDD;
- 33,034 from the Photos Library.

The full archive started as a 9,639-photo stratified sample. Expanding it increased the evidence base by about 5.8× and increased the Favorite signal from 72 to 649 examples.

### 2. Resonance ranking

The dispatch score is a configurable signal, not an emotional oracle:

```text
resonance = quadrant_affinity
          × candidness
          × favorite
          × known_person
          × taste_model
          × blur_adjustment
```

The current V1 global settings are:

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

The weights are intentionally visible and configurable. V1 does **not** yet provide per-person, per-camera, per-subject, or hard exclusion controls. That is a V2 requirement.

### 3. Insight graph

The graph is not a dump of photo rows. It is a set of engine-authored claims, each with retrieval tags and evidence. The LLM narrates selected claims; it does not invent the claims.

The final V1 graph contains **192 nodes across 33 dimensions**, including:

- color and seasonal patterns;
- focal length and framing;
- depth-of-field and bokeh-friendly optical proxies;
- ISO, horizon geometry, negative space, sharpness, and candidness;
- iPhone/DSLR comparisons and camera-model signatures;
- lens, travel, and location-coverage observations;
- people, relationships, named identity, anonymous recurring faces, and identity caveats;
- subjects, landscapes, golden-hour associations, and light behavior;
- emotional style by era;
- longitudinal changes and resonance gaps;
- taste and favorite explanations.

### 4. Swappable local narrators

The same grounded notes can be narrated by:

- Apple Foundation Models on-device;
- Apple Private Cloud Compute when packaging/entitlements permit;
- the local Claude CLI for development comparison.

The app compares narrators on the same evidence. This exposed an important product truth: better language synthesis does not repair missing or incorrectly selected evidence.

## The most important course corrections

### The insight graph was initially too coarse

The first graph had 126 nodes even after the archive grew dramatically. A test with finer year buckets produced 461 nodes from the same photographs, proving that node count was controlled by era granularity rather than archive richness.

But finer slicing made answers worse: the narration budget filled with redundant yearly fragments. The right lesson was not “add more nodes.” It was “add meaningful dimensions and protect the selection budget.”

### The narrator was not always the problem

Several bad answers were caused by retrieval:

- a color question pulled a golden-hour person node because both used the word “warm”;
- a camera-and-people question pulled a golden-hour friend-of-family node because it shared a generic `people` tag;
- an era-style question omitted human context because style nodes consumed the entire selection budget.

The fixes were retrieval boundaries, topic-specific filtering, and reserved human-context slots—not more poetic prompts.

### D700 identity coverage was a real blind spot

The D700 period contains many detected faces but almost no named identities:

- 15,251 HDD photos with detected faces;
- zero HDD photos with native named identities before identity bridging;
- only sparse cross-source named matches using experimental generic feature prints.

The system now persists face observations and clusters recurring anonymous faces. It has **871 anonymous recurring-person clusters** and **10,384 assigned face observations**. It can say that recurring unnamed people are present without pretending to know who they are.

The system now distinguishes:

> “People were central during this period.”

from:

> “This named person was definitely the center of that period.”

That distinction materially improved the 2010s answer.

### Golden-hour lift was not the same as frequency

The first golden-hour person rule promoted people with a high relative lift even when they appeared in only a handful of frames. That made a friend's wife from one trip compete with the people who appeared hundreds of times.

The fix added a frequency-ranked golden-hour node and raised support requirements for individual lift claims. The current frequency evidence identifies the people who actually occur most often, while low-count relative lifts no longer dominate.

### The camera data existed before the camera nodes did

The system always stored camera make and model, but the graph initially had no camera dimension. This led to the false narrator statement that the system did not know which photographs came from an iPhone or DSLR.

The graph now promotes camera families, camera models, camera-by-era patterns, camera-by-subject patterns, and iPhone-versus-DSLR comparisons.

## What was measured

The project ran a pre-registered dispatch holdout exercise, but the result was not promoted into a success claim or used to retune the final weights. The pooled ranking lift was above random control but below the pre-registered PASS bar, so it remained inconclusive. The feelings-label test was not informative.

That outcome was useful as a product lesson: measurement should not be used as decoration. If a test does not establish a conclusion, the product should say so.

The full-library expansion did establish useful engineering outcomes:

- taste-model held-out AUC improved from 0.831 to 0.853;
- the archive grew from 9,639 to 55,771 canonical photos;
- identity recurrence became measurable;
- camera, depth, lens, location coverage, travel, and resonance insights became possible.

## Privacy and safety decisions

- Photos never leave the Mac.
- Photos Library masters are never opened.
- Local derivatives are the only Library image source.
- Full-resolution photos are never copied into the project.
- Hidden photos are excluded from dispatch, identity gallery construction, and anonymous clustering.
- A name below the recognition threshold remains unnamed.
- Anonymous clusters never become invented human names.
- The graph preserves unweighted historical truth separately from resonance.
- Hosted narrators receive grounded text, not pixels.

## What V1 is—and is not

V1 is a working local-first research product for one archive. It has:

- ingest and clutter filtering;
- burst/keyframe reduction;
- optics, light, geometry, scene, face, and identity signals;
- resonance ranking and daily dispatch;
- a persisted insight graph;
- local narrator comparison;
- anonymous recurrence clustering;
- a Streamlit reflection app;
- local privacy and egress guardrails.

V1 is not yet a polished downloadable product for arbitrary friends. It still has:

- no tuning panel for personal weights;
- no per-person hard exclusions or soft weights;
- no friend-ready setup wizard;
- experimental generic Vision feature-print identity matching;
- unresolved place names because reverse geocoding is intentionally not performed;
- a database larger than the original 100 MB target after adding face-level persistence.

## V2 product brief

V2 should turn the one-archive research system into a configurable personal memory product.

### Personal tuning

Provide a local tuning panel for:

- Q1–Q5 quadrant weights;
- per-person soft weights;
- hard person exclusions;
- camera-family/model weights;
- subject weights;
- place and travel controls;
- bokeh, sharpness, negative-space, and candidness preferences.

Tuning must affect dispatch and resonance, not erase historical facts from the graph.

### Setup for a friend’s Mac

A new user should be able to:

1. install the package;
2. run a doctor/setup flow;
3. select their Photos Library and external archive roots;
4. grant Full Disk Access with clear instructions;
5. identify themselves among Apple Photos named faces;
6. confirm cameras and crop factors;
7. review clutter detections;
8. configure Hidden albums and exclusions;
9. build a local named gallery;
10. run identity and anonymous recurrence calibration;
11. generate the graph;
12. choose Apple Foundation Models as the default narrator.

Personal paths, names, eras, roles, weights, exclusions, and derived data must stay in gitignored local configuration.

### V2 product principle

The system should learn a user’s resurfacing preferences without pretending that those preferences are universal truth. The archive tells the historical story. The personal tuning layer decides what deserves attention today.

## Product-management takeaway

The most important work was not adding more AI. It was deciding where the AI was allowed to speak, what evidence it could use, how uncertainty was expressed, and when a plausible answer had to be rejected.

The system became more useful every time a fluent answer was treated as a bug report rather than a success.
