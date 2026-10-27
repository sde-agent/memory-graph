"""Local owner-labeling for tuning (evaluation.md A.2-A.4, B.6).

A 127.0.0.1-only web UI to collect the labels that turn the pre-registered thresholds
into tuned ones, against a target instead of blind:
  - base-rate:  "would I be glad to see this?" yes/no       -> p_random (A.2)
  - quadrant:   "which emotional quadrant does this evoke?" -> router accuracy (B.6)
  - pairwise:   "which would you rather be shown?"          -> resonance weights (A.3)

Photos are served only to the loopback address and never leave the device. Labels are
written to data/labels/*.jsonl, appended as you go, so the session is resumable.
"""

from memory_graph.labeling.store import LabelStore

__all__ = ["LabelStore"]
