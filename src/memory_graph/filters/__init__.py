"""Candidate filters. P2: clutter / document / screenshot rejection.

Filters never delete anything. They flag candidates for owner review and remove them
from the resonance/dispatch candidate pool only. The asymmetric-loss rule (a false
reject of a cherished photo is far worse than a false keep) is honored by favoring
KEEP wherever a signal is weak or unavailable.
"""

from memory_graph.filters.rejection import ClutterFlags, classify_clutter, is_clutter

__all__ = ["ClutterFlags", "classify_clutter", "is_clutter"]
