"""Burst clustering and canonical keyframe election (spec.md §3).

Collapses runs of near-identical frames (bursts, rapid re-shots of one moment) into a
single Shot with an elected canonical representative, so the memory graph holds one
node per moment rather than thirty. Only representatives flow downstream; members are
retained for grouping and audit.
"""

from memory_graph.cluster.bursts import Shot, cluster_and_elect, global_dedup

__all__ = ["Shot", "cluster_and_elect", "global_dedup"]
