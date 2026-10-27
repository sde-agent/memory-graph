"""Resonance scoring and daily dispatch (spec.md §6).

Resonance ranks the daily dispatch and is a first-class signal for the LLM critique.
It is a personal, multiplicative blend over the soft quadrant vector with graceful
degradation (any unavailable signal contributes a neutral 1.0). See evaluation.md
Part A for the full contract.
"""

from memory_graph.scoring.resonance import age_boost, resonance

__all__ = ["resonance", "age_boost"]
