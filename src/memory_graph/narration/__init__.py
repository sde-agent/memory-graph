"""Insight graph + narration (spec.md §8, evaluation.md Part C).

The memory graph is an INSIGHT graph. Insights are engine-authored, structured claims
derived by code from deterministic metrics + resonance; photos are evidence hanging off
them. The LLM narrates a committed insight — free to interpret, fenced by the insight's
claim — it does NOT author insights. The template layer is the trusted, exact-match
floor; the LLM layer is verified for groundedness and falls back HONESTLY (it says so)
when it cannot be trusted.
"""

from memory_graph.narration.templates import photo_facts, photo_sentence
from memory_graph.narration.verify import GroundednessResult, verify_narration

__all__ = ["photo_facts", "photo_sentence", "verify_narration", "GroundednessResult"]
