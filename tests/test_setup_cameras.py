from __future__ import annotations

from memory_graph.setup.cameras import propose_crop_factors


def test_lookup_includes_explicit_full_frame() -> None:
    proposals = propose_crop_factors([
        {"make": "Canon", "model": "Canon EOS R", "focal": 50.0},
    ])
    assert proposals[0].proposed_crop_factor == 1.0
    assert proposals[0].source == "lookup"


def test_derived_factor_uses_consistent_raw_equivalent_pairs() -> None:
    proposals = propose_crop_factors([
        {"make": "Test", "model": "Unknown", "focal": 10.0, "focal35": 15.0},
        {"make": "Test", "model": "Unknown", "focal": 20.0, "focal35": 30.0},
    ])
    assert proposals[0].proposed_crop_factor == 1.5
    assert proposals[0].source == "derived"
    assert proposals[0].raw_focal_min == 10.0
    assert proposals[0].raw_focal_max == 20.0


def test_unresolved_model_has_no_silent_factor() -> None:
    proposals = propose_crop_factors([
        {"make": "Test", "model": "Unknown", "focal": 10.0},
    ])
    assert proposals[0].proposed_crop_factor is None
    assert proposals[0].source == "unresolved"


def test_existing_equivalent_values_can_derive_missing_crop_factor() -> None:
    proposals = propose_crop_factors([
        {"model": "Mixed", "focal": 35.0, "focal35": 52.5},
    ])
    assert proposals[0].source == "derived"
    assert proposals[0].proposed_crop_factor == 1.5
