"""Metric extractors: pure functions over (thumbnail, exif) producing engine attributes.

P2 introduces only pixel statistics (used by the rejection filter). Later phases add
optics, geometry, light, and faces. All extractors here are side-effect-free and
unit-testable.
"""
