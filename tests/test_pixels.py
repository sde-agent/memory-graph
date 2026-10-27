"""Pixel statistics (P2)."""

from __future__ import annotations

from PIL import Image

from memory_graph.metrics.pixels import pixel_stats


def test_flat_gray_has_low_contrast_and_saturation() -> None:
    s = pixel_stats(Image.new("RGB", (128, 128), (128, 128, 128)))
    assert s.contrast < 0.03
    assert s.saturation < 0.05


def test_uniform_frame_high_uniformity_any_hue() -> None:
    # A flat purple frame: spatially uniform regardless of its high saturation.
    purple = pixel_stats(Image.new("RGB", (128, 128), (128, 0, 200)))
    assert purple.uniformity > 0.85
    black = pixel_stats(Image.new("RGB", (128, 128), (0, 0, 0)))
    assert black.uniformity > 0.85


def test_varied_image_low_uniformity() -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8), "RGB")
    assert pixel_stats(noisy).uniformity < 0.5


def test_vivid_image_has_saturation() -> None:
    s = pixel_stats(Image.new("RGB", (128, 128), (220, 20, 20)))
    assert s.saturation > 0.5


def test_warm_image_has_positive_warm_cool() -> None:
    warm = pixel_stats(Image.new("RGB", (64, 64), (200, 120, 40)))
    cool = pixel_stats(Image.new("RGB", (64, 64), (40, 120, 200)))
    assert warm.warm_cool > 0
    assert cool.warm_cool < 0


def test_high_contrast_gradient() -> None:
    grad = Image.linear_gradient("L").convert("RGB")
    assert pixel_stats(grad).contrast > 0.2
