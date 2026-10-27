"""Sharpness metric: Laplacian variance (P3)."""

from __future__ import annotations

from PIL import Image, ImageFilter

from memory_graph.metrics.sharpness import region_variance, sharpness


def _detailed() -> Image.Image:
    # High-frequency detail (fine noise) => strong Laplacian response. Blurring it must
    # reduce sharpness, which is the real-world crisp-vs-blurry case.
    import numpy as np

    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    return Image.fromarray(noise, mode="L").convert("RGB")


def test_blurred_is_less_sharp_than_crisp() -> None:
    crisp = _detailed()
    blurred = crisp.filter(ImageFilter.GaussianBlur(radius=4))
    assert sharpness(crisp) > sharpness(blurred)


def test_flat_image_is_near_zero_sharpness() -> None:
    flat = Image.new("RGB", (128, 128), (128, 128, 128))
    assert sharpness(flat) < 1e-3


def test_region_variance_falls_back_to_whole_on_none() -> None:
    img = _detailed()
    assert region_variance(img, None) == sharpness(img)


def test_region_variance_on_subbox_runs() -> None:
    img = _detailed()
    v = region_variance(img, (0.25, 0.25, 0.5, 0.5))
    assert v >= 0.0


def test_degenerate_box_falls_back() -> None:
    img = _detailed()
    # A box too small to analyze falls back to the whole-image variance.
    assert region_variance(img, (0.0, 0.0, 0.001, 0.001)) == sharpness(img)
