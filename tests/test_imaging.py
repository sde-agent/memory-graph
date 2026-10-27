"""Pure image helpers: thumbnail, orientation, pHash (P1)."""

from __future__ import annotations

from PIL import Image

from memory_graph.ingest.imaging import hamming, phash, to_thumb


def _solid(w: int, h: int, color=(120, 120, 120)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def test_thumb_respects_long_edge_and_never_upscales() -> None:
    big = _solid(2000, 1000)
    t = to_thumb(big, 768)
    assert max(t.size) == 768
    assert t.size == (768, 384)  # aspect preserved

    small = _solid(300, 200)
    t2 = to_thumb(small, 768)
    assert t2.size == (300, 200)  # not upscaled


def test_thumb_is_rgb() -> None:
    t = to_thumb(_solid(100, 100).convert("L"), 64)
    assert t.mode == "RGB"


def test_phash_identical_images_match() -> None:
    a = _solid(256, 256, (10, 200, 50))
    b = _solid(256, 256, (10, 200, 50))
    assert phash(a) == phash(b)
    assert hamming(phash(a), phash(b)) == 0


def test_phash_differs_for_different_content() -> None:
    gradient = Image.linear_gradient("L").convert("RGB")
    flat = _solid(256, 256, (0, 0, 0))
    assert hamming(phash(gradient), phash(flat)) > 0


def test_hamming_length_mismatch_raises() -> None:
    try:
        hamming("ff", "ffff")
        raised = False
    except ValueError:
        raised = True
    assert raised
