"""Pure image helpers shared by both ingest adapters.

Turns a decoded image upright per its EXIF orientation, downsamples to the working
thumbnail (long edge from config), and computes a perceptual hash for burst/near-dup
detection. Kept dependency-light and side-effect-free so it is fully unit-testable.
"""

from __future__ import annotations

from PIL import Image, ImageOps


def to_thumb(img: Image.Image, long_edge: int) -> Image.Image:
    """Return an upright RGB thumbnail whose long edge is at most `long_edge` px.

    `ImageOps.exif_transpose` applies the embedded EXIF orientation so the thumbnail is
    visually upright regardless of how the camera wrote it. We never upscale.
    """
    upright = ImageOps.exif_transpose(img)
    rgb = upright.convert("RGB")
    rgb.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    return rgb


def phash(img: Image.Image, hash_size: int = 8) -> str:
    """A simple average-hash hex string (perceptual hash).

    Average hash: downscale to (hash_size+... ) grayscale, threshold each pixel against
    the mean, pack the bits. Robust enough for burst / near-duplicate grouping and has
    no third-party dependency. Hamming distance between two hashes measures similarity.
    """
    small = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())  # row-major grayscale byte values
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, px in enumerate(pixels):
        if px >= avg:
            bits |= 1 << i
    width = (hash_size * hash_size + 3) // 4  # hex digits
    return f"{bits:0{width}x}"


def hamming(a: str, b: str) -> int:
    """Hamming distance between two equal-length hex hash strings."""
    if len(a) != len(b):
        raise ValueError("hash length mismatch")
    return bin(int(a, 16) ^ int(b, 16)).count("1")
