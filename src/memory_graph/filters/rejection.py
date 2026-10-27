"""Clutter / document / screenshot rejection (spec.md §2, evaluation.md B.2).

Flags candidates that are probably NOT photographs worth keeping — screenshots,
scanned documents, whiteboards, receipts, blank/flat frames. A flag only PROPOSES;
the owner reviews and decides (never auto-deleted). Flagging merely removes a candidate
from the resonance/dispatch pool.

ASYMMETRIC LOSS: a false reject of a cherished photo is far worse than a false keep.
So every rule here is conservative — it fires only on strong, specific evidence, and
when a signal is missing (no Vision, no pixel stats) the candidate is KEPT.

Signals, in order of reliability:
  screenshot : Library screenshot flag, or exact device-screen dimensions, or a PNG
               with a screenshot classification, or Vision `screenshot` >= threshold.
  document   : a Vision document-like identifier at/above the threshold.
  flat       : almost no contrast AND almost no saturation (a blank wall / scan).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory_graph.config import Config
from memory_graph.contracts import Candidate
from memory_graph.metrics.pixels import PixelStats, pixel_stats
from memory_graph.vision import classify, norm, text_coverage

# Device screen sizes, as sorted (short, long). An exact match is a strong screenshot cue.
_SCREEN_SIZES = frozenset(
    tuple(sorted(s))
    for s in (
        (640, 960), (640, 1136), (750, 1334), (828, 1792), (1080, 1920), (1125, 2436),
        (1170, 2532), (1179, 2556), (1206, 2622), (1242, 2208), (1242, 2688), (1284, 2778),
        (1290, 2796), (1320, 2868),
        (768, 1024), (1536, 2048), (1488, 2266), (1620, 2160), (1640, 2360), (1668, 2224),
        (1668, 2388), (2048, 2732), (2064, 2752),
        (1280, 800), (1440, 900), (1680, 1050), (1920, 1080), (1920, 1200), (2304, 1440),
        (2560, 1440), (2560, 1600), (2880, 1800), (3024, 1964), (3456, 2234), (3840, 2160),
        (5120, 2880),
    )
)  # fmt: skip


@dataclass(frozen=True)
class ClutterFlags:
    screenshot: bool = False
    document: bool = False
    flat: bool = False
    text_doc: bool = False    # high text density (document / ID card / receipt / text shot)
    uniform: bool = False     # a single-color / flat-wall frame, any hue

    @property
    def any(self) -> bool:
        return self.screenshot or self.document or self.flat or self.text_doc or self.uniform

    def reasons(self) -> tuple[str, ...]:
        out = []
        for flag, name in (
            (self.screenshot, "screenshot"),
            (self.document, "document"),
            (self.flat, "flat"),
            (self.text_doc, "text_doc"),
            (self.uniform, "uniform"),
        ):
            if flag:
                out.append(name)
        return tuple(out)


def _is_screen_size(w: Any, h: Any) -> bool:
    try:
        return bool(w and h and tuple(sorted((int(w), int(h)))) in _SCREEN_SIZES)
    except (TypeError, ValueError):
        return False


# Vision labels that mark REAL-PHOTO content — their presence protects an image from the
# text-density document reject (a child near a sign, a storefront, a banner at a party).
_PROTECTIVE_IDS = frozenset({
    "people", "adult", "child", "baby", "face", "person", "crowd", "group",
    "outdoor", "sky", "nature", "landscape", "mountain", "beach", "water", "tree",
    "plant", "flower", "animal", "pet", "dog", "cat", "food", "sunset", "sunrise",
})


def _has_identifier(found: list[tuple[str, float]] | None, wanted: tuple[str, ...],
                    threshold: float) -> bool:
    if not found:
        return False
    wanted_set = {norm(w) for w in wanted}
    return any(norm(i) in wanted_set and c >= threshold for i, c in found)


def _has_any(found: list[tuple[str, float]] | None, wanted: frozenset[str],
             threshold: float) -> bool:
    if not found:
        return False
    return any(norm(i) in wanted and c >= threshold for i, c in found)


def classify_clutter(
    cand: Candidate,
    cfg: Config,
    *,
    found: list[tuple[str, float]] | None = None,
    stats: PixelStats | None = None,
    text_cov: float | None = None,
    cheap_only: bool = False,
) -> ClutterFlags:
    """Return the clutter flags for a candidate.

    `found` is an optional precomputed Vision classification; `stats` optional pixel
    statistics; `text_cov` optional precomputed text-coverage fraction. Any omitted and
    computable (a thumbnail is present) are computed here. Vision-based rules simply do
    not fire when Vision is unavailable — asymmetric loss favors KEEPING.

    `cheap_only=True` runs ONLY the no-Vision signals (screen dimensions, Library flag,
    pixel-stats flat/uniform). This is the first gate when building a large pool: it
    avoids running Vision (document/text/scene classification) on photos that will be
    sampled out. The Vision gate then runs on the far smaller sampled set.
    """
    rej = cfg.rejection
    meta = cand.library_meta or {}

    if not cheap_only and found is None and cand.thumb is not None:
        found = classify(cand.thumb)
    if stats is None and cand.thumb is not None:
        stats = pixel_stats(cand.thumb)
    if not cheap_only and text_cov is None and cand.thumb is not None:
        text_cov = text_coverage(cand.thumb)

    width = cand.exif.get("width")
    height = cand.exif.get("height")

    # --- screenshot -------------------------------------------------------------------
    # A PNG alone is NOT enough (many real exports are PNG); it only counts when Vision
    # also calls it a screenshot, which the vision clause already captures.
    screenshot = (
        bool(meta.get("screenshot"))
        or _is_screen_size(width, height)
        or _has_identifier(found, rej.screenshot_ids, rej.vision_confidence)
    )

    # --- document (Vision scene label) ------------------------------------------------
    document = _has_identifier(found, rej.document_ids, rej.vision_confidence)

    # --- text-heavy document / ID card / receipt (text-region density) ----------------
    # Catches license/insurance/OCR-like images Vision's scene labels miss, by measuring
    # how much of the frame is text (NOT reading it — privacy). BUT text alone is not
    # enough: a real photo can contain a sign, a banner, or a storefront. A true document
    # is text-heavy AND has no real-photo content (people, faces, scenery). So we reject
    # only when text is high AND Vision does not strongly see protective content. This
    # keeps a child-near-a-sign while still catching a receipt (finding: a blanket text
    # threshold wrongly rejected a cherished child photo at text=0.27).
    protective = _has_any(found, _PROTECTIVE_IDS, rej.vision_confidence)
    text_doc = bool(
        text_cov is not None and text_cov >= rej.text_coverage_reject and not protective
    )

    # --- flat / nothing: near-no contrast AND near-no saturation (a black scan) -------
    flat = bool(
        stats is not None
        and stats.contrast < rej.low_contrast_below
        and stats.saturation < rej.low_saturation_below
    )

    # --- single-color / flat-wall frame, ANY hue -------------------------------------
    # Keys on SPATIAL UNIFORMITY, not hue, so a uniform purple wall is rejected but a
    # vivid, varied purple sunset or a moody low-key portrait (both spatially varied)
    # survives.
    uniform = bool(stats is not None and stats.uniformity >= rej.uniformity_reject)

    return ClutterFlags(
        screenshot=screenshot, document=document, flat=flat,
        text_doc=text_doc, uniform=uniform,
    )


def is_clutter(cand: Candidate, cfg: Config, **kw: Any) -> bool:
    """Convenience: True if the candidate carries any clutter flag."""
    return classify_clutter(cand, cfg, **kw).any
