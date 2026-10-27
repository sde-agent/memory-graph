"""Clutter rejection filter (P2).

Vision results are injected (`found=`) so these tests are deterministic and do not need
Apple Vision installed. The asymmetric-loss rule is asserted explicitly: a normal photo
with no strong signal is KEPT.
"""

from __future__ import annotations

from PIL import Image

from memory_graph.config import load_config
from memory_graph.contracts import Candidate, Source
from memory_graph.filters.rejection import classify_clutter, is_clutter
from memory_graph.metrics.pixels import PixelStats


def _cand(**kw) -> Candidate:
    base = dict(id="x", source=Source.HDD, original_ref="/v/x.jpg")
    base.update(kw)
    return Candidate(**base)


def cfg():
    return load_config()


def _stats(contrast=0.22, saturation=0.3, warm_cool=0.1, uniformity=0.0) -> PixelStats:
    return PixelStats(contrast=contrast, saturation=saturation, warm_cool=warm_cool,
                      uniformity=uniformity)


def test_normal_photo_is_kept(monkeypatch) -> None:
    # A rich outdoor scene: Vision sees non-document labels, stats are vivid.
    c = _cand(exif={"width": 4000, "height": 3000})
    found = [("beach", 0.9), ("sky", 0.7)]
    flags = classify_clutter(c, cfg(), found=found, stats=_stats(), text_cov=0.0)
    assert not flags.any


def test_document_flagged() -> None:
    c = _cand(exif={"width": 3000, "height": 2000})
    found = [("document", 0.8)]
    flags = classify_clutter(c, cfg(), found=found, stats=_stats(saturation=0.1),
                             text_cov=0.0)
    assert flags.document
    assert "document" in flags.reasons()


def test_whiteboard_flagged_as_document() -> None:
    c = _cand()
    flags = classify_clutter(c, cfg(), found=[("whiteboard", 0.5)], stats=None)
    assert flags.document


def test_low_confidence_document_not_flagged() -> None:
    # Below the 0.25 threshold => not flagged (favor keeping).
    c = _cand()
    flags = classify_clutter(c, cfg(), found=[("document", 0.1)], stats=None)
    assert not flags.document


def test_screenshot_by_exact_screen_dimensions() -> None:
    c = _cand(exif={"width": 1170, "height": 2532})  # iPhone screen
    flags = classify_clutter(c, cfg(), found=[], stats=None)
    assert flags.screenshot


def test_screenshot_by_library_flag() -> None:
    c = _cand(library_meta={"screenshot": True})
    flags = classify_clutter(c, cfg(), found=[], stats=None)
    assert flags.screenshot


def test_screenshot_by_vision_identifier() -> None:
    c = _cand(exif={"width": 1234, "height": 2000})
    flags = classify_clutter(c, cfg(), found=[("screenshot", 0.9)], stats=None)
    assert flags.screenshot


def test_flat_image_flagged() -> None:
    c = _cand()
    flags = classify_clutter(c, cfg(), found=[], stats=_stats(contrast=0.01, saturation=0.01),
                             text_cov=0.0)
    assert flags.flat


def test_flat_requires_both_low_contrast_and_low_saturation() -> None:
    c = _cand()
    # Low contrast but vivid color (e.g. a solid red card) => NOT flat (favor keeping).
    flags = classify_clutter(c, cfg(), found=[], stats=_stats(contrast=0.01, saturation=0.6),
                             text_cov=0.0)
    assert not flags.flat


def test_text_heavy_document_flagged() -> None:
    # A license/receipt/text screenshot: lots of the frame is text.
    c = _cand()
    flags = classify_clutter(c, cfg(), found=[], stats=_stats(), text_cov=0.4)
    assert flags.text_doc
    assert "text_doc" in flags.reasons()


def test_incidental_text_not_flagged() -> None:
    # A photo with a small sign in it: low text coverage => KEEP (favor keeping).
    c = _cand()
    flags = classify_clutter(c, cfg(), found=[], stats=_stats(), text_cov=0.05)
    assert not flags.text_doc


def test_uniform_frame_flagged_any_hue() -> None:
    # A flat purple wall: spatially uniform, high saturation -> still rejected.
    c = _cand()
    flags = classify_clutter(c, cfg(), found=[], stats=_stats(saturation=0.7, uniformity=0.95),
                             text_cov=0.0)
    assert flags.uniform


def test_varied_scene_not_uniform() -> None:
    # A vivid, varied sunset: high saturation but spatially varied -> KEEP.
    c = _cand()
    flags = classify_clutter(c, cfg(), found=[], stats=_stats(saturation=0.7, uniformity=0.2),
                             text_cov=0.0)
    assert not flags.uniform


def test_missing_vision_and_stats_keeps_photo() -> None:
    # No Vision, no stats, no thumbnail, ordinary dimensions: must KEEP (asymmetric loss).
    c = _cand(exif={"width": 4000, "height": 3000})
    flags = classify_clutter(c, cfg(), found=None, stats=None, text_cov=None)
    assert not flags.any


def test_is_clutter_convenience() -> None:
    c = _cand()
    assert is_clutter(c, cfg(), found=[("receipt", 0.7)], stats=None, text_cov=0.0)
    assert not is_clutter(c, cfg(), found=[("dog", 0.9)], stats=None, text_cov=0.0)


def test_classify_clutter_consults_vision_when_thumb_present(monkeypatch) -> None:
    # When found is omitted but a thumbnail exists, the filter calls vision.classify.
    import memory_graph.filters.rejection as rej

    called = {}

    def fake_classify(thumb, keep_min=0.05):
        called["yes"] = True
        return [("receipt", 0.8)]

    monkeypatch.setattr(rej, "classify", fake_classify)
    monkeypatch.setattr(rej, "text_coverage", lambda thumb: 0.0)
    c = _cand(thumb=Image.new("RGB", (100, 100), (200, 200, 200)))
    flags = rej.classify_clutter(c, cfg())
    assert called.get("yes")
    assert flags.document
