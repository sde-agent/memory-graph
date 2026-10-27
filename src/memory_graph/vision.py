"""Apple Vision bindings (on-device, local, read-only).

Two capabilities the engine reuses across phases:
  - `classify(thumb)`       -> scene identifiers + confidences (rejection, P2).
  - `saliency_boxes(thumb)` -> salient-object boxes (geometry / negative space, P4).

Everything runs on-device through pyobjc; nothing is written and no file is handed to
Vision (we pass in-memory JPEG bytes of the already-decoded thumbnail). Vision objects
are released after each call via an autorelease pool — without it, Vision fails with
"Failed to create CVPixelBuffer" after ~8000 calls (a lesson inherited from the
prototype). `available()` lets callers degrade gracefully where Vision is absent.
"""

from __future__ import annotations

import io
from typing import Any

ANALYSIS_PX = 1024


def available() -> bool:
    """True if the Apple Vision pyobjc bindings can be imported on this machine."""
    try:
        import Foundation  # noqa: F401
        import objc  # noqa: F401
        import Vision  # noqa: F401
    except Exception:
        return False
    return True


def _jpeg_bytes(thumb: Any) -> bytes:
    im = thumb.copy()
    im.thumbnail((ANALYSIS_PX, ANALYSIS_PX))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def norm(identifier: str) -> str:
    """Normalize a Vision identifier for comparison: spaces, lowercased, trimmed."""
    return identifier.replace("_", " ").strip().lower()


def analyze_image(thumb: Any, want: tuple[str, ...], keep_min: float = 0.05) -> dict | None:
    """Run several Vision requests on ONE image through ONE handler (the fast path).

    `want` is a subset of {"classify","saliency","faces","text","featureprint"}. Encoding
    the JPEG and building the handler ONCE for all requests is ~2.4x faster than a
    separate call per task (measured). Returns a dict keyed by the requested names, or
    None if Vision is unavailable. Individual results are None if that request failed.
    """
    if not available():
        return None
    import Foundation
    import objc
    import Vision

    builders = {
        "classify": Vision.VNClassifyImageRequest.alloc().init,
        "saliency_attention": Vision.VNGenerateAttentionBasedSaliencyImageRequest.alloc().init,
        "saliency_objectness": Vision.VNGenerateObjectnessBasedSaliencyImageRequest.alloc().init,
        "faces": Vision.VNDetectFaceLandmarksRequest.alloc().init,
        "text": Vision.VNDetectTextRectanglesRequest.alloc().init,
        "featureprint": Vision.VNGenerateImageFeaturePrintRequest.alloc().init,
    }
    wanted = set(want)
    if "saliency" in wanted:
        wanted.discard("saliency")
        wanted.update({"saliency_attention", "saliency_objectness"})

    raw = _jpeg_bytes(thumb)
    with objc.autorelease_pool():
        data = Foundation.NSData.dataWithBytes_length_(raw, len(raw))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        reqs = {name: builders[name]() for name in wanted if name in builders}
        ok, _ = handler.performRequests_error_(list(reqs.values()), None)
        if not ok:
            return None
        out: dict = {}
        if "classify" in reqs:
            out["classify"] = [
                (str(o.identifier()), float(o.confidence()))
                for o in reqs["classify"].results() or []
                if float(o.confidence()) >= keep_min
            ]
        if "saliency_attention" in reqs or "saliency_objectness" in reqs:
            out["saliency"] = {
                "attention": _boxes(reqs["saliency_attention"])
                if "saliency_attention" in reqs
                else [],
                "objectness": _boxes(reqs["saliency_objectness"])
                if "saliency_objectness" in reqs
                else [],
            }
        if "faces" in reqs:
            out["faces"] = [_face(o) for o in reqs["faces"].results() or []]
        if "text" in reqs:
            out["text"] = _text_coverage_from_request(reqs["text"])
        if "featureprint" in reqs:
            results = reqs["featureprint"].results() or []
            out["featureprint"] = _featureprint_to_list(results[0]) if results else None
        return out


def classify(thumb: Any, keep_min: float = 0.05) -> list[tuple[str, float]] | None:
    """Scene (identifier, confidence) pairs at/above keep_min, or None if Vision fails/absent."""
    res = analyze_image(thumb, ("classify",), keep_min=keep_min)
    return res.get("classify") if res else None


def saliency_boxes(thumb: Any) -> dict[str, list[dict[str, float]]] | None:
    """Salient-object boxes (attention + objectness) normalized top-left, or None."""
    res = analyze_image(thumb, ("saliency",))
    return res.get("saliency") if res else None


def _boxes(request: Any) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for observation in request.results() or []:
        for obj in observation.salientObjects() or []:
            box = obj.boundingBox()  # Vision origin is bottom-left
            out.append(
                {
                    "x": float(box.origin.x),
                    "y": float(1.0 - box.origin.y - box.size.height),  # flip to top-left
                    "w": float(box.size.width),
                    "h": float(box.size.height),
                    "confidence": float(obj.confidence()),
                }
            )
    return out


def _text_coverage_from_request(request: Any) -> float:
    """Fraction of the frame covered by detected text regions, from a text request."""
    import numpy as np

    grid = 100
    mask = np.zeros((grid, grid), dtype=bool)
    for obs in request.results() or []:
        box = obs.boundingBox()
        x0 = max(0, min(grid, int(box.origin.x * grid)))
        y0 = max(0, min(grid, int((1.0 - box.origin.y - box.size.height) * grid)))
        x1 = max(0, min(grid, int((box.origin.x + box.size.width) * grid)))
        y1 = max(0, min(grid, int((1.0 - box.origin.y) * grid)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def text_coverage(thumb: Any) -> float | None:
    """Fraction of the frame covered by text regions, 0..1, or None if Vision is absent.

    Region detection (NOT OCR) — measures how text-heavy an image is without reading the
    content (a privacy nicety: we never OCR an ID's numbers).
    """
    res = analyze_image(thumb, ("text",))
    return res.get("text") if res else None


def detect_faces(thumb: Any) -> list[dict[str, Any]] | None:
    """Detect faces with landmarks (bbox, eye-openness, yaw), or None if Vision absent."""
    res = analyze_image(thumb, ("faces",))
    return res.get("faces") if res else None


def _face(obs: Any) -> dict[str, Any]:
    box = obs.boundingBox()  # bottom-left origin, normalized
    face: dict[str, Any] = {
        "x": float(box.origin.x),
        "y": float(1.0 - box.origin.y - box.size.height),  # flip to top-left
        "w": float(box.size.width),
        "h": float(box.size.height),
        "area": float(box.size.width * box.size.height),
    }
    # Yaw (head turn) when exposed: large |yaw| => turned away => more "candid".
    try:
        yaw = obs.yaw()
        face["yaw"] = float(yaw) if yaw is not None else None
    except Exception:
        face["yaw"] = None
    landmarks = None
    try:
        landmarks = obs.landmarks()
    except Exception:
        landmarks = None
    face["eye_openness"] = _eye_openness(landmarks)
    return face


def _region_openness(region: Any) -> float | None:
    """Vertical/horizontal extent ratio of an eye landmark region (open eyes are taller)."""
    if region is None:
        return None
    try:
        pts = region.normalizedPoints()
        count = region.pointCount()
    except Exception:
        return None
    if not count:
        return None
    xs = [pts[i].x for i in range(count)]
    ys = [pts[i].y for i in range(count)]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0:
        return None
    return float(height / width)


def _eye_openness(landmarks: Any) -> float | None:
    """Mean eye open-ratio across both eyes, or None if landmarks are absent."""
    if landmarks is None:
        return None
    try:
        left = _region_openness(landmarks.leftEye())
        right = _region_openness(landmarks.rightEye())
    except Exception:
        return None
    vals = [v for v in (left, right) if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def feature_print(thumb: Any) -> list[float] | None:
    """A VNFeaturePrint embedding of the image (float vector), or None.

    Apple's on-device feature print — the iOS-portable alternative to CLIP embeddings
    (used by the taste model and the identity gallery).
    """
    res = analyze_image(thumb, ("featureprint",))
    return res.get("featureprint") if res else None


def _featureprint_to_list(obs: Any) -> list[float] | None:
    """Convert a VNFeaturePrintObservation's data into a Python float list."""
    try:
        import struct

        count = int(obs.elementCount())
        data = obs.data()  # NSData of float32 (elementType 1)
        buf = bytes(data)
        if len(buf) < count * 4:
            return None
        return list(struct.unpack(f"<{count}f", buf[: count * 4]))
    except Exception:
        return None
