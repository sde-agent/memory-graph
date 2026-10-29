"""Pure camera inventory and 35mm-equivalent crop proposals.

The functions in this module consume metadata-shaped records. They deliberately do
not open images, Photos masters, or source files, which makes the proposal step safe
to run before the first analysis.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from memory_graph.config import Config
from memory_graph.setup.models import CameraProposal

# Small, public, conservative lookup. Values are sensor crop factors, not lens crops.
# Full-frame bodies are explicit (1.0) so they can never be accidentally scaled.
# Matching is case-insensitive on the EXIF model string (see `_lookup`). This is a
# convenience for common bodies; anything absent is reported UNRESOLVED, never guessed.
CAMERA_CROP_LOOKUP: dict[str, float] = {
    # Canon full-frame
    "Canon EOS 5D Mark II": 1.0,
    "Canon EOS 5D Mark III": 1.0,
    "Canon EOS 5D Mark IV": 1.0,
    "Canon EOS 6D": 1.0,
    "Canon EOS 6D Mark II": 1.0,
    "Canon EOS R": 1.0,
    "Canon EOS R5": 1.0,
    "Canon EOS R6": 1.0,
    # Canon APS-C (1.6x)
    "Canon EOS 7D": 1.6,
    "Canon EOS Rebel XTi": 1.6,
    "Canon EOS DIGITAL REBEL XTi": 1.6,
    "Canon EOS REBEL T5": 1.6,
    "Canon EOS 70D": 1.6,
    "Canon EOS 80D": 1.6,
    "Canon EOS M50": 1.6,
    # Nikon full-frame (FX)
    "NIKON D700": 1.0,
    "NIKON D750": 1.0,
    "NIKON D810": 1.0,
    "NIKON Z 6": 1.0,
    "NIKON Z 7": 1.0,
    # Nikon APS-C (DX, 1.5x)
    "NIKON D50": 1.5,
    "NIKON D5100": 1.5,
    "NIKON D7000": 1.5,
    "NIKON D7200": 1.5,
    # Sony full-frame
    "Sony ILCE-7M3": 1.0,
    "ILCE-7M3": 1.0,
    "ILCE-7RM4": 1.0,
    # Fujifilm APS-C
    "FUJIFILM X-T3": 1.5,
    "FUJIFILM X-T4": 1.5,
    "X100S": 1.52,
    # Common iPhones (fixed main-camera sensor crop vs 35mm)
    "iPhone 4": 7.64,
    "iPhone 5": 7.21,
    "iPhone 5s": 7.21,
    "iPhone 6": 7.21,
    "iPhone 6s": 7.21,
    "iPhone 7": 7.21,
    "iPhone 8": 7.21,
    "iPhone 8 Plus": 6.7,
    "iPhone X": 6.86,
    "iPhone XS": 6.86,
    "iPhone 11": 6.86,
    "iPhone 11 Pro": 6.86,
    "iPhone 12": 6.86,
    "iPhone 12 Pro": 6.86,
    "iPhone 13 Pro": 5.6,
}


@dataclass(frozen=True)
class CameraObservation:
    """Metadata needed for one camera inventory observation."""

    make: str | None = None
    model: str | None = None
    raw_focal: float | None = None
    focal_35mm: float | None = None


def _value(record: Any, *names: str) -> Any:
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
        return None
    for name in names:
        value = getattr(record, name, None)
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, Real) and not isinstance(value, bool):
        value = float(value)
    else:
        try:
            value = float(str(value).split()[0])
        except (TypeError, ValueError, IndexError):
            return None
    return value if value > 0 else None


def _observation(record: Any) -> CameraObservation:
    """Read either a CameraObservation, EXIF dict, or candidate-like record."""
    exif = _value(record, "exif")
    if isinstance(exif, Mapping):
        source = exif
    else:
        source = record
    make = _value(source, "make", "camera_make")
    model = _value(source, "model", "camera_model")
    raw = _value(source, "focal", "focal_length", "raw_focal")
    equivalent = _value(source, "focal35", "focal_35mm", "focal_length_35mm")
    return CameraObservation(
        make=str(make).strip() if make else None,
        model=str(model).strip() if model else None,
        raw_focal=_number(raw),
        focal_35mm=_number(equivalent),
    )


def _model_key(model: str) -> str:
    return " ".join(model.split())


def inventory_cameras(records: Iterable[Any]) -> tuple[CameraObservation, ...]:
    """Normalize camera metadata records, retaining one observation per record."""
    return tuple(
        obs
        for obs in (_observation(record) for record in records)
        if obs.model
    )


def _lookup(model: str, lookup: Mapping[str, float]) -> float | None:
    if model in lookup:
        return lookup[model]
    folded = model.casefold()
    for key, factor in lookup.items():
        if key.casefold() == folded:
            return factor
    return None


def propose_crop_factors(
    records: Iterable[Any],
    config: Config | None = None,
    *,
    lookup: Mapping[str, float] = CAMERA_CROP_LOOKUP,
) -> tuple[CameraProposal, ...]:
    """Aggregate records and return explicit lookup, derived, or unresolved proposals.

    Existing local crop configuration wins over the public lookup. A derived factor is
    accepted only when raw/equivalent ratios agree closely (within 5%); otherwise the
    model remains unresolved instead of receiving a guessed factor.
    """
    observations = inventory_cameras(records)
    grouped: dict[str, list[CameraObservation]] = defaultdict(list)
    for obs in observations:
        assert obs.model is not None
        grouped[_model_key(obs.model)].append(obs)
    configured = config.framing.crop if config is not None else {}
    proposals: list[CameraProposal] = []
    for model, items in sorted(grouped.items()):
        makes = sorted({item.make for item in items if item.make})
        raw = [item.raw_focal for item in items if item.raw_focal is not None]
        ratios = [item.focal_35mm / item.raw_focal for item in items
                  if item.raw_focal and item.focal_35mm]
        factor = configured.get(model)
        source = "configured" if factor is not None else ""
        if factor is None:
            factor = _lookup(model, lookup)
            source = "lookup" if factor is not None else ""
        if factor is None and ratios:
            average = sum(ratios) / len(ratios)
            if max(ratios) - min(ratios) <= average * 0.05:
                factor = average
                source = "derived"
        if factor is None:
            source = "unresolved"
        proposals.append(
            CameraProposal(
                make=", ".join(makes),
                model=model,
                raw_focal_min=min(raw) if raw else None,
                raw_focal_max=max(raw) if raw else None,
                proposed_crop_factor=factor,
                source=source,
                affected_count=len(items),
            )
        )
    return tuple(proposals)


# Friendly aliases for callers that prefer noun/verb naming.
camera_inventory = inventory_cameras
camera_proposals = propose_crop_factors
