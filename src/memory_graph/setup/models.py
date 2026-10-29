"""Validated, side-effect-free records used by the camera backfill flow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraProposal:
    """A proposed crop factor awaiting explicit user confirmation."""

    make: str
    model: str
    raw_focal_min: float | None
    raw_focal_max: float | None
    proposed_crop_factor: float | None
    source: str
    affected_count: int | None = None
    confirmed: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("camera model must not be empty")
        if self.raw_focal_min is not None and self.raw_focal_min <= 0:
            raise ValueError("raw focal lengths must be positive")
        if self.raw_focal_max is not None and self.raw_focal_max <= 0:
            raise ValueError("raw focal lengths must be positive")
        if self.raw_focal_min is not None and self.raw_focal_max is not None:
            if self.raw_focal_min > self.raw_focal_max:
                raise ValueError("raw focal range must be increasing")
        if self.proposed_crop_factor is not None and self.proposed_crop_factor <= 0:
            raise ValueError("crop factor must be positive")
        if self.affected_count is not None and self.affected_count < 0:
            raise ValueError("affected photo count must not be negative")
        if not self.source.strip():
            raise ValueError("proposal source must not be empty")

    @property
    def camera_model(self) -> str:
        return self.model
