"""Camera focal-length backfill: inventory, crop-factor proposals, and local writes.

This package exists for one Stage 3 job: before the first analysis, find camera models
that lack a 35mm-equivalent focal length, propose a crop factor per model, and (after
explicit confirmation) write the factors to the gitignored local config overlay. The
existing `sample-run` / `graph` pipeline does the analysis; this does not re-implement it.
"""

from memory_graph.setup.cameras import inventory_cameras, propose_crop_factors
from memory_graph.setup.models import CameraProposal
from memory_graph.setup.writer import update_crop_factors

__all__ = [
    "CameraProposal",
    "inventory_cameras",
    "propose_crop_factors",
    "update_crop_factors",
]
