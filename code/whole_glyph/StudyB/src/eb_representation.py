"""One shared EB representation path used identically by EB-TopK and EB-SSE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.contracts import CandidateUtilityProjector, ElectrodeGrid, FloatImage, FloatVector
from src.eb_drive import EbDriveComponents, build_eb_drive_components


class EbRepresentationError(ValueError):
    """Raised when shared D_EB or candidate utility is invalid."""


@dataclass(frozen=True, slots=True)
class EbRepresentationResult:
    components: EbDriveComponents = field(repr=False)
    candidate_scores: FloatVector = field(repr=False)

    def __post_init__(self) -> None:
        if self.candidate_scores.dtype != np.float32 or self.candidate_scores.ndim != 1:
            raise TypeError("candidate_scores must be a one-dimensional float32 array")
        if not np.isfinite(self.candidate_scores).all():
            raise EbRepresentationError("candidate_scores contain non-finite values")
        if float(self.candidate_scores.min()) < 0.0 or float(
            self.candidate_scores.max()
        ) <= 0.0:
            raise EbRepresentationError(
                "candidate_scores must be non-negative with at least one positive value"
            )


def build_shared_eb_representation(
    image: FloatImage,
    grid: ElectrodeGrid,
    projector: CandidateUtilityProjector,
) -> EbRepresentationResult:
    """Build Stage-6 D_EB and project it once with the shared ideal projector."""

    components = build_eb_drive_components(image)
    scores = projector.project(components.drive_map)
    if scores.shape != (grid.candidate_count,) or scores.dtype != np.float32:
        raise EbRepresentationError("shared EB candidate scoring returned invalid output")
    return EbRepresentationResult(
        components=components,
        candidate_scores=np.ascontiguousarray(scores, dtype=np.float32),
    )
