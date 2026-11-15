"""Pure Zhang-skeleton Top-K baseline for Stage 5."""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize

from src.contracts import (
    CandidateUtilityProjector,
    ElectrodeGrid,
    FROZEN_TOPOLOGY_DETECTION,
    FloatImage,
    SelectionResult,
)
from src.encoders.base import BaseEncoder, EncoderError, deterministic_topk


class SkeletonTopKEncoder(BaseEncoder):
    """Rank the shared projections of a binary one-pixel Zhang skeleton."""

    name = "skeleton_topk"

    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: CandidateUtilityProjector,
    ) -> SelectionResult:
        self.validate_inputs(image, grid, budget, projector)
        spec = FROZEN_TOPOLOGY_DETECTION
        binary_mask = image >= np.float32(spec.binarization_threshold)
        skeleton = skeletonize(binary_mask, method=spec.skeletonization)
        drive_map = np.ascontiguousarray(skeleton, dtype=np.float32)
        if drive_map.shape != image.shape or not np.isfinite(drive_map).all():
            raise EncoderError("Zhang skeletonization produced an invalid drive map")
        if float(drive_map.max()) != 1.0:
            raise EncoderError("Zhang skeleton drive map is empty")
        if not np.all((drive_map == 0.0) | (drive_map == 1.0)):
            raise EncoderError("Zhang skeleton drive map must be binary")

        scores = projector.project(drive_map)
        if scores.shape != (grid.candidate_count,) or scores.dtype != np.float32:
            raise EncoderError("skeleton candidate scoring produced an invalid array")
        if float(scores.max()) <= 0.0:
            raise EncoderError("skeleton projection produced no positive utility")

        selected = deterministic_topk(scores, budget)
        return SelectionResult(
            encoder_name=self.name,
            budget=budget,
            candidate_count=grid.candidate_count,
            selected_indices=selected,
            candidate_scores=np.ascontiguousarray(scores, dtype=np.float32),
            drive_map=drive_map,
            metadata={
                "binarization_threshold": spec.binarization_threshold,
                "skeletonization": spec.skeletonization,
                "drive_map": "binary_one_pixel_skeleton",
                "candidate_projection": "l1_ideal_gaussian",
                "selection": "deterministic_topk",
                "tie_break": "candidate_index_ascending",
                "topology_enhancement": "none",
            },
        )
