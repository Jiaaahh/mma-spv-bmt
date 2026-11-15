"""Mapping-aware Zhang-skeleton Top-K control condition."""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize

from src.contracts import (
    ElectrodeGrid,
    FROZEN_TOPOLOGY_DETECTION,
    FloatImage,
    SelectionResult,
)
from src.encoders.base import EncoderError, deterministic_topk
from src.v2.mapped_projector import MappingAwareGaussianUtilityProjector


class MappingAwareSkeletonTopKEncoder:
    """Rank a binary Zhang skeleton with the target mapping's actual kernels."""

    name = "calibrated_skeleton_topk"

    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: MappingAwareGaussianUtilityProjector,
    ) -> SelectionResult:
        if not isinstance(grid, ElectrodeGrid):
            raise TypeError("grid must be an ElectrodeGrid")
        if not isinstance(projector, MappingAwareGaussianUtilityProjector):
            raise TypeError(
                "projector must be a MappingAwareGaussianUtilityProjector"
            )
        if not isinstance(image, np.ndarray) or image.dtype != np.float32:
            raise TypeError("image must be a float32 numpy array")
        if image.ndim != 2 or image.shape != grid.image_shape:
            raise EncoderError("image shape differs from the electrode grid")
        if not np.isfinite(image).all() or float(image.min()) < 0.0:
            raise EncoderError("image contains invalid values")
        if float(image.max()) > 1.0:
            raise EncoderError("image values must lie in [0, 1]")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise EncoderError("budget must be a positive integer")
        if budget > grid.candidate_count:
            raise EncoderError("budget exceeds the candidate count")
        if projector.image_shape != grid.image_shape:
            raise EncoderError("projector image shape differs from the grid")
        if projector.candidate_count != grid.candidate_count:
            raise EncoderError("projector candidate count differs from the grid")

        spec = FROZEN_TOPOLOGY_DETECTION
        mask = image >= np.float32(spec.binarization_threshold)
        skeleton = skeletonize(mask, method=spec.skeletonization)
        drive_map = np.ascontiguousarray(skeleton, dtype=np.float32)
        if drive_map.shape != image.shape or float(drive_map.max()) != 1.0:
            raise EncoderError("Zhang skeleton drive map is empty or invalid")
        scores = projector.project(drive_map)
        if float(scores.max()) <= 0.0:
            raise EncoderError("mapped skeleton projection has no positive utility")
        selected = deterministic_topk(scores, budget)
        return SelectionResult(
            encoder_name=self.name,
            budget=budget,
            candidate_count=grid.candidate_count,
            selected_indices=selected,
            candidate_scores=np.ascontiguousarray(scores, dtype=np.float32),
            drive_map=drive_map,
            metadata={
                "mapping_id": projector.mapping_id,
                "binarization_threshold": spec.binarization_threshold,
                "skeletonization": spec.skeletonization,
                "drive_map": "binary_one_pixel_skeleton",
                "candidate_projection": "l1_actual_mapped_gaussian",
                "selection": "deterministic_topk",
                "tie_break": "candidate_index_ascending",
                "topology_enhancement": "none",
            },
        )
