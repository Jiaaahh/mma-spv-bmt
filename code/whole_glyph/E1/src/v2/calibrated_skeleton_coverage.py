"""Calibration-aware Zhang-skeleton coverage encoder."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize

from src.contracts import (
    ElectrodeGrid,
    FROZEN_STAGE7_COVERAGE,
    FROZEN_TOPOLOGY_DETECTION,
    FloatImage,
    FloatVector,
    SelectionResult,
)
from src.coverage import CoverageGreedyTrace, greedy_coverage_order
from src.encoders.base import EncoderError
from src.spv_forward import GaussianPhospheneForwardModel
from src.v2.mapped_coverage import build_kernel_overlap_matrix
from src.v2.mapped_projector import MappingAwareGaussianUtilityProjector


@dataclass(frozen=True, slots=True)
class CalibratedSkeletonOrderResult:
    """The one mapped 128-step order shared by every supported budget."""

    mapping_id: str
    drive_map: FloatImage = field(repr=False)
    candidate_scores: FloatVector = field(repr=False)
    trace: CoverageGreedyTrace = field(repr=False)

    def __post_init__(self) -> None:
        if not self.mapping_id:
            raise ValueError("mapping_id must not be empty")
        if self.drive_map.dtype != np.float32 or self.drive_map.ndim != 2:
            raise TypeError("drive_map must be a two-dimensional float32 array")
        if not np.all((self.drive_map == 0.0) | (self.drive_map == 1.0)):
            raise ValueError("drive_map must be a binary skeleton")
        if self.candidate_scores.dtype != np.float32 or self.candidate_scores.ndim != 1:
            raise TypeError("candidate_scores must be a one-dimensional float32 array")
        if self.trace.final_residual.shape != self.candidate_scores.shape:
            raise ValueError("coverage residual and candidate utility shapes differ")

    @property
    def order(self) -> NDArray[np.int64]:
        """Convenience alias for the auditable greedy trace order."""

        return self.trace.order


class CalibrationAwareSkeletonCoverageEncoder:
    """Select mapped electrodes by Zhang-skeleton utility and kernel coverage."""

    name = "calibration_aware_skeleton_coverage"

    def __init__(
        self,
        forward_model: GaussianPhospheneForwardModel,
        *,
        max_order: int = FROZEN_STAGE7_COVERAGE.max_order_length,
    ) -> None:
        if not isinstance(forward_model, GaussianPhospheneForwardModel):
            raise TypeError(
                "forward_model must be a GaussianPhospheneForwardModel"
            )
        if isinstance(max_order, bool) or not isinstance(max_order, int):
            raise TypeError("max_order must be an integer")
        if max_order != FROZEN_STAGE7_COVERAGE.max_order_length:
            raise EncoderError(
                "calibration-aware coverage must generate one 128-step order"
            )
        if forward_model.mapping.candidate_count < max_order:
            raise EncoderError("max_order exceeds the mapped candidate count")

        self.mapping_id = forward_model.mapping_id
        self.image_shape = forward_model.mapping.image_shape
        self.candidate_count = forward_model.mapping.candidate_count
        self.max_order = max_order
        self._kernel_bank = forward_model.kernel_bank
        self._overlap_matrix = build_kernel_overlap_matrix(self._kernel_bank)

    @property
    def overlap_matrix(self) -> NDArray[np.float64]:
        """Read-only Bhattacharyya overlap cached once for this mapping."""

        return self._overlap_matrix

    def _validate_inputs(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        projector: MappingAwareGaussianUtilityProjector,
    ) -> None:
        if not isinstance(grid, ElectrodeGrid):
            raise TypeError("grid must be an ElectrodeGrid")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if image.dtype != np.float32 or image.ndim != 2:
            raise TypeError("image must be a two-dimensional float32 array")
        if image.shape != grid.image_shape:
            raise EncoderError(
                f"image shape {image.shape} does not match grid shape {grid.image_shape}"
            )
        if not np.isfinite(image).all():
            raise EncoderError("image contains non-finite values")
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise EncoderError("image values must lie in [0, 1]")
        if not isinstance(projector, MappingAwareGaussianUtilityProjector):
            raise TypeError(
                "projector must be a MappingAwareGaussianUtilityProjector"
            )
        if projector.mapping_id != self.mapping_id:
            raise EncoderError(
                "projector mapping_id differs from the encoder forward mapping"
            )
        if projector.image_shape != self.image_shape:
            raise EncoderError("projector image shape differs from the forward mapping")
        if projector.candidate_count != self.candidate_count:
            raise EncoderError(
                "projector candidate count differs from the forward mapping"
            )
        if grid.image_shape != self.image_shape:
            raise EncoderError("electrode grid shape differs from the forward mapping")
        if grid.candidate_count != self.candidate_count:
            raise EncoderError(
                "electrode grid candidate count differs from the forward mapping"
            )

    @staticmethod
    def _build_skeleton(image: FloatImage) -> FloatImage:
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
        return drive_map

    def encode_full_order(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        *,
        projector: MappingAwareGaussianUtilityProjector,
    ) -> CalibratedSkeletonOrderResult:
        """Build one deterministic 128-candidate mapped coverage order."""

        self._validate_inputs(image, grid, projector)
        drive_map = self._build_skeleton(image)
        raw_scores = projector.project(drive_map)
        skeleton_pixel_count = float(drive_map.sum(dtype=np.float64))
        if skeleton_pixel_count <= 0.0:
            raise EncoderError("Zhang skeleton drive map is empty")
        scores = np.ascontiguousarray(
            raw_scores / np.float32(skeleton_pixel_count), dtype=np.float32
        )
        if scores.shape != (self.candidate_count,) or not np.isfinite(scores).all():
            raise EncoderError("mapped skeleton scoring produced an invalid array")
        if float(scores.max()) <= 0.0:
            raise EncoderError("mapped skeleton projection produced no positive utility")

        trace = greedy_coverage_order(
            scores,
            self._overlap_matrix,
            max_steps=self.max_order,
        )
        return CalibratedSkeletonOrderResult(
            mapping_id=self.mapping_id,
            drive_map=drive_map,
            candidate_scores=scores,
            trace=trace,
        )

    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: MappingAwareGaussianUtilityProjector,
    ) -> SelectionResult:
        """Return a supported budget as a prefix of the full mapped order."""

        spec = FROZEN_STAGE7_COVERAGE
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError("budget must be an integer")
        if budget not in spec.budgets:
            raise EncoderError(
                f"calibration-aware coverage budget must be one of {spec.budgets}"
            )
        full = self.encode_full_order(image, grid, projector=projector)
        return SelectionResult(
            encoder_name=self.name,
            budget=budget,
            candidate_count=self.candidate_count,
            selected_indices=np.ascontiguousarray(
                full.trace.order[:budget], dtype=np.int64
            ),
            candidate_scores=full.candidate_scores,
            drive_map=full.drive_map,
            metadata={
                "mapping_id": self.mapping_id,
                "binarization_threshold": FROZEN_TOPOLOGY_DETECTION.binarization_threshold,
                "skeletonization": FROZEN_TOPOLOGY_DETECTION.skeletonization,
                "drive_map": "binary_one_pixel_skeleton",
                "utility_normalization": "divide_by_skeleton_pixel_count",
                "candidate_projection": "l1_actual_mapped_gaussian",
                "overlap": "bhattacharyya_coefficient_actual_kernel_bank",
                "selection": "redundancy_aware_coverage_greedy",
                "tie_break": spec.tie_break,
                "greedy_order_length": self.max_order,
                "budget_prefix": budget,
                "objective_at_budget": float(full.trace.objective_values[budget]),
                "theoretical_optimality_claim": spec.theoretical_optimality_claim,
                "topology_enhancement": "none",
            },
        )
