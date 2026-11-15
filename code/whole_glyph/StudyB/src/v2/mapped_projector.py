"""Candidate-utility projection using the target mapping's actual kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.contracts import FloatImage, FloatVector
from src.spv_forward import GaussianPhospheneForwardModel, SpvForwardError


class MappingAwareGaussianUtilityProjector:
    """Project a drive map with the exact kernels used by SPV generation.

    The kernel bank is deliberately borrowed, not reconstructed or copied.  A
    projector and its forward model therefore cannot silently disagree about
    canvas truncation, Gaussian size, or L1 normalization.
    """

    def __init__(self, forward_model: GaussianPhospheneForwardModel) -> None:
        if not isinstance(forward_model, GaussianPhospheneForwardModel):
            raise TypeError(
                "forward_model must be a GaussianPhospheneForwardModel"
            )
        self.mapping_id = forward_model.mapping_id
        self.image_shape = forward_model.mapping.image_shape
        self.candidate_count = forward_model.mapping.candidate_count
        # Do not rebuild or copy this array.  It is the actual forward bank.
        self._kernel_bank = forward_model.kernel_bank

    @property
    def kernel_bank(self) -> NDArray[np.float32]:
        """The forward model's read-only L1-normalized kernel bank."""

        return self._kernel_bank

    def project(self, drive_map: FloatImage) -> FloatVector:
        """Return one non-negative utility per candidate electrode."""

        if not isinstance(drive_map, np.ndarray):
            raise TypeError("drive_map must be a numpy array")
        if drive_map.dtype != np.float32 or drive_map.ndim != 2:
            raise TypeError("drive_map must be a two-dimensional float32 array")
        if drive_map.shape != self.image_shape:
            raise SpvForwardError(
                f"drive_map shape {drive_map.shape} does not match {self.image_shape}"
            )
        if not np.isfinite(drive_map).all():
            raise SpvForwardError("drive_map contains non-finite values")
        if float(drive_map.min()) < 0.0 or float(drive_map.max()) > 1.0:
            raise SpvForwardError("drive_map values must lie in [0, 1]")

        scores = np.tensordot(
            self._kernel_bank,
            drive_map,
            axes=((1, 2), (0, 1)),
        ).astype(np.float32)
        scores = np.ascontiguousarray(scores, dtype=np.float32)
        if scores.shape != (self.candidate_count,):
            raise SpvForwardError("mapped projection returned an invalid shape")
        if not np.isfinite(scores).all() or float(scores.min()) < 0.0:
            raise SpvForwardError("mapped projection returned invalid scores")
        return scores
