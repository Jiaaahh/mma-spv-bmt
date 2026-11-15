"""Overlap construction for actual mapped phosphene kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MappedCoverageError(ValueError):
    """Raised when a mapped kernel bank cannot define a valid overlap."""


def _validate_kernel_bank(kernel_bank: np.ndarray) -> NDArray[np.float64]:
    if not isinstance(kernel_bank, np.ndarray):
        raise TypeError("kernel_bank must be a numpy array")
    if kernel_bank.ndim != 3:
        raise TypeError("kernel_bank must have shape (candidate_count, height, width)")
    if any(size <= 0 for size in kernel_bank.shape):
        raise MappedCoverageError("kernel_bank dimensions must be non-empty")
    if not np.issubdtype(kernel_bank.dtype, np.floating):
        raise TypeError("kernel_bank must use a floating-point dtype")

    kernels = np.ascontiguousarray(kernel_bank, dtype=np.float64)
    if not np.isfinite(kernels).all():
        raise MappedCoverageError("kernel_bank contains non-finite values")
    if float(kernels.min()) < 0.0:
        raise MappedCoverageError("kernel_bank must be non-negative")
    kernel_sums = kernels.sum(axis=(1, 2), dtype=np.float64)
    if not np.allclose(kernel_sums, 1.0, rtol=0.0, atol=1e-5):
        raise MappedCoverageError("every mapped kernel must be L1-normalized")
    return kernels


def build_kernel_overlap_matrix(kernel_bank: np.ndarray) -> NDArray[np.float64]:
    """Build the pairwise Bhattacharyya coefficients of an L1 kernel bank.

    For kernels ``K_i`` and ``K_j``, the returned entry is
    ``sum_p sqrt(K_i[p] * K_j[p])``.  The final symmetrization, clipping, and
    exact unit diagonal make the floating-point postconditions explicit.
    """

    kernels = _validate_kernel_bank(kernel_bank)
    square_roots = np.sqrt(kernels).reshape(kernels.shape[0], -1)
    overlap = square_roots @ square_roots.T
    overlap = np.float64(0.5) * (overlap + overlap.T)
    np.clip(overlap, 0.0, 1.0, out=overlap)
    np.fill_diagonal(overlap, 1.0)
    overlap = np.ascontiguousarray(overlap, dtype=np.float64)

    if not np.isfinite(overlap).all():
        raise MappedCoverageError("mapped overlap contains non-finite values")
    if not np.array_equal(overlap, overlap.T):
        raise MappedCoverageError("mapped overlap symmetrization failed")
    if not np.array_equal(
        np.diag(overlap), np.ones(overlap.shape[0], dtype=np.float64)
    ):
        raise MappedCoverageError("mapped overlap diagonal must equal one")
    overlap.setflags(write=False)
    return overlap
