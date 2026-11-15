"""Frozen Stage-5.5 redundancy-aware coverage objective and greedy selector."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.contracts import ElectrodeGrid, FROZEN_STAGE7_COVERAGE


class CoverageError(ValueError):
    """Raised when coverage inputs or greedy state violate the frozen protocol."""


Float64Vector = NDArray[np.float64]
Float64Matrix = NDArray[np.float64]
Int64Vector = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CoverageGreedyTrace:
    """Full greedy order plus auditable objective and residual trajectories."""

    order: Int64Vector = field(repr=False)
    objective_values: Float64Vector = field(repr=False)
    marginal_gains: Float64Vector = field(repr=False)
    final_residual: Float64Vector = field(repr=False)

    def __post_init__(self) -> None:
        if self.order.dtype != np.int64 or self.order.ndim != 1:
            raise TypeError("coverage order must be a one-dimensional int64 array")
        for name in ("objective_values", "marginal_gains", "final_residual"):
            value = getattr(self, name)
            if value.dtype != np.float64 or value.ndim != 1:
                raise TypeError(f"{name} must be a one-dimensional float64 array")
            if not np.isfinite(value).all():
                raise CoverageError(f"{name} contains non-finite values")
        if self.objective_values.shape != (self.order.size + 1,):
            raise CoverageError("objective trace must contain initial value plus every step")
        if self.marginal_gains.shape != self.order.shape:
            raise CoverageError("marginal gain trace must match the order length")
        if np.unique(self.order).size != self.order.size:
            raise CoverageError("coverage order contains duplicate candidates")
        if np.any(self.order < 0) or np.any(self.order >= self.final_residual.size):
            raise CoverageError("coverage order contains an illegal candidate index")
        if float(self.objective_values[0]) != 0.0:
            raise CoverageError("coverage objective must start at zero")
        if np.any(np.diff(self.objective_values) < -1e-12):
            raise CoverageError("coverage objective decreased during greedy selection")
        if np.any(self.marginal_gains < -1e-12):
            raise CoverageError("coverage marginal gain must not be negative")
        if np.any(self.final_residual < -1e-15) or np.any(
            self.final_residual > 1.0 + 1e-15
        ):
            raise CoverageError("coverage residual must lie in [0, 1]")


def build_ideal_overlap_matrix(grid: ElectrodeGrid) -> Float64Matrix:
    """Build frozen R on nominal 16x16 geometry, separate from the projector."""

    if not isinstance(grid, ElectrodeGrid):
        raise TypeError("grid must be an ElectrodeGrid")
    spec = FROZEN_STAGE7_COVERAGE
    if grid.image_shape != (96, 96) or grid.candidate_count != 256:
        raise CoverageError("Stage-7 coverage requires the frozen 96x96/256 grid")
    expected_axis = (
        np.arange(16, dtype=np.float64) + 0.5
    ) * spec.d_e_px
    xx, yy = np.meshgrid(expected_axis, expected_axis, indexing="xy")
    expected_centers = np.column_stack((xx.ravel(), yy.ravel()))
    centers = grid.centers_xy.astype(np.float64, copy=False)
    if not np.allclose(centers, expected_centers, rtol=0.0, atol=1e-12):
        raise CoverageError(
            "coverage must use the frozen nominal row-major ideal candidate geometry"
        )
    delta = centers[:, None, :] - centers[None, :, :]
    squared_distance = np.sum(delta * delta, axis=2, dtype=np.float64)
    inside = squared_distance <= spec.cutoff_px**2
    overlap = np.zeros((grid.candidate_count, grid.candidate_count), dtype=np.float64)
    overlap[inside] = np.exp(
        -squared_distance[inside] / (2.0 * spec.sigma_r_px**2)
    )
    np.fill_diagonal(overlap, spec.diagonal)
    overlap = np.ascontiguousarray(overlap, dtype=np.float64)
    _validate_overlap_matrix(overlap)
    overlap.setflags(write=False)
    return overlap


def _validate_utility(candidate_utility: np.ndarray) -> Float64Vector:
    if not isinstance(candidate_utility, np.ndarray) or candidate_utility.ndim != 1:
        raise TypeError("candidate_utility must be a one-dimensional numpy array")
    if candidate_utility.size == 0:
        raise CoverageError("candidate_utility must not be empty")
    utility = np.ascontiguousarray(candidate_utility, dtype=np.float64)
    if not np.isfinite(utility).all() or float(utility.min()) < 0.0:
        raise CoverageError("candidate_utility must be finite and non-negative")
    return utility


def _validate_overlap_matrix(overlap: np.ndarray) -> Float64Matrix:
    if not isinstance(overlap, np.ndarray) or overlap.ndim != 2:
        raise TypeError("overlap must be a two-dimensional numpy array")
    if overlap.shape[0] == 0 or overlap.shape[0] != overlap.shape[1]:
        raise CoverageError("overlap must be a non-empty square matrix")
    matrix = np.ascontiguousarray(overlap, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise CoverageError("overlap contains non-finite values")
    if float(matrix.min()) < 0.0 or float(matrix.max()) > 1.0:
        raise CoverageError("overlap values must lie in [0, 1]")
    if not np.array_equal(np.diag(matrix), np.ones(matrix.shape[0], dtype=np.float64)):
        raise CoverageError("overlap diagonal must equal one exactly")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15):
        raise CoverageError("ideal-geometry overlap matrix must be symmetric")
    return matrix


def _validate_problem(
    candidate_utility: np.ndarray,
    overlap: np.ndarray,
    max_steps: int,
) -> tuple[Float64Vector, Float64Matrix, int]:
    utility = _validate_utility(candidate_utility)
    matrix = _validate_overlap_matrix(overlap)
    if matrix.shape != (utility.size, utility.size):
        raise CoverageError("utility and overlap candidate counts differ")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise CoverageError("max_steps must be a positive integer")
    if max_steps > utility.size:
        raise CoverageError("max_steps must not exceed candidate count")
    return utility, matrix, max_steps


def _validate_selected(selected: tuple[int, ...], candidate_count: int) -> None:
    if len(set(selected)) != len(selected):
        raise CoverageError("selected candidates must be unique")
    if any(
        isinstance(index, bool)
        or not isinstance(index, (int, np.integer))
        or int(index) < 0
        or int(index) >= candidate_count
        for index in selected
    ):
        raise CoverageError("selected candidates contain an illegal index")


def coverage_objective(
    candidate_utility: np.ndarray,
    overlap: np.ndarray,
    selected: tuple[int, ...] | list[int] | np.ndarray,
) -> float:
    """Directly evaluate F_D(A) from the frozen product definition."""

    utility = _validate_utility(candidate_utility)
    matrix = _validate_overlap_matrix(overlap)
    if matrix.shape != (utility.size, utility.size):
        raise CoverageError("utility and overlap candidate counts differ")
    if isinstance(selected, np.ndarray):
        if selected.ndim != 1:
            raise TypeError("selected array must be one-dimensional")
        selected_tuple = tuple(map(int, selected.tolist()))
    elif isinstance(selected, (tuple, list)):
        selected_tuple = tuple(selected)
    else:
        raise TypeError("selected must be a tuple, list, or numpy array")
    _validate_selected(selected_tuple, utility.size)
    residual = np.ones(utility.size, dtype=np.float64)
    for candidate in selected_tuple:
        residual *= 1.0 - matrix[int(candidate)]
    return float(np.dot(utility, 1.0 - residual))


def greedy_coverage_order(
    candidate_utility: np.ndarray,
    overlap: np.ndarray,
    *,
    max_steps: int = 128,
) -> CoverageGreedyTrace:
    """Optimized state-update implementation of the frozen greedy gain."""

    utility, matrix, steps = _validate_problem(
        candidate_utility, overlap, max_steps
    )
    candidate_count = utility.size
    residual = np.ones(candidate_count, dtype=np.float64)
    selected_mask = np.zeros(candidate_count, dtype=np.bool_)
    order = np.empty(steps, dtype=np.int64)
    objective_values = np.empty(steps + 1, dtype=np.float64)
    marginal_gains = np.empty(steps, dtype=np.float64)
    objective_values[0] = 0.0

    for step in range(steps):
        # Row k is R_ki and residual i is prod_{j in A}(1-R_ji).
        gains = matrix @ (utility * residual)
        gains[selected_mask] = -np.inf
        # np.argmax returns the first maximum, implementing ascending-index ties.
        candidate = int(np.argmax(gains))
        gain = float(gains[candidate])
        if not np.isfinite(gain) or gain < -1e-12:
            raise CoverageError("greedy selection produced an invalid marginal gain")
        order[step] = candidate
        selected_mask[candidate] = True
        residual *= 1.0 - matrix[candidate]
        objective = float(np.dot(utility, 1.0 - residual))
        direct_increment = objective - float(objective_values[step])
        if not np.isclose(gain, direct_increment, rtol=1e-11, atol=1e-12):
            raise CoverageError(
                "optimized marginal gain differs from the objective increment"
            )
        marginal_gains[step] = gain
        objective_values[step + 1] = objective

    return CoverageGreedyTrace(
        order=np.ascontiguousarray(order, dtype=np.int64),
        objective_values=np.ascontiguousarray(objective_values, dtype=np.float64),
        marginal_gains=np.ascontiguousarray(marginal_gains, dtype=np.float64),
        final_residual=np.ascontiguousarray(residual, dtype=np.float64),
    )


def reference_greedy_order(
    candidate_utility: np.ndarray,
    overlap: np.ndarray,
    *,
    max_steps: int,
) -> CoverageGreedyTrace:
    """Slow direct reference: enumerate every unselected candidate via F(A∪k)."""

    utility, matrix, steps = _validate_problem(
        candidate_utility, overlap, max_steps
    )
    selected: list[int] = []
    selected_mask = np.zeros(utility.size, dtype=np.bool_)
    order = np.empty(steps, dtype=np.int64)
    objective_values = np.empty(steps + 1, dtype=np.float64)
    marginal_gains = np.empty(steps, dtype=np.float64)
    objective_values[0] = 0.0

    for step in range(steps):
        base = coverage_objective(utility, matrix, tuple(selected))
        best_candidate = -1
        best_gain = -np.inf
        for candidate in range(utility.size):
            if selected_mask[candidate]:
                continue
            trial = tuple((*selected, candidate))
            gain = coverage_objective(utility, matrix, trial) - base
            # Strict greater-than preserves the first, smallest index on ties.
            if gain > best_gain:
                best_gain = gain
                best_candidate = candidate
        if best_candidate < 0 or not np.isfinite(best_gain):
            raise CoverageError("reference greedy failed to choose a candidate")
        selected.append(best_candidate)
        selected_mask[best_candidate] = True
        order[step] = best_candidate
        marginal_gains[step] = best_gain
        objective_values[step + 1] = coverage_objective(
            utility, matrix, tuple(selected)
        )

    residual = np.ones(utility.size, dtype=np.float64)
    for candidate in selected:
        residual *= 1.0 - matrix[candidate]
    return CoverageGreedyTrace(
        order=np.ascontiguousarray(order, dtype=np.int64),
        objective_values=np.ascontiguousarray(objective_values, dtype=np.float64),
        marginal_gains=np.ascontiguousarray(marginal_gains, dtype=np.float64),
        final_residual=np.ascontiguousarray(residual, dtype=np.float64),
    )
