"""Regular candidate-electrode grid generation for the SPV forward model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt

import numpy as np
from numpy.typing import NDArray

from src.contracts import ElectrodeGrid


class GridError(ValueError):
    """Raised when candidate-grid geometry is invalid."""


@dataclass(frozen=True, slots=True)
class RegularGrid:
    """Row-major regular grid plus its exact pixel spacing."""

    electrodes: ElectrodeGrid
    rows: int
    cols: int
    spacing_xy: tuple[float, float]

    @property
    def candidate_count(self) -> int:
        return self.electrodes.candidate_count

    @property
    def centers_xy(self) -> NDArray[np.float32]:
        return self.electrodes.centers_xy

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.electrodes.image_shape


def make_regular_grid(
    *,
    rows: int = 16,
    cols: int = 16,
    image_shape: tuple[int, int] = (96, 96),
) -> RegularGrid:
    """Place one candidate at the center of every regular grid cell."""

    for name, value in (("rows", rows), ("cols", cols)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GridError(f"{name} must be a positive integer")
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in image_shape
        )
    ):
        raise GridError("image_shape must contain two positive integers")

    height, width = image_shape
    spacing_x = width / cols
    spacing_y = height / rows
    x_centers = (np.arange(cols, dtype=np.float32) + np.float32(0.5)) * np.float32(
        spacing_x
    )
    y_centers = (np.arange(rows, dtype=np.float32) + np.float32(0.5)) * np.float32(
        spacing_y
    )
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    centers = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    electrodes = ElectrodeGrid(centers_xy=centers, image_shape=image_shape)
    regular = RegularGrid(
        electrodes=electrodes,
        rows=rows,
        cols=cols,
        spacing_xy=(float(spacing_x), float(spacing_y)),
    )
    validate_regular_grid(regular)
    return regular


def validate_regular_grid(grid: RegularGrid) -> None:
    if grid.candidate_count != grid.rows * grid.cols:
        raise GridError("candidate count does not equal rows*cols")
    height, width = grid.image_shape
    expected_x = (np.arange(grid.cols, dtype=np.float32) + 0.5) * (
        width / grid.cols
    )
    expected_y = (np.arange(grid.rows, dtype=np.float32) + 0.5) * (
        height / grid.rows
    )
    xx, yy = np.meshgrid(expected_x, expected_y, indexing="xy")
    expected = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    if not np.allclose(grid.centers_xy, expected, rtol=0.0, atol=1e-6):
        raise GridError("candidate centers are not in deterministic row-major order")
    expected_spacing = (width / grid.cols, height / grid.rows)
    if not np.allclose(grid.spacing_xy, expected_spacing, rtol=0.0, atol=1e-12):
        raise GridError("stored grid spacing is inconsistent with image geometry")


def make_uniform_diagnostic_indices(grid: RegularGrid, budget: int) -> NDArray[np.int64]:
    """Return a fixed image-independent subset used only to test the forward model.

    This is deliberately not an encoding strategy: every input image receives
    the same uniformly distributed indices.
    """

    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise GridError("budget must be a positive integer")
    side = isqrt(budget)
    if side * side != budget:
        raise GridError("diagnostic budget must be a perfect square")
    if side > grid.rows or side > grid.cols:
        raise GridError("diagnostic budget does not fit the candidate grid")
    row_positions = np.rint(np.linspace(0, grid.rows - 1, side)).astype(np.int64)
    col_positions = np.rint(np.linspace(0, grid.cols - 1, side)).astype(np.int64)
    if np.unique(row_positions).size != side or np.unique(col_positions).size != side:
        raise GridError("could not construct unique diagnostic rows and columns")
    indices = np.asarray(
        [row * grid.cols + col for row in row_positions for col in col_positions],
        dtype=np.int64,
    )
    if indices.shape != (budget,) or np.unique(indices).size != budget:
        raise GridError("diagnostic indices are not unique or do not match budget")
    return indices
