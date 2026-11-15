"""Ideal and fixed-seed moderate phosphene mappings."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from src.contracts import ElectrodeMapping
from src.grid import RegularGrid


class MappingError(ValueError):
    """Raised when a phosphene mapping parameter or output is invalid."""


@dataclass(frozen=True, slots=True)
class MappingStatistics:
    mean_displacement_px: float
    max_displacement_px: float
    mean_sigma_px: float
    min_sigma_px: float
    max_sigma_px: float


def _base_sigma_px(grid: RegularGrid, base_sigma_grid: float) -> float:
    if (
        isinstance(base_sigma_grid, bool)
        or not isinstance(base_sigma_grid, (int, float))
        or base_sigma_grid <= 0
    ):
        raise MappingError("base_sigma_grid must be positive")
    spacing_x, spacing_y = grid.spacing_xy
    return float(base_sigma_grid) * sqrt(spacing_x * spacing_y)


def make_ideal_mapping(
    grid: RegularGrid,
    *,
    base_sigma_grid: float = 0.45,
) -> ElectrodeMapping:
    """Create one circular Gaussian phosphene at every ideal grid center."""

    sigma = _base_sigma_px(grid, base_sigma_grid)
    centers = np.array(grid.centers_xy, dtype=np.float32, copy=True)
    sigma_px = np.full(grid.candidate_count, sigma, dtype=np.float32)
    return ElectrodeMapping(
        mapping_id="ideal",
        seed=0,
        centers_xy=centers,
        sigma_px=sigma_px,
        image_shape=grid.image_shape,
    )


def make_moderate_mapping(
    grid: RegularGrid,
    *,
    seed: int,
    base_sigma_grid: float = 0.45,
    position_jitter_std_grid: float = 0.30,
    sigma_scale_std: float = 0.20,
    sigma_scale_clip: tuple[float, float] = (0.60, 1.60),
) -> ElectrodeMapping:
    """Perturb phosphene position and size with one deterministic NumPy seed."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MappingError("seed must be a non-negative integer")
    for name, value in (
        ("position_jitter_std_grid", position_jitter_std_grid),
        ("sigma_scale_std", sigma_scale_std),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise MappingError(f"{name} must be non-negative")
    if (
        not isinstance(sigma_scale_clip, tuple)
        or len(sigma_scale_clip) != 2
        or not 0 < sigma_scale_clip[0] <= sigma_scale_clip[1]
    ):
        raise MappingError("sigma_scale_clip must contain two ordered positive values")

    rng = np.random.default_rng(seed)
    spacing_x, spacing_y = grid.spacing_xy
    offsets = np.empty((grid.candidate_count, 2), dtype=np.float32)
    offsets[:, 0] = rng.normal(
        loc=0.0,
        scale=float(position_jitter_std_grid) * spacing_x,
        size=grid.candidate_count,
    ).astype(np.float32)
    offsets[:, 1] = rng.normal(
        loc=0.0,
        scale=float(position_jitter_std_grid) * spacing_y,
        size=grid.candidate_count,
    ).astype(np.float32)
    centers = np.asarray(grid.centers_xy + offsets, dtype=np.float32)
    height, width = grid.image_shape
    centers[:, 0] = np.clip(centers[:, 0], 0.0, float(width - 1))
    centers[:, 1] = np.clip(centers[:, 1], 0.0, float(height - 1))

    scales = rng.normal(
        loc=1.0,
        scale=float(sigma_scale_std),
        size=grid.candidate_count,
    )
    scales = np.clip(scales, sigma_scale_clip[0], sigma_scale_clip[1])
    sigma_px = np.asarray(
        _base_sigma_px(grid, base_sigma_grid) * scales, dtype=np.float32
    )
    mapping = ElectrodeMapping(
        mapping_id=f"moderate_seed{seed}",
        seed=seed,
        centers_xy=centers,
        sigma_px=sigma_px,
        image_shape=grid.image_shape,
    )
    if np.array_equal(mapping.centers_xy, grid.centers_xy):
        raise MappingError("moderate mapping unexpectedly equals the ideal grid")
    return mapping


def mapping_statistics(
    ideal: ElectrodeMapping, perturbed: ElectrodeMapping
) -> MappingStatistics:
    if ideal.image_shape != perturbed.image_shape:
        raise MappingError("mapping image shapes differ")
    if ideal.candidate_count != perturbed.candidate_count:
        raise MappingError("mapping candidate counts differ")
    displacement = np.linalg.norm(
        perturbed.centers_xy.astype(np.float64) - ideal.centers_xy.astype(np.float64),
        axis=1,
    )
    sigma = perturbed.sigma_px.astype(np.float64)
    return MappingStatistics(
        mean_displacement_px=float(displacement.mean()),
        max_displacement_px=float(displacement.max()),
        mean_sigma_px=float(sigma.mean()),
        min_sigma_px=float(sigma.min()),
        max_sigma_px=float(sigma.max()),
    )
