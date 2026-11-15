"""Fixed-grid Scialom-style Gabor/clock-fragment selector.

This is an explicit fixed-grid adaptation, not an exact reproduction of the
continuous-location human object-recognition paradigm in Scialom et al.  The
adaptation preserves eight-orientation phase-invariant Gabor energy, the 28
clock-hand fragment dictionary, a 100-candidate pool, the 10% overlap rule,
and closest-pair density reduction.  Native-pixel parameters are frozen for
this project's 96x96 image and 16x16 candidate grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from src.contracts import (
    CandidateUtilityProjector,
    ElectrodeGrid,
    FloatImage,
    SelectionResult,
)
from src.encoders.base import BaseEncoder, EncoderError


@dataclass(frozen=True, slots=True)
class ContourSpec:
    orientation_count: int = 8
    gabor_sigma_px: float = 3.0
    gabor_lambda_px: float = 6.0
    clock_length_px: float = 9.0
    clock_sigma_px: float = 3.5
    pool_size: int = 100
    overlap_threshold: float = 0.10
    exclusion_radius_px: float = 5.5
    dispersion_sigma_px: float = 12.0


FROZEN_CONTOUR_SPEC = ContourSpec()


def clock_direction_pairs() -> tuple[tuple[int, int], ...]:
    """Return the 28 unordered pairs of eight 45-degree directions."""

    return tuple(
        (first, second)
        for first in range(8)
        for second in range(first + 1, 8)
    )


def _gabor_kernels(spec: ContourSpec) -> tuple[NDArray[np.complex128], ...]:
    radius = int(ceil(4.0 * spec.gabor_sigma_px))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    envelope = np.exp(-(xx * xx + yy * yy) / (2.0 * spec.gabor_sigma_px**2))
    c0 = np.exp(-2.0 * (pi * spec.gabor_sigma_px / spec.gabor_lambda_px) ** 2)
    kernels: list[NDArray[np.complex128]] = []
    for orientation in range(spec.orientation_count):
        phi = orientation * pi / spec.orientation_count
        coordinate = xx * np.cos(phi) + yy * np.sin(phi)
        phase = 2.0 * pi * coordinate / spec.gabor_lambda_px
        real = envelope * (np.cos(phase) - c0)
        imaginary = envelope * np.sin(phase)
        kernel = real + 1j * imaginary
        norm = float(np.sqrt(np.sum(np.abs(kernel) ** 2)))
        if not np.isfinite(norm) or norm <= 0:
            raise EncoderError("invalid Gabor kernel")
        kernels.append(
            np.ascontiguousarray(kernel / norm, dtype=np.complex128)
        )
    return tuple(kernels)


def _clock_templates(spec: ContourSpec) -> NDArray[np.float64]:
    pairs = clock_direction_pairs()
    radius = int(ceil(spec.clock_length_px + 4.0 * spec.clock_sigma_px))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    templates = np.zeros(
        (len(pairs), spec.orientation_count, axis.size, axis.size),
        dtype=np.float64,
    )
    for template_index, pair in enumerate(pairs):
        for direction in pair:
            angle = direction * pi / 4.0
            unit_x, unit_y = np.cos(angle), np.sin(angle)
            projection = np.clip(
                xx * unit_x + yy * unit_y,
                0.0,
                spec.clock_length_px,
            )
            distance = np.sqrt(
                (xx - projection * unit_x) ** 2
                + (yy - projection * unit_y) ** 2
            )
            tube = np.exp(-(distance**2) / (2.0 * spec.clock_sigma_px**2))
            tube[distance > 4.0 * spec.clock_sigma_px] = 0.0
            # Gabor orientation planes span 0..157.5 degrees in 22.5-degree
            # steps.  Clock directions span 0..315 in 45-degree steps and are
            # phase-independent, so antipodal hands share the same plane.
            orientation_plane = (2 * direction) % spec.orientation_count
            templates[template_index, orientation_plane] = np.maximum(
                templates[template_index, orientation_plane], tube
            )
    sums = templates.sum(axis=(1, 2, 3), dtype=np.float64)
    if np.any(sums <= 0) or not np.isfinite(templates).all():
        raise EncoderError("invalid clock-fragment dictionary")
    templates /= sums[:, None, None, None]
    return np.ascontiguousarray(templates, dtype=np.float64)


def _nearest_pair_reduction(
    pool_indices: NDArray[np.int64],
    scores: NDArray[np.float64],
    centers_xy: NDArray[np.float64],
    budget: int,
) -> NDArray[np.int64]:
    """Apply deterministic closest-pair pruning to the scored pool."""

    active = list(map(int, pool_indices))
    while len(active) > budget:
        best: tuple[float, int, int] | None = None
        for left_pos in range(len(active) - 1):
            left = active[left_pos]
            for right_pos in range(left_pos + 1, len(active)):
                right = active[right_pos]
                distance2 = float(
                    np.sum((centers_xy[left] - centers_xy[right]) ** 2)
                )
                key = (distance2, min(left, right), max(left, right))
                if best is None or key < best:
                    best = key
        if best is None:
            raise EncoderError("closest-pair reduction failed")
        left, right = best[1], best[2]
        if scores[left] < scores[right]:
            remove = left
        elif scores[right] < scores[left]:
            remove = right
        else:
            remove = max(left, right)
        active.remove(remove)
    return np.asarray(sorted(active), dtype=np.int64)


class ScialomStyleGridContourEncoder(BaseEncoder):
    """Select electrodes by contour/fragment overlap and spatial dispersion."""

    name = "scialom_gabor_clock_grid_b64"

    def __init__(self, spec: ContourSpec = FROZEN_CONTOUR_SPEC) -> None:
        if spec.orientation_count != 8 or spec.pool_size != 100:
            raise EncoderError(
                "E1 contour encoder requires 8 orientations and pool_size=100"
            )
        self.spec = spec
        self._gabors = _gabor_kernels(spec)
        self._templates = _clock_templates(spec)

    @property
    def template_count(self) -> int:
        return int(self._templates.shape[0])

    def _orientation_energy(self, image: FloatImage) -> NDArray[np.float64]:
        # The source study used dark content on a white background.  The
        # project's white-on-black glyph is explicitly inverted, and the
        # convolution boundary is padded white before cropping.
        source = 1.0 - image.astype(np.float64, copy=False)
        radius = self._gabors[0].shape[0] // 2
        padded = np.pad(source, radius, mode="constant", constant_values=1.0)
        energy = np.empty((8, *source.shape), dtype=np.float64)
        for index, kernel in enumerate(self._gabors):
            response = fftconvolve(padded, kernel, mode="same")
            cropped = response[radius:-radius, radius:-radius]
            energy[index] = np.abs(cropped)
        if not np.isfinite(energy).all() or float(energy.min()) < 0.0:
            raise EncoderError("Gabor energy is invalid")
        return energy

    def _candidate_overlap(
        self,
        energy: NDArray[np.float64],
        centers_xy: NDArray[np.float32],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        radius = self._templates.shape[-1] // 2
        padded = np.pad(
            energy,
            ((0, 0), (radius, radius), (radius, radius)),
            mode="constant",
        )
        scores = np.empty(centers_xy.shape[0], dtype=np.float64)
        best_templates = np.empty(centers_xy.shape[0], dtype=np.int64)
        for candidate_index, (x_coord, y_coord) in enumerate(centers_xy):
            center_x = int(round(float(x_coord))) + radius
            center_y = int(round(float(y_coord))) + radius
            patch = padded[
                :,
                center_y - radius : center_y + radius + 1,
                center_x - radius : center_x + radius + 1,
            ]
            overlaps = np.einsum(
                "tohw,ohw->t", self._templates, patch, optimize=True
            )
            winner = int(np.argmax(overlaps))
            scores[candidate_index] = float(overlaps[winner])
            best_templates[candidate_index] = winner
        if not np.isfinite(scores).all() or float(scores.max()) <= 0.0:
            raise EncoderError("contour overlap produced no finite positive score")
        return scores, best_templates

    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: CandidateUtilityProjector,
    ) -> SelectionResult:
        self.validate_inputs(image, grid, budget, projector)
        if budget != 64 or grid.candidate_count != 256 or image.shape != (96, 96):
            raise EncoderError(
                "Scialom-style E1 adaptation is frozen to 96x96/256/B=64"
            )
        energy = self._orientation_energy(image)
        scores, best_templates = self._candidate_overlap(energy, grid.centers_xy)

        mask = np.ones(grid.candidate_count, dtype=np.float64)
        selected_pool: list[int] = []
        centers = grid.centers_xy.astype(np.float64, copy=False)
        for _ in range(self.spec.pool_size):
            eligible = np.ones(grid.candidate_count, dtype=bool)
            if selected_pool:
                eligible[np.asarray(selected_pool, dtype=np.int64)] = False
            objective = scores * mask
            objective[~eligible] = -np.inf
            candidate = int(np.argmax(objective))
            if not np.isfinite(objective[candidate]):
                raise EncoderError(
                    "dispersion pool could not reach 100 unique candidates"
                )
            selected_pool.append(candidate)
            distance = np.linalg.norm(centers - centers[candidate], axis=1)
            factor = 1.0 - np.exp(
                -(distance**2) / (2.0 * self.spec.dispersion_sigma_px**2)
            )
            factor[distance < self.spec.exclusion_radius_px] = 0.0
            mask *= factor
            mask[np.asarray(selected_pool, dtype=np.int64)] = 0.0

        pool = np.asarray(selected_pool, dtype=np.int64)
        threshold = self.spec.overlap_threshold * float(np.max(scores[pool]))
        valid = pool[scores[pool] >= threshold]
        backfill_count = 0
        if valid.size < budget:
            valid_set = set(map(int, valid))
            remaining = [
                int(index) for index in pool if int(index) not in valid_set
            ]
            remaining.sort(key=lambda index: (-scores[index], index))
            backfill_count = budget - int(valid.size)
            valid = np.concatenate(
                (valid, np.asarray(remaining[:backfill_count], dtype=np.int64))
            )
        selected = _nearest_pair_reduction(valid, scores, centers, budget)
        if selected.shape != (budget,) or np.unique(selected).size != budget:
            raise EncoderError(
                "density reduction did not produce exactly B unique candidates"
            )

        drive = np.max(energy, axis=0)
        drive /= float(drive.max())
        drive_map = np.ascontiguousarray(drive, dtype=np.float32)
        candidate_scores = np.ascontiguousarray(scores, dtype=np.float32)
        return SelectionResult(
            encoder_name=self.name,
            budget=budget,
            candidate_count=grid.candidate_count,
            selected_indices=selected,
            candidate_scores=candidate_scores,
            drive_map=drive_map,
            metadata={
                "adaptation": "fixed_grid_not_exact_reproduction",
                "orientation_count": 8,
                "clock_template_count": self.template_count,
                "pool_size": self.spec.pool_size,
                "overlap_threshold": self.spec.overlap_threshold,
                "threshold_backfill_count": backfill_count,
                "best_template_indices": best_templates.tolist(),
                "pool_order": pool.tolist(),
                "selection": (
                    "overlap_dispersion_then_closest_pair_reduction"
                ),
                "tie_break": "candidate_index_ascending",
                "label_access": "none",
                "mapping_geometry_access": "ideal_only",
                "amplitude": "unit",
            },
        )
