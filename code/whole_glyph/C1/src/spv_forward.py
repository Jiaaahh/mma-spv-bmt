"""Unified circular-Gaussian phosphene forward model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.contracts import (
    ElectrodeGrid,
    ElectrodeMapping,
    FloatImage,
    FloatVector,
    IndexVector,
)
from src.grid import RegularGrid


class SpvForwardError(ValueError):
    """Raised when forward-model inputs or outputs violate the frozen contract."""


def _validate_indices(
    selected_indices: IndexVector, candidate_count: int
) -> NDArray[np.int64]:
    if not isinstance(selected_indices, np.ndarray):
        raise TypeError("selected_indices must be a numpy array")
    if selected_indices.dtype != np.int64 or selected_indices.ndim != 1:
        raise TypeError("selected_indices must be a one-dimensional int64 array")
    if selected_indices.size == 0:
        raise SpvForwardError("selected_indices must not be empty")
    if np.unique(selected_indices).size != selected_indices.size:
        raise SpvForwardError("selected_indices must be unique")
    if np.any(selected_indices < 0) or np.any(selected_indices >= candidate_count):
        raise SpvForwardError("selected_indices contains an out-of-range value")
    return selected_indices


def _validate_amplitudes(
    amplitudes: np.ndarray, expected_shape: tuple[int, ...]
) -> NDArray[np.float32]:
    if not isinstance(amplitudes, np.ndarray):
        raise TypeError("amplitudes must be a numpy array")
    if amplitudes.dtype != np.float32 or amplitudes.shape != expected_shape:
        raise TypeError(
            f"amplitudes must be float32 with shape {expected_shape}, got "
            f"{amplitudes.dtype}/{amplitudes.shape}"
        )
    if amplitudes.size == 0:
        raise SpvForwardError("amplitudes must not be empty")
    if not np.isfinite(amplitudes).all():
        raise SpvForwardError("amplitudes contains non-finite values")
    if float(amplitudes.min()) < 0.0 or float(amplitudes.max()) > 1.0:
        raise SpvForwardError("amplitudes must lie in [0, 1]")
    return amplitudes


class GaussianPhospheneForwardModel:
    """Precomputed L1-normalized Gaussian kernels for one fixed mapping.

    Kernel normalization is performed after canvas truncation, so each active
    electrode contributes exactly ``amplitude_scale * amplitude`` total image
    energy. The numeric output is never clipped or normalized globally.
    """

    def __init__(
        self,
        mapping: ElectrodeMapping,
        *,
        amplitude_scale: float = 1.0,
        kernel_normalization: str = "l1",
    ) -> None:
        if (
            isinstance(amplitude_scale, bool)
            or not isinstance(amplitude_scale, (int, float))
            or amplitude_scale <= 0
        ):
            raise SpvForwardError("amplitude_scale must be positive")
        if kernel_normalization.lower() != "l1":
            raise SpvForwardError("only L1 kernel normalization is supported")
        self.mapping = mapping
        self.mapping_id = mapping.mapping_id
        self.amplitude_scale = float(amplitude_scale)
        self.kernel_normalization = "l1"
        self._kernel_bank = self._build_kernel_bank(mapping)

    @staticmethod
    def _build_kernel_bank(mapping: ElectrodeMapping) -> NDArray[np.float32]:
        height, width = mapping.image_shape
        yy, xx = np.mgrid[0:height, 0:width]
        xx = xx.astype(np.float32)[None, :, :]
        yy = yy.astype(np.float32)[None, :, :]
        center_x = mapping.centers_xy[:, 0, None, None]
        center_y = mapping.centers_xy[:, 1, None, None]
        sigma = mapping.sigma_px[:, None, None]
        squared_distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        kernels = np.exp(-squared_distance / (np.float32(2.0) * sigma**2)).astype(
            np.float32
        )
        sums = kernels.sum(axis=(1, 2), dtype=np.float64)
        if not np.isfinite(sums).all() or np.any(sums <= 0):
            raise SpvForwardError("a Gaussian kernel has invalid total energy")
        kernels /= sums.astype(np.float32)[:, None, None]
        kernels = np.ascontiguousarray(kernels, dtype=np.float32)
        normalized_sums = kernels.sum(axis=(1, 2), dtype=np.float64)
        if not np.allclose(normalized_sums, 1.0, rtol=0.0, atol=1e-5):
            raise SpvForwardError("Gaussian kernel L1 normalization failed")
        kernels.setflags(write=False)
        return kernels

    @property
    def kernel_bank(self) -> NDArray[np.float32]:
        """Read-only array with shape (candidate_count, height, width)."""

        return self._kernel_bank

    def generate(
        self,
        selected_indices: IndexVector,
        amplitudes: FloatVector,
    ) -> FloatImage:
        indices = _validate_indices(selected_indices, self.mapping.candidate_count)
        values = _validate_amplitudes(amplitudes, (indices.size,))
        weights = values * np.float32(self.amplitude_scale)
        output = np.tensordot(
            weights, self._kernel_bank[indices], axes=((0,), (0,))
        ).astype(np.float32)
        output = np.ascontiguousarray(output, dtype=np.float32)
        self._validate_output(output)
        return output

    def generate_batch(
        self,
        selected_indices: IndexVector,
        amplitudes: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Generate a batch sharing one fixed diagnostic selection."""

        indices = _validate_indices(selected_indices, self.mapping.candidate_count)
        if not isinstance(amplitudes, np.ndarray) or amplitudes.ndim != 2:
            raise TypeError("batch amplitudes must be a two-dimensional numpy array")
        values = _validate_amplitudes(amplitudes, (amplitudes.shape[0], indices.size))
        kernels = self._kernel_bank[indices].reshape(indices.size, -1)
        flat = (values * np.float32(self.amplitude_scale)) @ kernels
        output = flat.reshape(amplitudes.shape[0], *self.mapping.image_shape).astype(
            np.float32
        )
        output = np.ascontiguousarray(output, dtype=np.float32)
        if not np.isfinite(output).all() or float(output.min()) < 0.0:
            raise SpvForwardError("batch forward output contains invalid values")
        return output

    def generate_variable_batch(
        self,
        selected_indices: NDArray[np.int64],
        amplitudes: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Generate a batch in which every sample has its own selection.

        The implementation deliberately processes one sample at a time.  This
        keeps peak memory small on a 16-GB CPU machine and avoids constructing
        an ``N x B x H x W`` temporary kernel tensor.
        """

        if not isinstance(selected_indices, np.ndarray):
            raise TypeError("batch selected_indices must be a numpy array")
        if selected_indices.dtype != np.int64 or selected_indices.ndim != 2:
            raise TypeError("batch selected_indices must be a two-dimensional int64 array")
        if selected_indices.shape[0] == 0 or selected_indices.shape[1] == 0:
            raise SpvForwardError("batch selected_indices must not be empty")
        if not isinstance(amplitudes, np.ndarray) or amplitudes.ndim != 2:
            raise TypeError("batch amplitudes must be a two-dimensional numpy array")
        values = _validate_amplitudes(amplitudes, selected_indices.shape)
        output = np.empty(
            (selected_indices.shape[0], *self.mapping.image_shape), dtype=np.float32
        )
        for sample_index in range(selected_indices.shape[0]):
            output[sample_index] = self.generate(
                selected_indices[sample_index], values[sample_index]
            )
        return np.ascontiguousarray(output, dtype=np.float32)

    def _validate_output(self, output: NDArray[np.float32]) -> None:
        if output.shape != self.mapping.image_shape or output.dtype != np.float32:
            raise SpvForwardError("forward output has invalid shape or dtype")
        if not np.isfinite(output).all() or float(output.min()) < 0.0:
            raise SpvForwardError("forward output contains invalid values")


class IdealGaussianUtilityProjector:
    """Project every drive map with the same ideal L1 Gaussian kernel bank.

    This is the protocol-v2 replacement for sampling a drive map only at an
    electrode centre.  It intentionally accepts only the frozen ideal mapping;
    irregular forward mappings cannot influence encoder selection.
    """

    def __init__(self, ideal_mapping: ElectrodeMapping) -> None:
        if ideal_mapping.mapping_id != "ideal" or ideal_mapping.seed != 0:
            raise SpvForwardError(
                "candidate utility projection requires the seed-0 ideal mapping"
            )
        model = GaussianPhospheneForwardModel(
            ideal_mapping,
            amplitude_scale=1.0,
            kernel_normalization="l1",
        )
        self.mapping_id = ideal_mapping.mapping_id
        self.image_shape = ideal_mapping.image_shape
        self.candidate_count = ideal_mapping.candidate_count
        self._kernel_bank = model.kernel_bank

    @property
    def kernel_bank(self) -> NDArray[np.float32]:
        """Read-only L1-normalized bank with shape ``(N, H, W)``."""

        return self._kernel_bank

    def project(self, drive_map: FloatImage) -> FloatVector:
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
            raise SpvForwardError("candidate projection returned an invalid shape")
        if not np.isfinite(scores).all() or float(scores.min()) < 0.0:
            raise SpvForwardError("candidate projection returned invalid scores")
        return scores


def sample_images_at_electrodes(
    images: NDArray[np.float32],
    grid: RegularGrid | ElectrodeGrid,
    selected_indices: IndexVector,
) -> NDArray[np.float32]:
    """Bilinearly sample image intensity at fixed ideal-grid coordinates."""

    indices = _validate_indices(selected_indices, grid.candidate_count)
    if not isinstance(images, np.ndarray) or images.dtype != np.float32 or images.ndim != 3:
        raise TypeError("images must be a float32 array with shape (N, H, W)")
    if images.shape[1:] != grid.image_shape:
        raise SpvForwardError(
            f"image shape {images.shape[1:]} does not match grid shape {grid.image_shape}"
        )
    if not np.isfinite(images).all() or float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise SpvForwardError("images must contain finite values in [0, 1]")

    height, width = grid.image_shape
    centers = grid.centers_xy[indices]
    x = centers[:, 0]
    y = centers[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0).astype(np.float32)[None, :]
    wy = (y - y0).astype(np.float32)[None, :]
    top = images[:, y0, x0] * (np.float32(1.0) - wx) + images[:, y0, x1] * wx
    bottom = images[:, y1, x0] * (np.float32(1.0) - wx) + images[:, y1, x1] * wx
    values = top * (np.float32(1.0) - wy) + bottom * wy
    return np.ascontiguousarray(np.clip(values, 0.0, 1.0), dtype=np.float32)


def expected_output_energy(
    amplitudes: NDArray[np.float32], amplitude_scale: float = 1.0
) -> NDArray[np.float64]:
    if not isinstance(amplitudes, np.ndarray) or amplitudes.dtype != np.float32:
        raise TypeError("amplitudes must be a float32 numpy array")
    if amplitudes.ndim not in {1, 2}:
        raise SpvForwardError("amplitudes must be one- or two-dimensional")
    if not np.isfinite(amplitudes).all() or np.any(amplitudes < 0):
        raise SpvForwardError("amplitudes contains invalid values")
    return amplitudes.sum(axis=-1, dtype=np.float64) * float(amplitude_scale)
