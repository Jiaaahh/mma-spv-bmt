"""Stage-6 EB topology drive-map construction without electrode selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize

from src.contracts import (
    FROZEN_STAGE6_EB_DRIVE,
    FROZEN_TOPOLOGY_DETECTION,
    FloatImage,
)
from src.topology import TopologyDetectionResult, detect_skeleton_topology


class EbDriveError(ValueError):
    """Raised when the EB topology drive map violates Stage-5.5."""


BoolImage = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class EbDriveComponents:
    """Complete EB intermediate state used by tests and topology QC figures."""

    mask: BoolImage = field(repr=False)
    skeleton: BoolImage = field(repr=False)
    topology: TopologyDetectionResult = field(repr=False)
    p_s: FloatImage = field(repr=False)
    p_e: FloatImage = field(repr=False)
    p_j: FloatImage = field(repr=False)
    drive_map: FloatImage = field(repr=False)

    def __post_init__(self) -> None:
        shape = self.mask.shape
        if self.mask.dtype != np.bool_ or self.skeleton.dtype != np.bool_:
            raise TypeError("mask and skeleton must use boolean dtype")
        if self.skeleton.shape != shape:
            raise ValueError("mask and skeleton shapes differ")
        for name in ("p_s", "p_e", "p_j", "drive_map"):
            value = getattr(self, name)
            if value.dtype != np.float32 or value.shape != shape:
                raise TypeError(f"{name} must be float32 and match the input shape")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains a non-finite value")
            if float(value.min()) < 0.0 or float(value.max()) > 1.0:
                raise ValueError(f"{name} values must lie in [0, 1]")
        if not self.skeleton.any() or float(self.p_s.max()) != 1.0:
            raise ValueError("a valid EB sample requires a non-empty skeleton component")
        if float(self.drive_map.max()) != 1.0:
            raise ValueError("the final EB drive map must be max-normalized")


def _validate_input_image(image: np.ndarray) -> FloatImage:
    if not isinstance(image, np.ndarray) or image.dtype != np.float32 or image.ndim != 2:
        raise TypeError("image must be a two-dimensional float32 numpy array")
    if image.shape != (96, 96):
        raise EbDriveError("Stage-6 EB input must have shape 96x96")
    if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise EbDriveError("image must contain finite values in [0, 1]")
    return np.ascontiguousarray(image, dtype=np.float32)


@lru_cache(maxsize=8)
def _full_gaussian_axis(length: int, sigma_px: float) -> NDArray[np.float64]:
    coordinates = np.arange(length, dtype=np.float64)
    squared_distance = (coordinates[:, None] - coordinates[None, :]) ** 2
    axis = np.exp(-squared_distance / (2.0 * float(sigma_px) ** 2))
    axis = np.ascontiguousarray(axis, dtype=np.float64)
    axis.setflags(write=False)
    return axis


def gaussian_diffuse_full_canvas(
    binary_map: np.ndarray, sigma_px: float
) -> NDArray[np.float64]:
    """Evaluate the untruncated Stage-5.5 Gaussian sum on the full canvas."""

    if not isinstance(binary_map, np.ndarray) or binary_map.dtype != np.bool_:
        raise TypeError("binary_map must be a boolean numpy array")
    if binary_map.ndim != 2 or binary_map.size == 0:
        raise TypeError("binary_map must be a non-empty two-dimensional array")
    if isinstance(sigma_px, bool) or not isinstance(sigma_px, (int, float)) or sigma_px <= 0:
        raise EbDriveError("sigma_px must be positive")
    height, width = binary_map.shape
    if not binary_map.any():
        return np.zeros((height, width), dtype=np.float64)
    axis_y = _full_gaussian_axis(height, float(sigma_px))
    axis_x = _full_gaussian_axis(width, float(sigma_px))
    result = axis_y @ binary_map.astype(np.float64) @ axis_x.T
    if result.shape != binary_map.shape or not np.isfinite(result).all():
        raise EbDriveError("full-canvas Gaussian diffusion returned an invalid map")
    if float(result.min()) < 0.0 or float(result.max()) <= 0.0:
        raise EbDriveError("non-empty Gaussian diffusion must be positive")
    return np.ascontiguousarray(result, dtype=np.float64)


def max_normalize_nonzero(image: np.ndarray) -> FloatImage:
    """Apply N(X)=X/max(X), preserving an absent component as an all-zero map."""

    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise TypeError("image must be a non-empty two-dimensional numpy array")
    values = np.asarray(image, dtype=np.float64)
    if not np.isfinite(values).all() or float(values.min()) < 0.0:
        raise EbDriveError("normalization input must be finite and non-negative")
    maximum = float(values.max())
    if maximum == 0.0:
        return np.zeros(values.shape, dtype=np.float32)
    normalized = np.ascontiguousarray(values / maximum, dtype=np.float32)
    if float(normalized.max()) != 1.0:
        raise EbDriveError("maximum normalization failed")
    return normalized


def build_eb_drive_components(image: np.ndarray) -> EbDriveComponents:
    """Build S/E/J components and the fixed equal-weight Stage-6 EB drive map."""

    glyph = _validate_input_image(image)
    topology_spec = FROZEN_TOPOLOGY_DETECTION
    eb_spec = FROZEN_STAGE6_EB_DRIVE
    mask = np.ascontiguousarray(
        glyph >= np.float32(topology_spec.binarization_threshold),
        dtype=np.bool_,
    )
    skeleton = np.ascontiguousarray(
        skeletonize(mask, method=topology_spec.skeletonization),
        dtype=np.bool_,
    )
    if not skeleton.any():
        raise EbDriveError("Zhang skeletonization produced an empty skeleton")
    topology = detect_skeleton_topology(skeleton)
    p_s = max_normalize_nonzero(
        gaussian_diffuse_full_canvas(skeleton, eb_spec.sigma_s_px)
    )
    p_e = max_normalize_nonzero(
        gaussian_diffuse_full_canvas(topology.endpoints, eb_spec.sigma_t_px)
    )
    p_j = max_normalize_nonzero(
        gaussian_diffuse_full_canvas(
            topology.junction_representatives, eb_spec.sigma_t_px
        )
    )
    combined = (
        float(eb_spec.alpha_s) * p_s.astype(np.float64)
        + float(eb_spec.alpha_e) * p_e.astype(np.float64)
        + float(eb_spec.alpha_j) * p_j.astype(np.float64)
    )
    drive_map = max_normalize_nonzero(combined)
    return EbDriveComponents(
        mask=mask,
        skeleton=skeleton,
        topology=topology,
        p_s=p_s,
        p_e=p_e,
        p_j=p_j,
        drive_map=drive_map,
    )
