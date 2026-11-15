"""Deterministic Crossing-Number topology detection for Stage 6."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from src.contracts import FROZEN_TOPOLOGY_DETECTION


class TopologyError(ValueError):
    """Raised when a skeleton or topology result violates the frozen contract."""


BoolImage = NDArray[np.bool_]
UInt8Image = NDArray[np.uint8]
Int32Image = NDArray[np.int32]


def _validate_binary_image(image: np.ndarray, name: str) -> BoolImage:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise TypeError(f"{name} must be a non-empty two-dimensional numpy array")
    if image.dtype != np.bool_:
        raise TypeError(f"{name} must use boolean dtype")
    return np.ascontiguousarray(image, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class TopologyDetectionResult:
    """All intermediate maps needed for tests and visual quality control."""

    crossing_numbers: UInt8Image = field(repr=False)
    endpoints: BoolImage = field(repr=False)
    junction_candidates: BoolImage = field(repr=False)
    junction_labels: Int32Image = field(repr=False)
    junction_representatives: BoolImage = field(repr=False)
    junction_cluster_count: int

    def __post_init__(self) -> None:
        shape = self.crossing_numbers.shape
        if self.crossing_numbers.dtype != np.uint8 or self.crossing_numbers.ndim != 2:
            raise TypeError("crossing_numbers must be a two-dimensional uint8 array")
        for name in ("endpoints", "junction_candidates", "junction_representatives"):
            value = getattr(self, name)
            if value.dtype != np.bool_ or value.shape != shape:
                raise TypeError(f"{name} must be a boolean array matching the CN map")
        if self.junction_labels.dtype != np.int32 or self.junction_labels.shape != shape:
            raise TypeError("junction_labels must be int32 and match the CN map")
        if (
            isinstance(self.junction_cluster_count, bool)
            or not isinstance(self.junction_cluster_count, int)
            or self.junction_cluster_count < 0
        ):
            raise ValueError("junction_cluster_count must be a non-negative integer")
        if int(self.junction_representatives.sum()) != self.junction_cluster_count:
            raise ValueError("each junction cluster must have exactly one representative")
        if np.any(self.junction_representatives & ~self.junction_candidates):
            raise ValueError("a junction representative is not a junction candidate")


def crossing_number_map(skeleton: np.ndarray) -> UInt8Image:
    """Return the frozen eight-neighbour Crossing Number at skeleton pixels."""

    binary = _validate_binary_image(skeleton, "skeleton")
    spec = FROZEN_TOPOLOGY_DETECTION
    padded = np.pad(binary.astype(np.int8), 1, mode="constant", constant_values=0)
    height, width = binary.shape
    neighbours = []
    for dx, dy in spec.neighbor_order_xy:
        neighbours.append(
            padded[
                1 + dy : 1 + dy + height,
                1 + dx : 1 + dx + width,
            ]
        )
    ring = np.stack(neighbours, axis=0).astype(np.int8, copy=False)
    transitions = np.abs(ring - np.roll(ring, -1, axis=0)).sum(
        axis=0, dtype=np.int16
    )
    if np.any(transitions % 2):
        raise TopologyError("binary neighbour transitions must be even")
    result = (transitions // 2).astype(np.uint8)
    result[~binary] = np.uint8(0)
    return np.ascontiguousarray(result, dtype=np.uint8)


def cluster_junction_candidates(
    junction_candidates: np.ndarray,
) -> tuple[Int32Image, BoolImage, int]:
    """Merge 8-connected candidates and choose one deterministic representative."""

    candidates = _validate_binary_image(
        junction_candidates, "junction_candidates"
    )
    structure = np.ones((3, 3), dtype=np.uint8)
    labels_raw, cluster_count = ndimage.label(candidates, structure=structure)
    labels = np.ascontiguousarray(labels_raw, dtype=np.int32)
    representatives = np.zeros(candidates.shape, dtype=np.bool_)
    for label_id in range(1, int(cluster_count) + 1):
        coordinates_yx = np.argwhere(labels == label_id)
        if coordinates_yx.size == 0:
            raise TopologyError("junction labelling produced an empty component")
        centroid_yx = coordinates_yx.mean(axis=0, dtype=np.float64)
        squared_distance = np.sum(
            (coordinates_yx.astype(np.float64) - centroid_yx) ** 2,
            axis=1,
        )
        # np.argwhere is row-major; np.argmin therefore implements the frozen
        # y-then-x tie break for candidates equally near the centroid.
        representative_index = int(np.argmin(squared_distance))
        y, x = map(int, coordinates_yx[representative_index])
        representatives[y, x] = True
    return labels, np.ascontiguousarray(representatives), int(cluster_count)


def detect_skeleton_topology(skeleton: np.ndarray) -> TopologyDetectionResult:
    """Detect endpoints, junction candidates, clusters, and representatives."""

    binary = _validate_binary_image(skeleton, "skeleton")
    spec = FROZEN_TOPOLOGY_DETECTION
    crossing_numbers = crossing_number_map(binary)
    endpoints = np.ascontiguousarray(
        binary & (crossing_numbers == spec.endpoint_crossing_number),
        dtype=np.bool_,
    )
    junction_candidates = np.ascontiguousarray(
        binary & (crossing_numbers >= spec.junction_min_crossing_number),
        dtype=np.bool_,
    )
    labels, representatives, cluster_count = cluster_junction_candidates(
        junction_candidates
    )
    return TopologyDetectionResult(
        crossing_numbers=crossing_numbers,
        endpoints=endpoints,
        junction_candidates=junction_candidates,
        junction_labels=labels,
        junction_representatives=representatives,
        junction_cluster_count=cluster_count,
    )
