"""Stable data and plug-in contracts for the Hanzi-SPV pipeline.

Only interfaces and inexpensive invariant checks belong here.  Concrete
renderers, encoders, SPV simulators, feature extractors, and classifiers live in
separate modules so later baselines can be added without changing boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


FloatImage = NDArray[np.float32]
FloatVector = NDArray[np.float32]
IndexVector = NDArray[np.int64]
LabelVector = NDArray[np.int64]
SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class TopologyDetectionSpec:
    """Frozen convention for future topology encoders; no detector lives here.

    For clockwise binary neighbours ``b_1, ..., b_8`` around one skeleton
    pixel, with ``b_9 = b_1``, the Crossing Number is

        CN = 0.5 * sum_k |b_k - b_(k+1)|.

    A skeleton pixel with CN=1 is an endpoint and a pixel with CN>=3 is a
    junction candidate.  Adjacent junction candidates will later be merged;
    Stage 4.5 deliberately freezes only this convention and implements no
    Skeleton/EB encoder.
    """

    method: Literal["crossing_number"]
    skeletonization: Literal["zhang"]
    binarization_threshold: float
    neighbor_order_xy: tuple[tuple[int, int], ...]
    border_policy: Literal["zero_padding"]
    endpoint_crossing_number: int
    junction_min_crossing_number: int
    junction_component_connectivity: int
    junction_representative: Literal["nearest_centroid_then_row_major"]

    def __post_init__(self) -> None:
        if self.method != "crossing_number":
            raise ValueError("topology method must be crossing_number")
        if self.skeletonization != "zhang":
            raise ValueError("skeletonization convention must be zhang")
        if not 0.0 < self.binarization_threshold < 1.0:
            raise ValueError("binarization_threshold must lie in (0, 1)")
        if len(self.neighbor_order_xy) != 8 or len(set(self.neighbor_order_xy)) != 8:
            raise ValueError("neighbor_order_xy must contain eight unique offsets")
        if self.border_policy != "zero_padding":
            raise ValueError("topology border_policy must be zero_padding")
        if self.endpoint_crossing_number != 1:
            raise ValueError("endpoint Crossing Number must equal 1")
        if self.junction_min_crossing_number != 3:
            raise ValueError("junction Crossing Number threshold must equal 3")
        if self.junction_component_connectivity != 8:
            raise ValueError("junction components must use 8-connectivity")


FROZEN_TOPOLOGY_DETECTION = TopologyDetectionSpec(
    method="crossing_number",
    skeletonization="zhang",
    binarization_threshold=0.5,
    neighbor_order_xy=(
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
    ),
    border_policy="zero_padding",
    endpoint_crossing_number=1,
    junction_min_crossing_number=3,
    junction_component_connectivity=8,
    junction_representative="nearest_centroid_then_row_major",
)


@dataclass(frozen=True, slots=True)
class EbTopologyDriveSpec:
    """Stage-6 MVP constants frozen by the Stage-5.5 truth table.

    These are engineering-validation values only.  The formal development
    search may later replace ``alpha_e`` and ``alpha_j`` with one pair selected
    from the preregistered grid, but Stage 6 must not search or adapt them.
    """

    sigma_s_px: float
    sigma_t_px: float
    alpha_s: float
    alpha_e: float
    alpha_j: float
    diffusion_support: Literal["full_canvas"]
    component_normalization: Literal["max_if_nonzero"]
    drive_normalization: Literal["max_if_nonzero"]
    weight_status: Literal["stage6_mvp_fixed_no_search"]

    def __post_init__(self) -> None:
        expected = {
            "sigma_s_px": 2.1,
            "sigma_t_px": 3.0,
            "alpha_s": 1.0,
            "alpha_e": 1.0,
            "alpha_j": 1.0,
        }
        for name, target in expected.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not np.isclose(float(value), target, rtol=0.0, atol=1e-12):
                raise ValueError(f"Stage-6 {name} must equal {target}")
        if self.diffusion_support != "full_canvas":
            raise ValueError("EB diffusion support must be full_canvas")
        if self.component_normalization != "max_if_nonzero":
            raise ValueError("EB components require max_if_nonzero normalization")
        if self.drive_normalization != "max_if_nonzero":
            raise ValueError("EB drive map requires max_if_nonzero normalization")
        if self.weight_status != "stage6_mvp_fixed_no_search":
            raise ValueError("Stage-6 EB weights must be fixed without search")


FROZEN_STAGE6_EB_DRIVE = EbTopologyDriveSpec(
    sigma_s_px=2.1,
    sigma_t_px=3.0,
    alpha_s=1.0,
    alpha_e=1.0,
    alpha_j=1.0,
    diffusion_support="full_canvas",
    component_normalization="max_if_nonzero",
    drive_normalization="max_if_nonzero",
    weight_status="stage6_mvp_fixed_no_search",
)


@dataclass(frozen=True, slots=True)
class CoverageSelectionSpec:
    """Stage-7 selector constants copied from the Stage-5.5 truth table."""

    d_e_px: float
    sigma_r_px: float
    cutoff_px: float
    diagonal: float
    geometry: Literal["ideal_grid_only"]
    tie_break: Literal["candidate_index_ascending"]
    budgets: tuple[int, int, int]
    max_order_length: int
    objective: Literal[
        "F_D(A)=sum_i u_i(D)*(1-prod_{j in A}(1-R_ji))"
    ]
    greedy_gain: Literal[
        "Delta(k|A)=sum_i u_i(D)*R_ki*prod_{j in A}(1-R_ji)"
    ]
    theoretical_optimality_claim: Literal["none"]

    def __post_init__(self) -> None:
        expected_float = {
            "d_e_px": 6.0,
            "sigma_r_px": 4.5,
            "cutoff_px": 12.0,
            "diagonal": 1.0,
        }
        for name, expected in expected_float.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not np.isclose(float(value), expected, rtol=0.0, atol=1e-12):
                raise ValueError(f"Stage-7 {name} must equal {expected}")
        if self.geometry != "ideal_grid_only":
            raise ValueError("coverage geometry must be ideal_grid_only")
        if self.tie_break != "candidate_index_ascending":
            raise ValueError("coverage tie break must use ascending candidate index")
        if self.budgets != (32, 64, 128):
            raise ValueError("coverage budgets must be (32, 64, 128)")
        if self.max_order_length != 128:
            raise ValueError("coverage must generate one 128-candidate order")
        if self.objective != (
            "F_D(A)=sum_i u_i(D)*(1-prod_{j in A}(1-R_ji))"
        ):
            raise ValueError("coverage objective differs from Stage 5.5")
        if self.greedy_gain != (
            "Delta(k|A)=sum_i u_i(D)*R_ki*prod_{j in A}(1-R_ji)"
        ):
            raise ValueError("coverage gain differs from Stage 5.5")
        if self.theoretical_optimality_claim != "none":
            raise ValueError("Stage 7 must make no theoretical optimality claim")


FROZEN_STAGE7_COVERAGE = CoverageSelectionSpec(
    d_e_px=6.0,
    sigma_r_px=4.5,
    cutoff_px=12.0,
    diagonal=1.0,
    geometry="ideal_grid_only",
    tie_break="candidate_index_ascending",
    budgets=(32, 64, 128),
    max_order_length=128,
    objective="F_D(A)=sum_i u_i(D)*(1-prod_{j in A}(1-R_ji))",
    greedy_gain="Delta(k|A)=sum_i u_i(D)*R_ki*prod_{j in A}(1-R_ji)",
    theoretical_optimality_claim="none",
)


def _require_float32_array(value: Any, field_name: str, ndim: int) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name} must be a numpy.ndarray")
    if value.dtype != np.float32:
        raise TypeError(f"{field_name} must use dtype float32")
    if value.ndim != ndim:
        raise ValueError(f"{field_name} must be {ndim}-dimensional")
    if not np.isfinite(value).all():
        raise ValueError(f"{field_name} must contain only finite values")


def _require_image_shape(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in value)
    ):
        raise ValueError("image_shape must be a tuple of two positive integers")
    return value


@dataclass(frozen=True, slots=True)
class GlyphSample:
    """One normalized glyph plus bookkeeping used outside the encoder.

    The encoder protocol below deliberately receives only ``image`` rather than
    a ``GlyphSample``.  This prevents labels, split roles, and font identifiers
    from becoming accidental inputs to an encoding method.
    """

    sample_id: str
    character: str
    class_id: int
    font_id: str
    split: SplitName
    image: FloatImage = field(repr=False)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if len(self.character) != 1:
            raise ValueError("character must contain exactly one Unicode character")
        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id < 0
        ):
            raise ValueError("class_id must be a non-negative integer")
        if not self.font_id:
            raise ValueError("font_id must not be empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        _require_float32_array(self.image, "image", ndim=2)
        if self.image.size == 0:
            raise ValueError("image must not be empty")
        if float(self.image.min()) < 0.0 or float(self.image.max()) > 1.0:
            raise ValueError("image values must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class ElectrodeGrid:
    """Candidate electrode centers in image pixel coordinates (x, y)."""

    centers_xy: FloatImage = field(repr=False)
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        _require_float32_array(self.centers_xy, "centers_xy", ndim=2)
        if self.centers_xy.shape[1:] != (2,) or self.centers_xy.shape[0] == 0:
            raise ValueError("centers_xy must have shape (N, 2) with N > 0")
        height, width = _require_image_shape(self.image_shape)
        x, y = self.centers_xy[:, 0], self.centers_xy[:, 1]
        if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
            raise ValueError("candidate centers must lie inside the image canvas")
        if np.unique(self.centers_xy, axis=0).shape[0] != self.centers_xy.shape[0]:
            raise ValueError("candidate centers must be unique")

    @property
    def candidate_count(self) -> int:
        return int(self.centers_xy.shape[0])


@dataclass(frozen=True, slots=True)
class ElectrodeMapping:
    """One frozen ideal or irregular mapping shared fairly by all encoders."""

    mapping_id: str
    seed: int
    centers_xy: FloatImage = field(repr=False)
    sigma_px: FloatVector = field(repr=False)
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.mapping_id:
            raise ValueError("mapping_id must not be empty")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        _require_float32_array(self.centers_xy, "centers_xy", ndim=2)
        _require_float32_array(self.sigma_px, "sigma_px", ndim=1)
        if self.centers_xy.shape[1:] != (2,) or self.centers_xy.shape[0] == 0:
            raise ValueError("centers_xy must have shape (N, 2) with N > 0")
        if self.sigma_px.shape != (self.centers_xy.shape[0],):
            raise ValueError("sigma_px must contain one value per candidate electrode")
        if np.any(self.sigma_px <= 0):
            raise ValueError("all phosphene sigma values must be positive")
        height, width = _require_image_shape(self.image_shape)
        x, y = self.centers_xy[:, 0], self.centers_xy[:, 1]
        if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height)):
            raise ValueError("mapped centers must lie inside the image canvas")

    @property
    def candidate_count(self) -> int:
        return int(self.centers_xy.shape[0])


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Validated v2 output returned by every electrode encoder."""

    encoder_name: str
    budget: int
    candidate_count: int
    selected_indices: IndexVector = field(repr=False)
    candidate_scores: FloatVector = field(repr=False)
    drive_map: FloatImage = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.encoder_name:
            raise ValueError("encoder_name must not be empty")
        if isinstance(self.budget, bool) or self.budget <= 0:
            raise ValueError("budget must be a positive integer")
        if isinstance(self.candidate_count, bool) or self.candidate_count <= 0:
            raise ValueError("candidate_count must be a positive integer")
        if not isinstance(self.selected_indices, np.ndarray):
            raise TypeError("selected_indices must be a numpy.ndarray")
        if self.selected_indices.dtype != np.int64 or self.selected_indices.ndim != 1:
            raise TypeError("selected_indices must be a one-dimensional int64 array")
        _require_float32_array(self.candidate_scores, "candidate_scores", ndim=1)
        _require_float32_array(self.drive_map, "drive_map", ndim=2)
        if self.selected_indices.shape != (self.budget,):
            raise ValueError("selected_indices length must equal budget")
        if self.candidate_scores.shape != (self.candidate_count,):
            raise ValueError("candidate_scores length must equal candidate_count")
        if self.budget > self.candidate_count:
            raise ValueError("budget must not exceed candidate_count")
        if np.unique(self.selected_indices).size != self.budget:
            raise ValueError("selected_indices must be unique")
        if np.any(self.selected_indices < 0) or np.any(
            self.selected_indices >= self.candidate_count
        ):
            raise ValueError("selected_indices contains an out-of-range index")
        if self.drive_map.size == 0:
            raise ValueError("drive_map must not be empty")
        if float(self.drive_map.min()) < 0.0 or float(self.drive_map.max()) > 1.0:
            raise ValueError("drive_map values must lie in [0, 1]")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


@runtime_checkable
class CandidateUtilityProjector(Protocol):
    """Shared L1-normalized ideal-Gaussian drive-map projector."""

    mapping_id: str
    image_shape: tuple[int, int]
    candidate_count: int

    def project(self, drive_map: FloatImage) -> FloatVector:
        """Project one non-negative drive map to one score per candidate."""


@runtime_checkable
class Encoder(Protocol):
    """Fair plug-in boundary for electrode selection under protocol v2."""

    name: str

    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: CandidateUtilityProjector,
    ) -> SelectionResult:
        """Return drive map, shared projected utilities, and selected indices."""


@runtime_checkable
class SpvForwardModel(Protocol):
    """Boundary for converting selected electrodes into one SPV image."""

    mapping_id: str

    def generate(
        self,
        selected_indices: IndexVector,
        amplitudes: FloatVector,
    ) -> FloatImage:
        """Generate a finite float32 image using explicit stimulation amplitudes."""


@runtime_checkable
class FeatureExtractor(Protocol):
    """Boundary for deterministic feature extraction (HOG in this project)."""

    name: str

    def extract(self, image: FloatImage) -> FloatVector:
        """Convert one SPV image into a finite one-dimensional feature vector."""


@runtime_checkable
class Classifier(Protocol):
    """Small interface required from the fixed LinearSVC decoder."""

    def fit(self, features: FloatImage, labels: LabelVector) -> Any:
        """Fit the classifier and return the fitted object."""

    def predict(self, features: FloatImage) -> LabelVector:
        """Predict integer class IDs."""


def _self_check() -> None:
    centers = np.array(
        [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]], dtype=np.float32
    )
    grid = ElectrodeGrid(centers_xy=centers, image_shape=(4, 4))
    selection = SelectionResult(
        encoder_name="contract_self_check",
        budget=2,
        candidate_count=grid.candidate_count,
        selected_indices=np.array([0, 3], dtype=np.int64),
        candidate_scores=np.array([1.0, 0.2, 0.1, 0.9], dtype=np.float32),
        drive_map=np.eye(4, dtype=np.float32),
    )
    assert selection.budget == 2
    print("[PASS] Contract self-check: arrays, bounds, uniqueness, and budget invariants")


if __name__ == "__main__":
    _self_check()
