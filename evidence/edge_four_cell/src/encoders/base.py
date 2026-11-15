"""Formal base class and shared validation for electrode encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.contracts import (
    CandidateUtilityProjector,
    ElectrodeGrid,
    FloatImage,
    SelectionResult,
)


class EncoderError(ValueError):
    """Raised when an encoder input or output violates the shared contract."""


def deterministic_topk(
    scores: np.ndarray, budget: int
) -> np.ndarray:
    """Select descending scores with ascending-index tie resolution."""

    if not isinstance(scores, np.ndarray):
        raise TypeError("scores must be a numpy array")
    if scores.dtype != np.float32 or scores.ndim != 1 or scores.size == 0:
        raise TypeError("scores must be a non-empty one-dimensional float32 array")
    if not np.isfinite(scores).all() or float(scores.min()) < 0.0:
        raise EncoderError("scores must contain finite non-negative values")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise EncoderError("budget must be a positive integer")
    if budget > scores.size:
        raise EncoderError("budget must not exceed the score count")
    candidate_indices = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort(
        (candidate_indices, -scores.astype(np.float64, copy=False))
    )
    return np.ascontiguousarray(order[:budget], dtype=np.int64)


class BaseEncoder(ABC):
    """Minimal implementation boundary used by every electrode selector."""

    name: str

    @abstractmethod
    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: CandidateUtilityProjector,
    ) -> SelectionResult:
        """Select exactly ``budget`` unique candidates from an unlabeled image."""

    @staticmethod
    def validate_inputs(
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        projector: CandidateUtilityProjector,
    ) -> None:
        if not isinstance(grid, ElectrodeGrid):
            raise TypeError("grid must be an ElectrodeGrid")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if image.dtype != np.float32 or image.ndim != 2:
            raise TypeError("image must be a two-dimensional float32 array")
        if image.shape != grid.image_shape:
            raise EncoderError(
                f"image shape {image.shape} does not match grid shape {grid.image_shape}"
            )
        if not np.isfinite(image).all():
            raise EncoderError("image contains non-finite values")
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise EncoderError("image values must lie in [0, 1]")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise EncoderError("budget must be a positive integer")
        if budget > grid.candidate_count:
            raise EncoderError("budget must not exceed the candidate count")
        if not isinstance(projector, CandidateUtilityProjector):
            raise TypeError("projector must satisfy CandidateUtilityProjector")
        if projector.mapping_id != "ideal":
            raise EncoderError("candidate utility projector must use the ideal mapping")
        if projector.image_shape != grid.image_shape:
            raise EncoderError("projector image shape differs from the electrode grid")
        if projector.candidate_count != grid.candidate_count:
            raise EncoderError("projector candidate count differs from the electrode grid")
