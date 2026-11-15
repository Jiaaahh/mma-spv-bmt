"""Parameter-free cosine prototypes pooled across fonts and mappings.

Each training row represents one class/font/mapping observation.  The
classifier requires a complete, balanced Cartesian product of training fonts
and mapping replicas for every class.  A class prototype is the arithmetic
mean of those observations followed by L2 normalization.  Prediction uses the
largest cosine similarity, with exact ties resolved by the explicitly frozen
class order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class MultiMappingPrototypeError(ValueError):
    """Raised when prototype inputs or fitted state violate the contract."""


def _validate_class_order(class_order: Sequence[int]) -> NDArray[np.int64]:
    if isinstance(class_order, (str, bytes)) or not isinstance(class_order, Sequence):
        raise TypeError("class_order must be a sequence of integer class IDs")
    values = tuple(class_order)
    if len(values) < 2:
        raise MultiMappingPrototypeError(
            "class_order must contain at least two classes"
        )
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise TypeError("class_order must contain only integer class IDs")
    order = np.asarray(values, dtype=np.int64)
    if np.any(order < 0):
        raise MultiMappingPrototypeError("class_order IDs must be non-negative")
    if np.unique(order).size != order.size:
        raise MultiMappingPrototypeError("class_order must not contain duplicates")
    return np.ascontiguousarray(order)


def _validate_features(features: np.ndarray) -> NDArray[np.float32]:
    if not isinstance(features, np.ndarray):
        raise TypeError("features must be a numpy array")
    if features.dtype != np.float32 or features.ndim != 2:
        raise TypeError("features must be a two-dimensional float32 array")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise MultiMappingPrototypeError("features must not be empty")
    if not np.isfinite(features).all():
        raise MultiMappingPrototypeError("features contain non-finite values")
    return features


def _validate_labels(
    labels: np.ndarray,
    *,
    sample_count: int,
    class_order: NDArray[np.int64],
) -> NDArray[np.int64]:
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy array")
    if labels.dtype != np.int64 or labels.ndim != 1:
        raise TypeError("labels must be a one-dimensional int64 array")
    if labels.shape != (sample_count,):
        raise MultiMappingPrototypeError("label count does not match the feature rows")
    observed = set(map(int, np.unique(labels)))
    expected = set(map(int, class_order))
    if observed != expected:
        raise MultiMappingPrototypeError(
            "labels must contain every class in class_order and no other classes"
        )
    return labels


def _validate_ids(
    values: np.ndarray,
    *,
    name: str,
    sample_count: int,
) -> NDArray[np.str_]:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if values.ndim != 1 or values.dtype.kind != "U":
        raise TypeError(f"{name} must be a one-dimensional Unicode array")
    if values.shape != (sample_count,):
        raise MultiMappingPrototypeError(
            f"{name} count does not match the feature rows"
        )
    stripped = np.char.strip(values)
    if np.any(stripped == ""):
        raise MultiMappingPrototypeError(f"{name} must not contain empty IDs")
    if np.any(stripped != values):
        raise MultiMappingPrototypeError(
            f"{name} IDs must not have leading or trailing whitespace"
        )
    return values


def _validate_balanced_replicas(
    labels: NDArray[np.int64],
    font_ids: NDArray[np.str_],
    mapping_ids: NDArray[np.str_],
    class_order: NDArray[np.int64],
) -> None:
    fonts = tuple(map(str, np.unique(font_ids)))
    mappings = tuple(map(str, np.unique(mapping_ids)))
    expected_pairs = {
        (font_id, mapping_id) for font_id in fonts for mapping_id in mappings
    }

    for class_id in class_order:
        class_mask = labels == class_id
        observed_pairs = [
            (str(font_id), str(mapping_id))
            for font_id, mapping_id in zip(
                font_ids[class_mask], mapping_ids[class_mask], strict=True
            )
        ]
        counts = Counter(observed_pairs)
        if set(counts) != expected_pairs or any(
            count != 1 for count in counts.values()
        ):
            raise MultiMappingPrototypeError(
                "each class must contain exactly one observation for every "
                "training-font/mapping pair"
            )


class MultiMappingCosinePrototypeClassifier:
    """Balanced multi-mapping class means decoded by cosine similarity."""

    name = "multimapping_cosine_prototype"

    def __init__(self, class_order: Sequence[int]) -> None:
        self._class_order = _validate_class_order(class_order)
        self._prototypes: NDArray[np.float64] | None = None
        self._feature_count: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether a complete prototype bank has been fitted."""

        return self._prototypes is not None

    @property
    def class_order_(self) -> NDArray[np.int64]:
        """Frozen class order as a read-only copy."""

        result = self._class_order.copy()
        result.flags.writeable = False
        return result

    @property
    def prototypes_(self) -> NDArray[np.float64]:
        """L2-normalized class prototypes as a read-only copy."""

        if self._prototypes is None:
            raise MultiMappingPrototypeError("classifier has not been fitted")
        result = self._prototypes.copy()
        result.flags.writeable = False
        return result

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        font_ids: NDArray[np.str_],
        mapping_ids: NDArray[np.str_],
    ) -> "MultiMappingCosinePrototypeClassifier":
        """Fit one balanced, normalized prototype per frozen class."""

        checked_features = _validate_features(features)
        checked_labels = _validate_labels(
            labels,
            sample_count=checked_features.shape[0],
            class_order=self._class_order,
        )
        checked_fonts = _validate_ids(
            font_ids,
            name="font_ids",
            sample_count=checked_features.shape[0],
        )
        checked_mappings = _validate_ids(
            mapping_ids,
            name="mapping_ids",
            sample_count=checked_features.shape[0],
        )
        _validate_balanced_replicas(
            checked_labels,
            checked_fonts,
            checked_mappings,
            self._class_order,
        )

        feature64 = checked_features.astype(np.float64, copy=False)
        prototypes = np.empty(
            (self._class_order.size, checked_features.shape[1]), dtype=np.float64
        )
        for row_index, class_id in enumerate(self._class_order):
            class_mean = np.mean(
                feature64[checked_labels == class_id], axis=0, dtype=np.float64
            )
            norm = float(np.linalg.norm(class_mean))
            if not np.isfinite(norm) or norm == 0.0:
                raise MultiMappingPrototypeError(
                    f"class {int(class_id)} has a zero-norm mean prototype"
                )
            prototypes[row_index] = class_mean / norm

        if not np.isfinite(prototypes).all():
            raise MultiMappingPrototypeError("normalized prototypes are non-finite")
        self._prototypes = np.ascontiguousarray(prototypes, dtype=np.float64)
        self._feature_count = int(checked_features.shape[1])
        return self

    def decision_function(self, features: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return cosine similarities in the exact order of ``class_order_``."""

        checked_features = _validate_features(features)
        if self._prototypes is None or self._feature_count is None:
            raise MultiMappingPrototypeError("classifier must be fitted before scoring")
        if checked_features.shape[1] != self._feature_count:
            raise MultiMappingPrototypeError(
                "prediction feature width differs from training"
            )
        feature64 = checked_features.astype(np.float64, copy=False)
        norms = np.linalg.norm(feature64, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
            raise MultiMappingPrototypeError(
                "prediction features must have non-zero L2 norm"
            )
        normalized = feature64 / norms[:, None]
        scores = normalized @ self._prototypes.T
        scores = np.clip(scores, -1.0, 1.0)
        if scores.shape != (checked_features.shape[0], self._class_order.size):
            raise MultiMappingPrototypeError("cosine score matrix has invalid shape")
        if not np.isfinite(scores).all():
            raise MultiMappingPrototypeError("cosine scores are non-finite")
        return np.ascontiguousarray(scores, dtype=np.float64)

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.int64]:
        """Predict by maximum cosine; exact ties follow the frozen class order."""

        scores = self.decision_function(features)
        winner_rows = np.argmax(scores, axis=1)
        predictions = self._class_order[winner_rows]
        return np.ascontiguousarray(predictions, dtype=np.int64)
