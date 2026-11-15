"""Balanced deterministic LinearSVC baseline used by the E1 extension."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import io
import pickle

import numpy as np
from numpy.typing import NDArray
from sklearn.svm import LinearSVC


class E1ClassifierError(ValueError):
    """Raised when the supplemental classifier contract is violated."""


def _class_order(values: Sequence[int]) -> NDArray[np.int64]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("class_order must be a sequence")
    if len(values) < 2:
        raise E1ClassifierError("class_order requires at least two classes")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise TypeError("class_order must contain integers")
    result = np.asarray(values, dtype=np.int64)
    if np.any(result < 0) or np.unique(result).size != result.size:
        raise E1ClassifierError("class_order must contain unique non-negative IDs")
    return np.ascontiguousarray(result)


def _features(value: np.ndarray) -> NDArray[np.float32]:
    if not isinstance(value, np.ndarray):
        raise TypeError("features must be a numpy array")
    if value.dtype != np.float32 or value.ndim != 2:
        raise TypeError("features must be a two-dimensional float32 array")
    if value.shape[0] == 0 or value.shape[1] == 0 or not np.isfinite(value).all():
        raise E1ClassifierError("features must be finite and non-empty")
    return np.ascontiguousarray(value)


def _labels(
    value: np.ndarray, n: int, order: NDArray[np.int64]
) -> NDArray[np.int64]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.int64
        or value.ndim != 1
    ):
        raise TypeError("labels must be a one-dimensional int64 array")
    if value.shape != (n,) or set(map(int, np.unique(value))) != set(map(int, order)):
        raise E1ClassifierError("labels must match rows and the frozen class order")
    return np.ascontiguousarray(value)


def _ids(value: np.ndarray, name: str, n: int) -> NDArray[np.str_]:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.dtype.kind != "U"
    ):
        raise TypeError(f"{name} must be a one-dimensional Unicode array")
    if (
        value.shape != (n,)
        or np.any(np.char.strip(value) != value)
        or np.any(value == "")
    ):
        raise E1ClassifierError(f"{name} must contain one non-empty ID per row")
    return np.asarray(value, dtype=np.str_)


def _require_cartesian_balance(
    labels: NDArray[np.int64],
    font_ids: NDArray[np.str_],
    mapping_ids: NDArray[np.str_],
    order: NDArray[np.int64],
) -> int:
    fonts = tuple(map(str, np.unique(font_ids)))
    mappings = tuple(map(str, np.unique(mapping_ids)))
    expected = {(font, mapping) for font in fonts for mapping in mappings}
    for class_id in order:
        mask = labels == class_id
        observed = Counter(
            (str(font), str(mapping))
            for font, mapping in zip(
                font_ids[mask], mapping_ids[mask], strict=True
            )
        )
        if set(observed) != expected or any(count != 1 for count in observed.values()):
            raise E1ClassifierError(
                "each class requires exactly one row for every font/mapping pair"
            )
    return len(mappings)


class BalancedLinearSVC:
    """Linear ERM baseline with equal total weight for every mapping replica."""

    name = "balanced_linear_svc"

    def __init__(self, class_order: Sequence[int], *, c: float) -> None:
        if (
            isinstance(c, bool)
            or not isinstance(c, (int, float))
            or float(c) <= 0
        ):
            raise E1ClassifierError("c must be positive")
        self._order = _class_order(class_order)
        self.c = float(c)
        self._model: LinearSVC | None = None
        self._feature_count: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    @property
    def n_iter_(self) -> int:
        if self._model is None:
            raise E1ClassifierError("classifier has not been fitted")
        return int(np.max(np.atleast_1d(self._model.n_iter_)))

    @property
    def model_size_bytes(self) -> int:
        if self._model is None:
            raise E1ClassifierError("classifier has not been fitted")
        buffer = io.BytesIO()
        pickle.dump(self._model, buffer, protocol=5)
        return len(buffer.getvalue())

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        font_ids: NDArray[np.str_],
        mapping_ids: NDArray[np.str_],
    ) -> "BalancedLinearSVC":
        x = _features(features)
        y = _labels(labels, x.shape[0], self._order)
        fonts = _ids(font_ids, "font_ids", x.shape[0])
        mappings = _ids(mapping_ids, "mapping_ids", x.shape[0])
        mapping_count = _require_cartesian_balance(y, fonts, mappings, self._order)
        sample_weight = np.full(x.shape[0], 1.0 / mapping_count, dtype=np.float64)
        model = LinearSVC(
            C=self.c,
            dual="auto",
            tol=1e-4,
            max_iter=20_000,
            random_state=0,
        )
        model.fit(x, y, sample_weight=sample_weight)
        if not np.array_equal(model.classes_, self._order):
            raise E1ClassifierError("fitted classes differ from class_order")
        self._model = model
        self._feature_count = int(x.shape[1])
        return self

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.int64]:
        x = _features(features)
        if self._model is None or self._feature_count is None:
            raise E1ClassifierError("classifier must be fitted before prediction")
        if x.shape[1] != self._feature_count:
            raise E1ClassifierError("prediction feature width differs from training")
        result = np.asarray(self._model.predict(x), dtype=np.int64)
        if result.shape != (x.shape[0],):
            raise E1ClassifierError("prediction shape is invalid")
        return np.ascontiguousarray(result)
