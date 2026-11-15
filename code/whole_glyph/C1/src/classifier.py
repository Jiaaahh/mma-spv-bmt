"""Fixed LinearSVC decoder used by the CPU-only MVP."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from src.contracts import LabelVector


class ClassifierError(ValueError):
    """Raised when classifier data or state violates the frozen contract."""


def _validate_features(features: np.ndarray, *, allow_empty: bool = False) -> None:
    if not isinstance(features, np.ndarray):
        raise TypeError("features must be a numpy array")
    if features.dtype != np.float32 or features.ndim != 2:
        raise TypeError("features must be a two-dimensional float32 array")
    if (not allow_empty and features.shape[0] == 0) or features.shape[1] == 0:
        raise ClassifierError("features must not be empty")
    if not np.isfinite(features).all():
        raise ClassifierError("features contain non-finite values")


def _validate_labels(labels: np.ndarray, sample_count: int) -> None:
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy array")
    if labels.dtype != np.int64 or labels.ndim != 1:
        raise TypeError("labels must be a one-dimensional int64 array")
    if labels.shape != (sample_count,):
        raise ClassifierError("label count does not match the feature rows")
    if labels.size == 0 or np.any(labels < 0):
        raise ClassifierError("labels must contain non-negative class IDs")


class LinearSvcClassifier:
    """Validated wrapper around scikit-learn's CPU LinearSVC."""

    name = "linear_svc"

    def __init__(
        self,
        *,
        c: float = 1.0,
        dual: str = "auto",
        max_iter: int = 5000,
        random_state: int = 2026,
    ) -> None:
        if isinstance(c, bool) or not isinstance(c, (int, float)) or c <= 0:
            raise ClassifierError("c must be positive")
        if dual != "auto":
            raise ClassifierError("MVP LinearSVC requires dual='auto'")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
            raise ClassifierError("max_iter must be a positive integer")
        if (
            isinstance(random_state, bool)
            or not isinstance(random_state, int)
            or random_state < 0
        ):
            raise ClassifierError("random_state must be a non-negative integer")
        self.c = float(c)
        self.dual = dual
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = LinearSVC(
            C=self.c,
            dual=self.dual,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._feature_count: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self._feature_count is not None

    @property
    def n_iter(self) -> int:
        if not self.is_fitted:
            raise ClassifierError("classifier has not been fitted")
        return int(np.max(np.atleast_1d(self._model.n_iter_)))

    def fit(
        self, features: NDArray[np.float32], labels: LabelVector
    ) -> "LinearSvcClassifier":
        _validate_features(features)
        _validate_labels(labels, features.shape[0])
        if np.unique(labels).size < 2:
            raise ClassifierError("training data must contain at least two classes")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            self._model.fit(features, labels)
        if any(
            issubclass(item.category, ConvergenceWarning) for item in captured
        ):
            raise ClassifierError(
                f"LinearSVC did not converge within max_iter={self.max_iter}"
            )
        self._feature_count = int(features.shape[1])
        return self

    def predict(self, features: NDArray[np.float32]) -> LabelVector:
        _validate_features(features)
        if self._feature_count is None:
            raise ClassifierError("classifier must be fitted before prediction")
        if features.shape[1] != self._feature_count:
            raise ClassifierError("prediction feature width differs from training")
        predictions = np.asarray(self._model.predict(features), dtype=np.int64)
        if predictions.shape != (features.shape[0],):
            raise ClassifierError("LinearSVC returned an invalid prediction shape")
        return np.ascontiguousarray(predictions, dtype=np.int64)


def classification_accuracy(
    true_labels: LabelVector, predicted_labels: LabelVector
) -> float:
    if not isinstance(true_labels, np.ndarray):
        raise TypeError("true_labels must be a numpy array")
    if true_labels.ndim != 1:
        raise TypeError("true_labels must be one-dimensional")
    _validate_labels(true_labels, int(true_labels.shape[0]))
    _validate_labels(predicted_labels, int(true_labels.shape[0]))
    if true_labels.size == 0:
        raise ClassifierError("accuracy requires at least one sample")
    return float(np.mean(true_labels == predicted_labels))
