"""Locked multi-budget accuracy analysis.

The formal entry point is :func:`load_released_predictions`.  It validates a
50/50 release manifest *before* invoking the supplied prediction reader.  The
remaining functions are pure and are therefore safe to exercise with synthetic
fixtures during preflight.

All accuracy values use the unit interval.  Presentation code may multiply a
delta by 100 to display percentage points.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_COLUMNS = (
    "method",
    "budget",
    "seed",
    "font_id",
    "char_id",
    "y_true",
    "y_pred",
)
FORMAL_EXPECTED_CONDITIONS = 50
ANALYSIS_RELEASE_STATUS = "ANALYSIS_RELEASED"
COMMITTED_STATUS = "PASS_COMMITTED"


class AnalysisContractError(ValueError):
    """Raised when predictions do not match the frozen paired design."""


class ReleaseGateError(RuntimeError):
    """Raised before predictions are read when a formal release is incomplete."""


@dataclass(frozen=True)
class AnalysisSpec:
    """Frozen design and statistical choices for one analysis."""

    reference_method: str
    treatment_method: str
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    font_ids: tuple[str, ...]
    char_ids: tuple[str, ...]
    reference_budget: int = 64
    bootstrap_replicates: int = 100_000
    bootstrap_seed: int = 20260829
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        for name in ("reference_method", "treatment_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise AnalysisContractError(f"{name} must be a non-empty string")
        if self.reference_method == self.treatment_method:
            raise AnalysisContractError("reference and treatment methods must differ")

        _require_unique_nonempty("budgets", self.budgets)
        _require_unique_nonempty("seeds", self.seeds)
        _require_unique_nonempty("font_ids", self.font_ids)
        _require_unique_nonempty("char_ids", self.char_ids)
        if tuple(sorted(self.budgets)) != self.budgets:
            raise AnalysisContractError("budgets must be strictly increasing")
        if any(isinstance(value, bool) or int(value) <= 0 for value in self.budgets):
            raise AnalysisContractError("budgets must contain positive integers")
        if self.reference_budget not in self.budgets:
            raise AnalysisContractError("reference_budget must be one of budgets")
        if self.bootstrap_replicates <= 0:
            raise AnalysisContractError("bootstrap_replicates must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise AnalysisContractError("confidence_level must be between zero and one")

    @property
    def methods(self) -> tuple[str, str]:
        return (self.reference_method, self.treatment_method)

    @property
    def expected_prediction_rows(self) -> int:
        return (
            len(self.methods)
            * len(self.budgets)
            * len(self.seeds)
            * len(self.font_ids)
            * len(self.char_ids)
        )

    @property
    def expected_conditions(self) -> int:
        """Number of method-budget-seed runs in the frozen design."""

        return len(self.methods) * len(self.budgets) * len(self.seeds)


@dataclass(frozen=True)
class PredictionRow:
    method: str
    budget: int
    seed: int
    font_id: str
    char_id: str
    y_true: str
    y_pred: str

    @property
    def key(self) -> tuple[str, int, int, str, str]:
        return (self.method, self.budget, self.seed, self.font_id, self.char_id)


@dataclass(frozen=True)
class ValidatedPredictions:
    rows: tuple[PredictionRow, ...]
    spec: AnalysisSpec


@dataclass(frozen=True)
class AccuracyPoint:
    method: str
    budget: int
    n_samples: int
    n_correct: int
    accuracy: float


@dataclass(frozen=True)
class DeltaPoint:
    budget: int
    n_pairs: int
    reference_accuracy: float
    treatment_accuracy: float
    delta_accuracy: float


@dataclass(frozen=True)
class AubcPoint:
    method: str
    naubc: float


@dataclass(frozen=True)
class BEquivalent:
    reference_budget: int
    target_accuracy: float
    budget: int | None
    attained: bool
    treatment_accuracy: float | None
    status: str


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    lower: float
    upper: float


@dataclass(frozen=True)
class BootstrapResult:
    accuracy: dict[tuple[str, int], ConfidenceInterval]
    delta: dict[int, ConfidenceInterval]
    naubc: dict[str, ConfidenceInterval]
    delta_naubc: ConfidenceInterval
    replicates: int
    seed: int
    confidence_level: float
    cluster: str = "char_id"


@dataclass(frozen=True)
class AnalysisResult:
    accuracy: tuple[AccuracyPoint, ...]
    delta: tuple[DeltaPoint, ...]
    naubc: tuple[AubcPoint, ...]
    delta_naubc: float
    b_eq64: BEquivalent
    bootstrap: BootstrapResult | None


RawPrediction = Mapping[str, object] | PredictionRow
PredictionReader = Callable[[], Iterable[RawPrediction]]


def _require_unique_nonempty(name: str, values: Sequence[object]) -> None:
    if not values:
        raise AnalysisContractError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise AnalysisContractError(f"{name} must not contain duplicates")


def _nonempty_text(value: object, field: str, row_number: int) -> str:
    if value is None:
        raise AnalysisContractError(f"row {row_number}: {field} is null")
    text = str(value)
    if not text.strip():
        raise AnalysisContractError(f"row {row_number}: {field} is empty")
    return text


def _integer(value: object, field: str, row_number: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise AnalysisContractError(f"row {row_number}: {field} is not an integer")
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise AnalysisContractError(f"row {row_number}: {field} is not an integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisContractError(
            f"row {row_number}: {field} is not an integer"
        ) from exc
    return parsed


def _coerce_prediction(row: RawPrediction, row_number: int) -> PredictionRow:
    if isinstance(row, PredictionRow):
        return row
    missing = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing:
        raise AnalysisContractError(
            f"row {row_number}: missing required columns {missing}"
        )
    return PredictionRow(
        method=_nonempty_text(row["method"], "method", row_number),
        budget=_integer(row["budget"], "budget", row_number),
        seed=_integer(row["seed"], "seed", row_number),
        font_id=_nonempty_text(row["font_id"], "font_id", row_number),
        char_id=_nonempty_text(row["char_id"], "char_id", row_number),
        y_true=_nonempty_text(row["y_true"], "y_true", row_number),
        y_pred=_nonempty_text(row["y_pred"], "y_pred", row_number),
    )


def validate_predictions(
    predictions: Iterable[RawPrediction], spec: AnalysisSpec
) -> ValidatedPredictions:
    """Validate the exact balanced and paired prediction cross-product."""

    rows: list[PredictionRow] = []
    actual_keys: set[tuple[str, int, int, str, str]] = set()
    for row_number, raw in enumerate(predictions, start=2):
        row = _coerce_prediction(raw, row_number)
        if row.key in actual_keys:
            raise AnalysisContractError(f"duplicate prediction key: {row.key!r}")
        actual_keys.add(row.key)
        rows.append(row)

    if len(rows) != spec.expected_prediction_rows:
        raise AnalysisContractError(
            "prediction row count mismatch: "
            f"expected {spec.expected_prediction_rows}, observed {len(rows)}"
        )

    allowed_methods = set(spec.methods)
    allowed_budgets = set(spec.budgets)
    allowed_seeds = set(spec.seeds)
    allowed_fonts = set(spec.font_ids)
    allowed_chars = set(spec.char_ids)
    for row in rows:
        if row.method not in allowed_methods:
            raise AnalysisContractError(f"unexpected method: {row.method!r}")
        if row.budget not in allowed_budgets:
            raise AnalysisContractError(f"unexpected budget: {row.budget!r}")
        if row.seed not in allowed_seeds:
            raise AnalysisContractError(f"unexpected seed: {row.seed!r}")
        if row.font_id not in allowed_fonts:
            raise AnalysisContractError(f"unexpected font_id: {row.font_id!r}")
        if row.char_id not in allowed_chars:
            raise AnalysisContractError(f"unexpected char_id: {row.char_id!r}")

    expected_keys = set(
        product(
            spec.methods,
            spec.budgets,
            spec.seeds,
            spec.font_ids,
            spec.char_ids,
        )
    )
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys, key=repr)
        extra = sorted(actual_keys - expected_keys, key=repr)
        raise AnalysisContractError(
            "prediction design is not the exact frozen cross-product; "
            f"missing={missing[:3]!r}, extra={extra[:3]!r}"
        )

    truth_by_char: dict[str, str] = {}
    char_by_truth: dict[str, str] = {}
    for row in rows:
        previous_truth = truth_by_char.setdefault(row.char_id, row.y_true)
        if previous_truth != row.y_true:
            raise AnalysisContractError(
                f"char_id {row.char_id!r} maps to multiple y_true labels"
            )
        previous_char = char_by_truth.setdefault(row.y_true, row.char_id)
        if previous_char != row.char_id:
            raise AnalysisContractError(
                f"y_true {row.y_true!r} maps to multiple char_id values"
            )
    allowed_labels = set(truth_by_char.values())
    unknown_predictions = sorted(
        {row.y_pred for row in rows if row.y_pred not in allowed_labels}
    )
    if unknown_predictions:
        raise AnalysisContractError(
            f"y_pred contains labels outside the frozen class set: "
            f"{unknown_predictions[:3]!r}"
        )

    method_order = {value: index for index, value in enumerate(spec.methods)}
    budget_order = {value: index for index, value in enumerate(spec.budgets)}
    seed_order = {value: index for index, value in enumerate(spec.seeds)}
    font_order = {value: index for index, value in enumerate(spec.font_ids)}
    char_order = {value: index for index, value in enumerate(spec.char_ids)}
    rows.sort(
        key=lambda row: (
            char_order[row.char_id],
            method_order[row.method],
            budget_order[row.budget],
            seed_order[row.seed],
            font_order[row.font_id],
        )
    )
    return ValidatedPredictions(rows=tuple(rows), spec=spec)


def _correctness_tensor(validated: ValidatedPredictions) -> np.ndarray:
    spec = validated.spec
    tensor = np.empty(
        (
            len(spec.char_ids),
            len(spec.methods),
            len(spec.budgets),
            len(spec.seeds),
            len(spec.font_ids),
        ),
        dtype=np.int8,
    )
    method_index = {value: index for index, value in enumerate(spec.methods)}
    budget_index = {value: index for index, value in enumerate(spec.budgets)}
    seed_index = {value: index for index, value in enumerate(spec.seeds)}
    font_index = {value: index for index, value in enumerate(spec.font_ids)}
    char_index = {value: index for index, value in enumerate(spec.char_ids)}
    for row in validated.rows:
        tensor[
            char_index[row.char_id],
            method_index[row.method],
            budget_index[row.budget],
            seed_index[row.seed],
            font_index[row.font_id],
        ] = int(row.y_pred == row.y_true)
    return tensor


def accuracy_budget(validated: ValidatedPredictions) -> tuple[AccuracyPoint, ...]:
    """Return pooled Accuracy--Budget points for each frozen method."""

    tensor = _correctness_tensor(validated)
    spec = validated.spec
    n_samples = len(spec.char_ids) * len(spec.seeds) * len(spec.font_ids)
    points: list[AccuracyPoint] = []
    for method_index, method in enumerate(spec.methods):
        for budget_index, budget in enumerate(spec.budgets):
            n_correct = int(tensor[:, method_index, budget_index, :, :].sum())
            points.append(
                AccuracyPoint(
                    method=method,
                    budget=budget,
                    n_samples=n_samples,
                    n_correct=n_correct,
                    accuracy=n_correct / n_samples,
                )
            )
    return tuple(points)


def delta_budget(validated: ValidatedPredictions) -> tuple[DeltaPoint, ...]:
    """Return paired treatment-minus-reference accuracy at every budget."""

    tensor = _correctness_tensor(validated)
    spec = validated.spec
    n_pairs = len(spec.char_ids) * len(spec.seeds) * len(spec.font_ids)
    points: list[DeltaPoint] = []
    for budget_index, budget in enumerate(spec.budgets):
        reference_correct = int(tensor[:, 0, budget_index, :, :].sum())
        treatment_correct = int(tensor[:, 1, budget_index, :, :].sum())
        reference_accuracy = reference_correct / n_pairs
        treatment_accuracy = treatment_correct / n_pairs
        paired_difference = (
            tensor[:, 1, budget_index, :, :]
            - tensor[:, 0, budget_index, :, :]
        )
        points.append(
            DeltaPoint(
                budget=budget,
                n_pairs=n_pairs,
                reference_accuracy=reference_accuracy,
                treatment_accuracy=treatment_accuracy,
                delta_accuracy=float(paired_difference.mean(dtype=np.float64)),
            )
        )
    return tuple(points)


def normalized_aubc(
    budgets: Sequence[int], accuracies: Sequence[float]
) -> float:
    """Trapezoidal area normalized by the tested budget span."""

    budget_array = np.asarray(budgets, dtype=np.float64)
    accuracy_array = np.asarray(accuracies, dtype=np.float64)
    if budget_array.ndim != 1 or accuracy_array.ndim != 1:
        raise AnalysisContractError("budgets and accuracies must be one-dimensional")
    if len(budget_array) != len(accuracy_array) or len(budget_array) < 2:
        raise AnalysisContractError(
            "budgets and accuracies must have equal length of at least two"
        )
    widths = np.diff(budget_array)
    if np.any(widths <= 0):
        raise AnalysisContractError("budgets must be strictly increasing")
    if not np.all(np.isfinite(accuracy_array)) or np.any(
        (accuracy_array < 0.0) | (accuracy_array > 1.0)
    ):
        raise AnalysisContractError("accuracies must be finite values in [0, 1]")
    area = np.sum(widths * (accuracy_array[:-1] + accuracy_array[1:]) / 2.0)
    return float(area / (budget_array[-1] - budget_array[0]))


def aubc_by_method(
    curve: Sequence[AccuracyPoint], spec: AnalysisSpec
) -> tuple[AubcPoint, ...]:
    lookup = {(point.method, point.budget): point.accuracy for point in curve}
    result: list[AubcPoint] = []
    for method in spec.methods:
        try:
            accuracies = [lookup[(method, budget)] for budget in spec.budgets]
        except KeyError as exc:
            raise AnalysisContractError("accuracy curve is incomplete") from exc
        result.append(
            AubcPoint(
                method=method,
                naubc=normalized_aubc(spec.budgets, accuracies),
            )
        )
    return tuple(result)


def b_eq64(curve: Sequence[AccuracyPoint], spec: AnalysisSpec) -> BEquivalent:
    """Return the lowest tested MMA budget reaching ideal-only at B=64.

    The function uses point estimates only and never interpolates.  Despite the
    historical name, ``spec.reference_budget`` is recorded in the result.
    """

    lookup = {(point.method, point.budget): point.accuracy for point in curve}
    try:
        target = lookup[(spec.reference_method, spec.reference_budget)]
    except KeyError as exc:
        raise AnalysisContractError("reference curve lacks the reference budget") from exc
    for budget in spec.budgets:
        key = (spec.treatment_method, budget)
        if key not in lookup:
            raise AnalysisContractError("treatment accuracy curve is incomplete")
        treatment_accuracy = lookup[key]
        if treatment_accuracy >= target:
            return BEquivalent(
                reference_budget=spec.reference_budget,
                target_accuracy=target,
                budget=budget,
                attained=True,
                treatment_accuracy=treatment_accuracy,
                status="attained_at_tested_budget",
            )
    return BEquivalent(
        reference_budget=spec.reference_budget,
        target_accuracy=target,
        budget=None,
        attained=False,
        treatment_accuracy=None,
        status="not_attained_within_tested_range",
    )


def _confidence_interval(
    samples: np.ndarray, point: float, confidence_level: float
) -> ConfidenceInterval:
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        samples,
        np.asarray([alpha, 1.0 - alpha], dtype=np.float64),
        method="linear",
    )
    return ConfidenceInterval(point=point, lower=float(lower), upper=float(upper))


def paired_character_cluster_bootstrap(
    validated: ValidatedPredictions,
    *,
    replicates: int | None = None,
    seed: int | None = None,
    batch_size: int = 2048,
) -> BootstrapResult:
    """Paired percentile bootstrap over character clusters.

    A sampled ``char_id`` carries all methods, budgets, seeds, and fonts.  One
    character-resampling weight vector is reused for every statistic in a
    replicate, preserving the full paired curve design.
    """

    spec = validated.spec
    n_replicates = spec.bootstrap_replicates if replicates is None else replicates
    rng_seed = spec.bootstrap_seed if seed is None else seed
    if n_replicates <= 0:
        raise AnalysisContractError("replicates must be positive")
    if batch_size <= 0:
        raise AnalysisContractError("batch_size must be positive")

    tensor = _correctness_tensor(validated)
    # char_correct: character x method x budget, with every font/seed retained.
    char_correct = tensor.sum(axis=(3, 4), dtype=np.int64)
    n_chars = len(spec.char_ids)
    observations_per_char = len(spec.seeds) * len(spec.font_ids)
    denominator = n_chars * observations_per_char
    draws = np.empty(
        (n_replicates, len(spec.methods), len(spec.budgets)), dtype=np.float64
    )
    flat_correct = char_correct.reshape(n_chars, -1)
    probabilities = np.full(n_chars, 1.0 / n_chars, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    for start in range(0, n_replicates, batch_size):
        stop = min(start + batch_size, n_replicates)
        weights = rng.multinomial(
            n_chars, probabilities, size=stop - start
        ).astype(np.int64, copy=False)
        numerators = weights @ flat_correct
        draws[start:stop] = numerators.reshape(
            stop - start, len(spec.methods), len(spec.budgets)
        ) / denominator

    point_accuracy = tensor.mean(axis=(0, 3, 4), dtype=np.float64)
    delta_draws = draws[:, 1, :] - draws[:, 0, :]
    point_delta = point_accuracy[1, :] - point_accuracy[0, :]
    widths = np.diff(np.asarray(spec.budgets, dtype=np.float64))
    budget_span = float(spec.budgets[-1] - spec.budgets[0])
    aubc_draws = np.sum(
        widths[None, None, :]
        * (draws[:, :, :-1] + draws[:, :, 1:])
        / 2.0,
        axis=2,
    ) / budget_span
    point_aubc = np.sum(
        widths[None, :]
        * (point_accuracy[:, :-1] + point_accuracy[:, 1:])
        / 2.0,
        axis=1,
    ) / budget_span
    delta_aubc_draws = aubc_draws[:, 1] - aubc_draws[:, 0]
    point_delta_aubc = float(point_aubc[1] - point_aubc[0])

    accuracy_intervals: dict[tuple[str, int], ConfidenceInterval] = {}
    for method_index, method in enumerate(spec.methods):
        for budget_index, budget in enumerate(spec.budgets):
            accuracy_intervals[(method, budget)] = _confidence_interval(
                draws[:, method_index, budget_index],
                float(point_accuracy[method_index, budget_index]),
                spec.confidence_level,
            )
    delta_intervals = {
        budget: _confidence_interval(
            delta_draws[:, budget_index],
            float(point_delta[budget_index]),
            spec.confidence_level,
        )
        for budget_index, budget in enumerate(spec.budgets)
    }
    aubc_intervals = {
        method: _confidence_interval(
            aubc_draws[:, method_index],
            float(point_aubc[method_index]),
            spec.confidence_level,
        )
        for method_index, method in enumerate(spec.methods)
    }
    return BootstrapResult(
        accuracy=accuracy_intervals,
        delta=delta_intervals,
        naubc=aubc_intervals,
        delta_naubc=_confidence_interval(
            delta_aubc_draws, point_delta_aubc, spec.confidence_level
        ),
        replicates=n_replicates,
        seed=rng_seed,
        confidence_level=spec.confidence_level,
    )


def analyze_predictions(
    predictions: Iterable[RawPrediction],
    spec: AnalysisSpec,
    *,
    include_bootstrap: bool = True,
) -> AnalysisResult:
    """Validate and compute all preregistered point and interval estimates."""

    validated = validate_predictions(predictions, spec)
    curve = accuracy_budget(validated)
    deltas = delta_budget(validated)
    aubcs = aubc_by_method(curve, spec)
    delta_naubc = aubcs[1].naubc - aubcs[0].naubc
    bootstrap = (
        paired_character_cluster_bootstrap(validated)
        if include_bootstrap
        else None
    )
    return AnalysisResult(
        accuracy=curve,
        delta=deltas,
        naubc=aubcs,
        delta_naubc=delta_naubc,
        b_eq64=b_eq64(curve, spec),
        bootstrap=bootstrap,
    )


def _manifest_int(manifest: Mapping[str, Any], field: str) -> int:
    value = manifest.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ReleaseGateError(f"release manifest field {field!r} must be an integer")
    return int(value)


def require_analysis_release(
    manifest: Mapping[str, Any],
    *,
    expected_conditions: int = FORMAL_EXPECTED_CONDITIONS,
) -> None:
    """Fail closed unless the frozen formal experiment is exactly 50/50.

    This function performs no prediction I/O and must be called before a
    prediction reader is invoked.
    """

    if manifest.get("status") != ANALYSIS_RELEASE_STATUS:
        raise ReleaseGateError(
            f"formal predictions are not released for analysis: "
            f"status={manifest.get('status')!r}"
        )
    manifest_expected = _manifest_int(manifest, "expected_conditions")
    completed = _manifest_int(manifest, "completed_conditions")
    unexpected = _manifest_int(manifest, "unexpected_conditions")
    if manifest_expected != expected_conditions:
        raise ReleaseGateError(
            f"expected_conditions must equal frozen value {expected_conditions}"
        )
    if completed != expected_conditions:
        raise ReleaseGateError(
            f"formal analysis requires {expected_conditions}/{expected_conditions}; "
            f"observed {completed}/{expected_conditions}"
        )
    if unexpected != 0:
        raise ReleaseGateError("release manifest contains unexpected conditions")
    if manifest.get("hashes_verified") is not True:
        raise ReleaseGateError("frozen hashes have not been verified")

    statuses = manifest.get("condition_statuses")
    if not isinstance(statuses, (list, tuple)):
        raise ReleaseGateError("condition_statuses must be a sequence")
    if len(statuses) != expected_conditions or any(
        status != COMMITTED_STATUS for status in statuses
    ):
        raise ReleaseGateError(
            f"all {expected_conditions} conditions must be {COMMITTED_STATUS}"
        )

    run_ids = manifest.get("run_ids")
    frozen_run_ids = manifest.get("frozen_run_ids")
    if not isinstance(run_ids, (list, tuple)) or not isinstance(
        frozen_run_ids, (list, tuple)
    ):
        raise ReleaseGateError("run_ids and frozen_run_ids must be sequences")
    if len(run_ids) != expected_conditions or len(frozen_run_ids) != expected_conditions:
        raise ReleaseGateError(
            f"run_ids and frozen_run_ids must each contain {expected_conditions} entries"
        )
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids) or any(
        not isinstance(run_id, str) or not run_id for run_id in frozen_run_ids
    ):
        raise ReleaseGateError("run_ids and frozen_run_ids must be non-empty strings")
    if len(set(run_ids)) != len(run_ids) or len(set(frozen_run_ids)) != len(
        frozen_run_ids
    ):
        raise ReleaseGateError("run_ids and frozen_run_ids must not contain duplicates")
    if tuple(run_ids) != tuple(frozen_run_ids):
        raise ReleaseGateError("run_ids do not exactly match frozen_run_ids")


def load_released_predictions(
    manifest: Mapping[str, Any],
    prediction_reader: PredictionReader,
    spec: AnalysisSpec,
    *,
    expected_conditions: int = FORMAL_EXPECTED_CONDITIONS,
) -> ValidatedPredictions:
    """Gate first, then invoke ``prediction_reader`` exactly once and validate."""

    require_analysis_release(manifest, expected_conditions=expected_conditions)
    if spec.expected_conditions != expected_conditions:
        raise ReleaseGateError(
            "analysis spec condition count does not match the frozen release: "
            f"expected {expected_conditions}, observed {spec.expected_conditions}"
        )
    predictions = prediction_reader()
    return validate_predictions(predictions, spec)


def read_predictions_csv(path: str | Path) -> tuple[dict[str, str], ...]:
    """Read prediction rows without computing or logging any metric."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AnalysisContractError("prediction CSV has no header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise AnalysisContractError(
                f"prediction CSV is missing required columns: {missing}"
            )
        return tuple(dict(row) for row in reader)
