"""Validated paired statistics for the independent E1 confirmation.

This module deliberately has no experiment-running responsibilities.  It accepts
already-produced prediction rows, verifies the frozen Cartesian evaluation
design, computes seed-level metrics and paired comparisons, and writes report
artifacts atomically.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


EXPECTED_FONT_COUNT = 4
EXPECTED_SEED_COUNT = 5
EXPECTED_CLASS_COUNT = 180
BOOTSTRAP_ITERATIONS = 100_000
BOOTSTRAP_SEED = 20_260_826
PRIMARY_CONDITION = "moderate"
PRIMARY_METHOD = "proto_skeleton_ideal_plus_multi4"
PRIMARY_REFERENCE_METHOD = "proto_skeleton_ideal_only"

_REQUIRED_FIELDS = frozenset(
    {
        "method",
        "condition",
        "mapping_seed",
        "font_id",
        "true_label",
        "predicted_label",
        "character",
        "correct",
    }
)


class E1StatisticsError(ValueError):
    """Raised when prediction rows violate the E1 statistical contract."""


def _nonempty_text(value: Any, name: str, row_index: int) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise E1StatisticsError(
            f"row {row_index}: {name} must be a non-empty stripped string"
        )
    return value


def _integer(value: Any, name: str, row_index: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise E1StatisticsError(f"row {row_index}: {name} must be an integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as error:
            raise E1StatisticsError(
                f"row {row_index}: {name} must be an integer"
            ) from error
        if str(parsed) != value.strip():
            raise E1StatisticsError(
                f"row {row_index}: {name} must use canonical integer text"
            )
        return parsed
    raise E1StatisticsError(f"row {row_index}: {name} must be an integer")


def _correct(value: Any, row_index: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    parsed = _integer(value, "correct", row_index)
    if parsed not in (0, 1):
        raise E1StatisticsError(f"row {row_index}: correct must be 0 or 1")
    return parsed


def validate_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_font_count: int = EXPECTED_FONT_COUNT,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
    class_count: int = EXPECTED_CLASS_COUNT,
) -> tuple[dict[str, Any], ...]:
    """Normalize and validate complete method/condition Cartesian evaluations.

    Every method within every condition must contain exactly one prediction for
    each mapping-seed x font x true-class cell.  All methods in a condition must
    use the same seeds and fonts, which makes subsequent differences genuinely
    paired rather than merely differences between aggregate means.
    """

    for name, value in (
        ("expected_font_count", expected_font_count),
        ("expected_seed_count", expected_seed_count),
        ("class_count", class_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise E1StatisticsError(f"{name} must be a positive integer")

    normalized: list[dict[str, Any]] = []
    group_cells: dict[
        tuple[str, str], dict[tuple[int, str, int], dict[str, Any]]
    ] = defaultdict(dict)
    character_by_label: dict[int, str] = {}

    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise E1StatisticsError(f"row {index}: prediction row must be a mapping")
        missing = _REQUIRED_FIELDS.difference(source)
        if missing:
            raise E1StatisticsError(
                f"row {index}: missing required fields {sorted(missing)}"
            )
        method = _nonempty_text(source["method"], "method", index)
        condition = _nonempty_text(source["condition"], "condition", index)
        mapping_seed = _integer(source["mapping_seed"], "mapping_seed", index)
        font_id = _nonempty_text(source["font_id"], "font_id", index)
        true_label = _integer(source["true_label"], "true_label", index)
        predicted_label = _integer(
            source["predicted_label"], "predicted_label", index
        )
        character = _nonempty_text(source["character"], "character", index)
        correct = _correct(source["correct"], index)
        if not 0 <= true_label < class_count:
            raise E1StatisticsError(
                f"row {index}: true_label is outside [0, {class_count - 1}]"
            )
        if not 0 <= predicted_label < class_count:
            raise E1StatisticsError(
                f"row {index}: predicted_label is outside [0, {class_count - 1}]"
            )
        expected_correct = int(true_label == predicted_label)
        if correct != expected_correct:
            raise E1StatisticsError(
                f"row {index}: correct disagrees with true/predicted labels"
            )
        previous_character = character_by_label.setdefault(true_label, character)
        if previous_character != character:
            raise E1StatisticsError(
                f"row {index}: character is inconsistent for true_label {true_label}"
            )
        row = {
            "method": method,
            "condition": condition,
            "mapping_seed": mapping_seed,
            "font_id": font_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "character": character,
            "correct": correct,
        }
        cell = (mapping_seed, font_id, true_label)
        group = group_cells[(condition, method)]
        if cell in group:
            raise E1StatisticsError(
                "duplicate prediction cell for "
                f"condition={condition}, method={method}, seed={mapping_seed}, "
                f"font={font_id}, class={true_label}"
            )
        group[cell] = row
        normalized.append(row)

    if not normalized:
        raise E1StatisticsError("prediction rows must not be empty")
    expected_labels = set(range(class_count))
    design_by_condition: dict[str, tuple[set[int], set[str]]] = {}
    for (condition, method), cells in group_cells.items():
        seeds = {cell[0] for cell in cells}
        fonts = {cell[1] for cell in cells}
        labels = {cell[2] for cell in cells}
        if len(seeds) != expected_seed_count:
            raise E1StatisticsError(
                f"{condition}/{method} requires exactly {expected_seed_count} seeds"
            )
        if len(fonts) != expected_font_count:
            raise E1StatisticsError(
                f"{condition}/{method} requires exactly {expected_font_count} fonts"
            )
        if labels != expected_labels:
            raise E1StatisticsError(
                f"{condition}/{method} requires classes 0..{class_count - 1}"
            )
        expected_cells = {
            (seed, font, label)
            for seed in seeds
            for font in fonts
            for label in range(class_count)
        }
        if set(cells) != expected_cells:
            raise E1StatisticsError(
                f"{condition}/{method} is not a complete seed x font x class grid"
            )
        design = design_by_condition.setdefault(condition, (seeds, fonts))
        if design != (seeds, fonts):
            raise E1StatisticsError(
                f"all methods in condition {condition} must share seeds and fonts"
            )

    return tuple(
        sorted(
            normalized,
            key=lambda row: (
                row["condition"],
                row["method"],
                row["mapping_seed"],
                row["font_id"],
                row["true_label"],
            ),
        )
    )


def _seed_metrics_from_validated(
    rows: Sequence[Mapping[str, Any]], *, class_count: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["method"], row["mapping_seed"])].append(row)
    result: list[dict[str, Any]] = []
    labels = np.arange(class_count, dtype=np.int64)
    for (condition, method, mapping_seed), group in sorted(groups.items()):
        true = np.asarray([row["true_label"] for row in group], dtype=np.int64)
        predicted = np.asarray(
            [row["predicted_label"] for row in group], dtype=np.int64
        )
        result.append(
            {
                "condition": condition,
                "method": method,
                "mapping_seed": mapping_seed,
                "sample_count": len(group),
                "accuracy": float(np.mean(true == predicted)),
                "macro_f1": float(
                    f1_score(
                        true,
                        predicted,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    return result


def compute_seed_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_font_count: int = EXPECTED_FONT_COUNT,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
    class_count: int = EXPECTED_CLASS_COUNT,
) -> list[dict[str, Any]]:
    """Return Accuracy and Macro-F1 for every condition/method/mapping seed."""

    validated = validate_prediction_rows(
        rows,
        expected_font_count=expected_font_count,
        expected_seed_count=expected_seed_count,
        class_count=class_count,
    )
    return _seed_metrics_from_validated(validated, class_count=class_count)


def _paired_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    method: str,
    reference_method: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if method == reference_method:
        raise E1StatisticsError("paired methods must be different")
    selected = [row for row in rows if row["condition"] == condition]
    method_rows = [row for row in selected if row["method"] == method]
    reference_rows = [row for row in selected if row["method"] == reference_method]
    if not method_rows:
        raise E1StatisticsError(f"missing method {method!r} in condition {condition!r}")
    if not reference_rows:
        raise E1StatisticsError(
            f"missing reference method {reference_method!r} in condition {condition!r}"
        )
    def key(row: Mapping[str, Any]) -> tuple[int, str, int]:
        return (row["mapping_seed"], row["font_id"], row["true_label"])

    method_by_key = {key(row): row for row in method_rows}
    reference_by_key = {key(row): row for row in reference_rows}
    if method_by_key.keys() != reference_by_key.keys():
        raise E1StatisticsError("paired methods do not share identical evaluation cells")
    ordered_keys = sorted(method_by_key)
    return (
        [method_by_key[cell] for cell in ordered_keys],
        [reference_by_key[cell] for cell in ordered_keys],
    )


def _bootstrap_cluster_values(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    cluster_count = values.size
    chunk_size = 10_000
    for start in range(0, iterations, chunk_size):
        stop = min(iterations, start + chunk_size)
        indices = rng.integers(
            0, cluster_count, size=(stop - start, cluster_count), dtype=np.int64
        )
        estimates[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.percentile(estimates, (2.5, 97.5))
    return float(lower), float(upper)


def _paired_cluster_bootstrap_from_validated(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    method: str,
    reference_method: str,
    cluster_field: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if cluster_field not in {"character", "true_label", "font_id", "mapping_seed"}:
        raise E1StatisticsError(
            "cluster_field must be character, true_label, font_id, or mapping_seed"
        )
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise E1StatisticsError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise E1StatisticsError("seed must be a non-negative integer")
    method_rows, reference_rows = _paired_rows(
        rows,
        condition=condition,
        method=method,
        reference_method=reference_method,
    )
    grouped: dict[Any, list[float]] = defaultdict(list)
    for method_row, reference_row in zip(method_rows, reference_rows, strict=True):
        cluster = method_row[cluster_field]
        if cluster != reference_row[cluster_field]:
            raise E1StatisticsError("cluster labels differ across paired methods")
        grouped[cluster].append(
            float(method_row["correct"] - reference_row["correct"])
        )
    cell_counts = {len(values) for values in grouped.values()}
    if len(cell_counts) != 1:
        raise E1StatisticsError("every bootstrap cluster must contain the same cells")
    cluster_values = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped, key=str)],
        dtype=np.float64,
    )
    if cluster_values.size < 2:
        raise E1StatisticsError("paired bootstrap requires at least two clusters")
    lower, upper = _bootstrap_cluster_values(
        cluster_values, iterations=iterations, seed=seed
    )
    return {
        "unit": cluster_field,
        "condition": condition,
        "method": method,
        "reference_method": reference_method,
        "cluster_count": int(cluster_values.size),
        "within_cluster_cells": int(next(iter(cell_counts))),
        "iterations": iterations,
        "seed": seed,
        "mean_difference_pp": float(cluster_values.mean() * 100.0),
        "ci95_pp": [lower * 100.0, upper * 100.0],
    }


def paired_cluster_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    condition: str,
    method: str,
    reference_method: str,
    cluster_field: str = "character",
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    expected_font_count: int = EXPECTED_FONT_COUNT,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
    class_count: int = EXPECTED_CLASS_COUNT,
) -> dict[str, Any]:
    """Paired percentile bootstrap after aggregating within arbitrary clusters."""

    validated = validate_prediction_rows(
        rows,
        expected_font_count=expected_font_count,
        expected_seed_count=expected_seed_count,
        class_count=class_count,
    )
    return _paired_cluster_bootstrap_from_validated(
        validated,
        condition=condition,
        method=method,
        reference_method=reference_method,
        cluster_field=cluster_field,
        iterations=iterations,
        seed=seed,
    )


def _comparison_from_validated(
    rows: Sequence[Mapping[str, Any]],
    seed_metrics: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    method: str,
    reference_method: str,
) -> dict[str, Any]:
    method_rows, reference_rows = _paired_rows(
        rows,
        condition=condition,
        method=method,
        reference_method=reference_method,
    )
    differences_by_seed: dict[int, list[float]] = defaultdict(list)
    for method_row, reference_row in zip(method_rows, reference_rows, strict=True):
        differences_by_seed[method_row["mapping_seed"]].append(
            float(method_row["correct"] - reference_row["correct"])
        )

    metrics_by_key = {
        (row["condition"], row["method"], row["mapping_seed"]): row
        for row in seed_metrics
    }
    per_seed: dict[str, dict[str, float]] = {}
    for mapping_seed, differences in sorted(differences_by_seed.items()):
        method_metric = metrics_by_key[(condition, method, mapping_seed)]
        reference_metric = metrics_by_key[
            (condition, reference_method, mapping_seed)
        ]
        per_seed[str(mapping_seed)] = {
            "accuracy_difference_pp": float(np.mean(differences) * 100.0),
            "macro_f1_difference": float(
                method_metric["macro_f1"] - reference_metric["macro_f1"]
            ),
        }
    accuracy_values = [
        value["accuracy_difference_pp"] for value in per_seed.values()
    ]
    f1_values = [value["macro_f1_difference"] for value in per_seed.values()]
    return {
        "condition": condition,
        "method": method,
        "reference_method": reference_method,
        "mean_accuracy_difference_pp": float(np.mean(accuracy_values)),
        "mean_macro_f1_difference": float(np.mean(f1_values)),
        "per_seed": per_seed,
    }


def build_statistics_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    comparisons: Sequence[tuple[str, str, str]] | None = None,
    primary_condition: str = PRIMARY_CONDITION,
    primary_method: str = PRIMARY_METHOD,
    primary_reference_method: str = PRIMARY_REFERENCE_METHOD,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_font_count: int = EXPECTED_FONT_COUNT,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
    class_count: int = EXPECTED_CLASS_COUNT,
) -> dict[str, Any]:
    """Build the complete E1 statistical report without writing any files.

    ``comparisons`` accepts any number of ``(condition, method, reference)``
    triples.  The frozen primary comparison is always included and receives the
    100,000-resample character-cluster bootstrap.
    """

    validated = validate_prediction_rows(
        rows,
        expected_font_count=expected_font_count,
        expected_seed_count=expected_seed_count,
        class_count=class_count,
    )
    seed_metrics = _seed_metrics_from_validated(validated, class_count=class_count)

    conditions: dict[str, dict[str, Any]] = {}
    grouped_metrics: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_metrics:
        grouped_metrics[(row["condition"], row["method"])].append(row)
    for (condition, method), method_metrics in sorted(grouped_metrics.items()):
        condition_result = conditions.setdefault(condition, {"methods": {}})
        condition_result["methods"][method] = {
            "mean_accuracy": float(
                np.mean([row["accuracy"] for row in method_metrics])
            ),
            "mean_macro_f1": float(
                np.mean([row["macro_f1"] for row in method_metrics])
            ),
            "per_seed": {
                str(row["mapping_seed"]): {
                    "accuracy": row["accuracy"],
                    "macro_f1": row["macro_f1"],
                }
                for row in method_metrics
            },
        }

    primary_key = (
        primary_condition,
        primary_method,
        primary_reference_method,
    )
    requested = list(comparisons or ())
    if primary_key not in requested:
        requested.insert(0, primary_key)
    if len(set(requested)) != len(requested):
        raise E1StatisticsError("comparisons must not contain duplicates")
    paired_comparisons = [
        _comparison_from_validated(
            validated,
            seed_metrics,
            condition=condition,
            method=method,
            reference_method=reference,
        )
        for condition, method, reference in requested
    ]
    primary = paired_comparisons[requested.index(primary_key)].copy()
    primary["paired_bootstrap"] = _paired_cluster_bootstrap_from_validated(
        validated,
        condition=primary_condition,
        method=primary_method,
        reference_method=primary_reference_method,
        cluster_field="character",
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    if not np.isclose(
        primary["mean_accuracy_difference_pp"],
        primary["paired_bootstrap"]["mean_difference_pp"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise E1StatisticsError(
            "primary aggregate difference disagrees with paired cluster mean"
        )

    return {
        "schema_version": 1,
        "validation": {
            "row_count": len(validated),
            "condition_count": len(conditions),
            "method_condition_count": len(grouped_metrics),
            "font_count_per_method_condition": expected_font_count,
            "seed_count_per_method_condition": expected_seed_count,
            "class_count": class_count,
            "cells_per_character": expected_font_count * expected_seed_count,
        },
        "bootstrap_parameters": {
            "unit": "character",
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "ci": "percentile_95",
        },
        "conditions": conditions,
        "seed_metrics": seed_metrics,
        "paired_comparisons": paired_comparisons,
        "primary_comparison": primary,
    }


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path | str, document: Any) -> Path:
    """Serialize strict UTF-8 JSON and atomically replace ``path``."""

    destination = Path(path)
    payload = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _atomic_text(destination, payload)
    return destination


def write_csv_atomic(
    path: Path | str,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> Path:
    """Serialize a deterministic RFC-style CSV and atomically replace ``path``."""

    if not fieldnames or len(set(fieldnames)) != len(fieldnames):
        raise E1StatisticsError("fieldnames must be a non-empty unique sequence")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    destination = Path(path)
    _atomic_text(destination, buffer.getvalue())
    return destination


def write_statistics_outputs(
    output_directory: Path | str, report: Mapping[str, Any]
) -> dict[str, Path]:
    """Write JSON summary plus flat seed/comparison CSV tables atomically."""

    if not isinstance(report, Mapping):
        raise E1StatisticsError("report must be a mapping")
    seed_metrics = report.get("seed_metrics")
    comparisons = report.get("paired_comparisons")
    if not isinstance(seed_metrics, list) or not isinstance(comparisons, list):
        raise E1StatisticsError("report is missing seed_metrics or paired_comparisons")
    comparison_rows = [
        {
            "condition": row["condition"],
            "method": row["method"],
            "reference_method": row["reference_method"],
            "mean_accuracy_difference_pp": row["mean_accuracy_difference_pp"],
            "mean_macro_f1_difference": row["mean_macro_f1_difference"],
            "per_seed_json": json.dumps(
                row["per_seed"], ensure_ascii=False, sort_keys=True, allow_nan=False
            ),
        }
        for row in comparisons
    ]
    root = Path(output_directory)
    paths = {
        "json": write_json_atomic(root / "statistics_summary.json", report),
        "seed_csv": write_csv_atomic(
            root / "seed_metrics.csv",
            seed_metrics,
            fieldnames=(
                "condition",
                "method",
                "mapping_seed",
                "sample_count",
                "accuracy",
                "macro_f1",
            ),
        ),
        "comparison_csv": write_csv_atomic(
            root / "paired_comparisons.csv",
            comparison_rows,
            fieldnames=(
                "condition",
                "method",
                "reference_method",
                "mean_accuracy_difference_pp",
                "mean_macro_f1_difference",
                "per_seed_json",
            ),
        ),
    }
    return paths
