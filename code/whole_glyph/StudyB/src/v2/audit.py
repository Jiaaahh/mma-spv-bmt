"""Integrity audit and descriptive summary for completed v2 conditions."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.v2.experiment import (
    ConditionSpec,
    ExperimentConfig,
    ExperimentError,
    validate_completed_result,
)


def audit_plan(
    config: ExperimentConfig,
    *,
    require_current_source: bool = True,
) -> dict[str, Any]:
    """Audit expected conditions without changing any experiment output."""

    completed: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    for condition in config.conditions:
        run_id = condition.run_id(config.experiment_id)
        run_dir = config.output_root / run_id
        if not run_dir.exists():
            missing.append(run_id)
            continue
        try:
            completed.append(
                validate_completed_result(
                    config,
                    condition,
                    require_current_source=require_current_source,
                )
            )
        except ExperimentError as error:
            invalid.append({"run_id": run_id, "error": str(error)})
    source_hashes = {
        str(result.get("recorded_source_tree_sha256", "")) for result in completed
    }
    environment_hashes = {
        str(result.get("recorded_environment_sha256", "")) for result in completed
    }
    if len(source_hashes) > 1:
        invalid.append(
            {
                "run_id": "__plan__",
                "error": "completed conditions mix multiple source-tree hashes",
            }
        )
    if len(environment_hashes) > 1:
        invalid.append(
            {
                "run_id": "__plan__",
                "error": "completed conditions mix multiple runtime environments",
            }
        )
    expected = len(config.conditions)
    passed = len(completed) == expected and not missing and not invalid
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "PASS" if passed else "INCOMPLETE_OR_INVALID",
        "expected": expected,
        "completed": len(completed),
        "missing": missing,
        "invalid": invalid,
        "uniform_source_tree_sha256": next(iter(source_hashes), None),
        "uniform_environment_sha256": next(iter(environment_hashes), None),
        "results": completed,
    }


def _condition_lookup(config: ExperimentConfig) -> dict[str, ConditionSpec]:
    return {
        condition.run_id(config.experiment_id): condition
        for condition in config.conditions
    }


def summarize_plan(
    config: ExperimentConfig,
    *,
    allow_source_drift: bool = False,
) -> dict[str, Any]:
    """Create descriptive per-condition and paired method summaries."""

    audit = audit_plan(config, require_current_source=not allow_source_drift)
    if audit["status"] != "PASS":
        raise ExperimentError("all planned conditions must pass integrity audit first")
    lookup = _condition_lookup(config)
    rows: list[dict[str, Any]] = []
    for result in audit["results"]:
        condition = lookup[str(result["run_id"])]
        rows.append(
            {
                "run_id": result["run_id"],
                "method": condition.method,
                "budget": condition.budget,
                "mapping_type": condition.mapping_type,
                "mapping_seed": condition.mapping_seed,
                "accuracy": float(result["accuracy"]),
                "macro_f1": float(result["macro_f1"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
            }
        )
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["budget"]), str(row["method"]))].append(row)
    method_summary = []
    for (budget, method), items in sorted(grouped.items()):
        accuracy = np.asarray([item["accuracy"] for item in items], dtype=np.float64)
        macro_f1 = np.asarray([item["macro_f1"] for item in items], dtype=np.float64)
        method_summary.append(
            {
                "budget": budget,
                "method": method,
                "condition_count": len(items),
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_std": float(accuracy.std(ddof=1)) if accuracy.size > 1 else None,
                "macro_f1_mean": float(macro_f1.mean()),
                "macro_f1_std": float(macro_f1.std(ddof=1)) if macro_f1.size > 1 else None,
            }
        )
    by_key = {
        (row["budget"], row["mapping_type"], row["mapping_seed"], row["method"]): row
        for row in rows
    }
    paired = []
    for row in rows:
        if row["method"] == "skeleton_topk":
            continue
        baseline = by_key.get(
            (
                row["budget"],
                row["mapping_type"],
                row["mapping_seed"],
                "skeleton_topk",
            )
        )
        if baseline is None:
            continue
        paired.append(
            {
                "budget": row["budget"],
                "mapping_seed": row["mapping_seed"],
                "method": row["method"],
                "accuracy_difference_pp": 100.0
                * (float(row["accuracy"]) - float(baseline["accuracy"])),
                "macro_f1_difference": float(row["macro_f1"])
                - float(baseline["macro_f1"]),
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "note": "Descriptive development summary; not a confirmatory inference.",
        "conditions": sorted(rows, key=lambda item: str(item["run_id"])),
        "method_summary": method_summary,
        "paired_vs_skeleton_topk": sorted(
            paired,
            key=lambda item: (
                int(item["budget"]),
                str(item["method"]),
                int(item["mapping_seed"]),
            ),
        ),
    }


def write_summary_files(
    summary: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = list(summary["conditions"])
    if not rows:
        raise ExperimentError("summary contains no conditions")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
