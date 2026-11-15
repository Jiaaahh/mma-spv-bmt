"""Generate the read-only E1 post-hoc resampling sensitivity supplement.

This script is intentionally separated from the formal E1 analysis pipeline. It
reads the already frozen prediction table, verifies its SHA-256 digest before
parsing, and writes derived publication artifacts only under
``publication_materials/supplements``. It never writes to ``outputs``, ``locks``,
``configs``, or the frozen data assets.

The primary character-cluster result reproduces the preregistered analysis. The
additional resampling schemes and non-primary contrasts in this package are
POST HOC and were NOT PREREGISTERED.
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PREDICTIONS_SHA256 = (
    "87ce2ec00a86a373237abb305a7a1ab7931b9fda58bda5cdaff69aa1d5dbbde8"
)
CONDITION = "moderate"
ITERATIONS = 100_000
SCHEME_SEEDS = {
    "character": 20_260_826,
    "font": 20_260_827,
    "mapping_seed": 20_260_828,
    "crossed_character_font_mapping_seed": 20_260_829,
}

METHOD_IDEAL = "proto_skeleton_ideal_only"
METHOD_SINGLE = "proto_skeleton_ideal_plus_moderate1"
METHOD_MULTI2 = "proto_skeleton_ideal_plus_moderate2"
METHOD_MULTI4 = "proto_skeleton_ideal_plus_multi4"
METHOD_MULTI4_NO_IDEAL = "proto_skeleton_multi4_no_ideal"

COMPARISONS: tuple[dict[str, str], ...] = (
    {
        "comparison_id": "primary_multi4_vs_ideal",
        "label": "Ideal + four moderate mappings vs ideal only",
        "method": METHOD_MULTI4,
        "reference_method": METHOD_IDEAL,
        "role": "primary",
    },
    {
        "comparison_id": "single1_vs_ideal",
        "label": "Ideal + one moderate mapping vs ideal only",
        "method": METHOD_SINGLE,
        "reference_method": METHOD_IDEAL,
        "role": "exploratory_ablation",
    },
    {
        "comparison_id": "multi2_vs_single1",
        "label": "Ideal + two moderate mappings vs ideal + one",
        "method": METHOD_MULTI2,
        "reference_method": METHOD_SINGLE,
        "role": "exploratory_ablation",
    },
    {
        "comparison_id": "multi4_vs_multi2",
        "label": "Ideal + four moderate mappings vs ideal + two",
        "method": METHOD_MULTI4,
        "reference_method": METHOD_MULTI2,
        "role": "exploratory_ablation",
    },
    {
        "comparison_id": "multi4_vs_single1",
        "label": "Ideal + four moderate mappings vs ideal + one",
        "method": METHOD_MULTI4,
        "reference_method": METHOD_SINGLE,
        "role": "exploratory_ablation",
    },
    {
        "comparison_id": "ideal_multi4_vs_multi4_noideal",
        "label": "Ideal + four moderate mappings vs four mappings without ideal",
        "method": METHOD_MULTI4,
        "reference_method": METHOD_MULTI4_NO_IDEAL,
        "role": "exploratory_ablation",
    },
)

SCHEME_DESCRIPTIONS = {
    "character": (
        "Resample 180 character clusters; retain all 4 fonts x 5 mapping seeds "
        "within each sampled character."
    ),
    "font": (
        "Resample 4 font clusters; retain all 180 characters x 5 mapping seeds "
        "within each sampled font."
    ),
    "mapping_seed": (
        "Resample 5 mapping-seed clusters; retain all 180 characters x 4 fonts "
        "within each sampled mapping seed."
    ),
    "crossed_character_font_mapping_seed": (
        "Independently resample characters, fonts, and mapping seeds, then average "
        "the Cartesian product of the sampled levels (crossed multiway bootstrap)."
    ),
}

REQUIRED_FIELDS = frozenset(
    {
        "method",
        "condition",
        "mapping_seed",
        "font_id",
        "true_label",
        "character",
        "correct",
    }
)


class SensitivityError(RuntimeError):
    """Raised when the frozen source or paired E1 design is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_frozen_rows(path: Path) -> list[dict[str, str]]:
    observed_sha256 = _sha256(path)
    if observed_sha256 != EXPECTED_PREDICTIONS_SHA256:
        raise SensitivityError(
            "frozen predictions SHA-256 mismatch: "
            f"expected {EXPECTED_PREDICTIONS_SHA256}, observed {observed_sha256}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS.difference(reader.fieldnames or ())
        if missing:
            raise SensitivityError(
                f"predictions.csv is missing required fields: {sorted(missing)}"
            )
        return list(reader)


def _paired_difference_tensor(
    rows: Sequence[dict[str, str]],
    *,
    method: str,
    reference_method: str,
) -> tuple[np.ndarray, list[int], list[str], list[int]]:
    selected = [
        row
        for row in rows
        if row["condition"] == CONDITION
        and row["method"] in {method, reference_method}
    ]
    by_method: dict[str, dict[tuple[int, str, int], int]] = {
        method: {},
        reference_method: {},
    }
    for row in selected:
        try:
            mapping_seed = int(row["mapping_seed"])
            true_label = int(row["true_label"])
            correct = int(row["correct"])
        except ValueError as error:
            raise SensitivityError("prediction integer fields are malformed") from error
        if correct not in (0, 1):
            raise SensitivityError("correct must be 0 or 1")
        key = (true_label, row["font_id"], mapping_seed)
        method_rows = by_method[row["method"]]
        if key in method_rows:
            raise SensitivityError(
                f"duplicate paired cell for {row['method']}: {key}"
            )
        method_rows[key] = correct

    method_cells = by_method[method]
    reference_cells = by_method[reference_method]
    if not method_cells or not reference_cells:
        raise SensitivityError(
            f"missing comparison data for {method} vs {reference_method}"
        )
    if method_cells.keys() != reference_cells.keys():
        raise SensitivityError(
            f"paired cells differ for {method} vs {reference_method}"
        )

    labels = sorted({key[0] for key in method_cells})
    fonts = sorted({key[1] for key in method_cells})
    mapping_seeds = sorted({key[2] for key in method_cells})
    if labels != list(range(180)):
        raise SensitivityError("paired comparison must contain labels 0..179")
    if len(fonts) != 4:
        raise SensitivityError("paired comparison must contain exactly 4 fonts")
    if len(mapping_seeds) != 5:
        raise SensitivityError(
            "paired comparison must contain exactly 5 mapping seeds"
        )
    expected_cells = {
        (label, font, mapping_seed)
        for label in labels
        for font in fonts
        for mapping_seed in mapping_seeds
    }
    if method_cells.keys() != expected_cells:
        raise SensitivityError(
            f"comparison is not a complete 180 x 4 x 5 grid: "
            f"{method} vs {reference_method}"
        )

    tensor = np.empty((len(labels), len(fonts), len(mapping_seeds)), dtype=np.int8)
    for label_index, label in enumerate(labels):
        for font_index, font in enumerate(fonts):
            for seed_index, mapping_seed in enumerate(mapping_seeds):
                key = (label, font, mapping_seed)
                tensor[label_index, font_index, seed_index] = (
                    method_cells[key] - reference_cells[key]
                )
    return tensor, labels, fonts, mapping_seeds


def _cluster_bootstrap(
    cluster_values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap one factor for every comparison using shared resamples."""

    comparison_count, cluster_count = cluster_values.shape
    if cluster_count < 2:
        raise SensitivityError("bootstrap requires at least two clusters")
    estimates = np.empty((comparison_count, iterations), dtype=np.float64)
    rng = np.random.default_rng(seed)
    chunk_size = 10_000
    for start in range(0, iterations, chunk_size):
        stop = min(iterations, start + chunk_size)
        indices = rng.integers(
            0,
            cluster_count,
            size=(stop - start, cluster_count),
            dtype=np.int64,
        )
        for comparison_index in range(comparison_count):
            estimates[comparison_index, start:stop] = cluster_values[
                comparison_index, indices
            ].mean(axis=1)
    lower = np.percentile(estimates, 2.5, axis=1)
    upper = np.percentile(estimates, 97.5, axis=1)
    return lower, upper


def _crossed_bootstrap(
    tensors: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Independently resample all three crossed factors using shared draws."""

    comparison_count, character_count, font_count, mapping_seed_count = tensors.shape
    estimates = np.empty((comparison_count, iterations), dtype=np.float64)
    rng = np.random.default_rng(seed)
    chunk_size = 1_000
    for start in range(0, iterations, chunk_size):
        stop = min(iterations, start + chunk_size)
        current = stop - start
        character_indices = rng.integers(
            0,
            character_count,
            size=(current, character_count),
            dtype=np.int64,
        )
        font_indices = rng.integers(
            0,
            font_count,
            size=(current, font_count),
            dtype=np.int64,
        )
        mapping_seed_indices = rng.integers(
            0,
            mapping_seed_count,
            size=(current, mapping_seed_count),
            dtype=np.int64,
        )
        for comparison_index in range(comparison_count):
            sampled = tensors[comparison_index][
                character_indices[:, :, None, None],
                font_indices[:, None, :, None],
                mapping_seed_indices[:, None, None, :],
            ]
            estimates[comparison_index, start:stop] = sampled.mean(axis=(1, 2, 3))
    lower = np.percentile(estimates, 2.5, axis=1)
    upper = np.percentile(estimates, 97.5, axis=1)
    return lower, upper


def _build_report(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    tensors: list[np.ndarray] = []
    common_labels: list[int] | None = None
    common_fonts: list[str] | None = None
    common_mapping_seeds: list[int] | None = None
    for comparison in COMPARISONS:
        tensor, labels, fonts, mapping_seeds = _paired_difference_tensor(
            rows,
            method=comparison["method"],
            reference_method=comparison["reference_method"],
        )
        if common_labels is None:
            common_labels = labels
            common_fonts = fonts
            common_mapping_seeds = mapping_seeds
        elif (
            labels != common_labels
            or fonts != common_fonts
            or mapping_seeds != common_mapping_seeds
        ):
            raise SensitivityError("comparisons do not share the frozen E1 design")
        tensors.append(tensor)

    tensor_array = np.stack(tensors, axis=0)
    character_by_label: dict[int, str] = {}
    for row in rows:
        if row["condition"] != CONDITION:
            continue
        label = int(row["true_label"])
        character = row["character"]
        previous = character_by_label.setdefault(label, character)
        if previous != character:
            raise SensitivityError(
                f"character text is inconsistent for true_label {label}"
            )
    if set(character_by_label) != set(range(180)):
        raise SensitivityError("character mapping must contain labels 0..179")
    # The formal E1 implementation sorts character cluster keys as strings.
    # Matching that deterministic order exactly reproduces its finite Monte Carlo CI.
    character_order = np.asarray(
        sorted(range(180), key=lambda label: character_by_label[label]),
        dtype=np.int64,
    )
    character_ordered_tensors = tensor_array[:, character_order, :, :]
    scheme_values = {
        "character": character_ordered_tensors.mean(axis=(2, 3)),
        "font": tensor_array.mean(axis=(1, 3)),
        "mapping_seed": tensor_array.mean(axis=(1, 2)),
    }
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scheme, values in scheme_values.items():
        intervals[scheme] = _cluster_bootstrap(
            values,
            iterations=ITERATIONS,
            seed=SCHEME_SEEDS[scheme],
        )
    intervals["crossed_character_font_mapping_seed"] = _crossed_bootstrap(
        character_ordered_tensors,
        iterations=ITERATIONS,
        seed=SCHEME_SEEDS["crossed_character_font_mapping_seed"],
    )

    results: list[dict[str, Any]] = []
    for comparison_index, comparison in enumerate(COMPARISONS):
        point_estimate_pp = float(tensor_array[comparison_index].mean() * 100.0)
        scheme_results: list[dict[str, Any]] = []
        for scheme in (
            "character",
            "font",
            "mapping_seed",
            "crossed_character_font_mapping_seed",
        ):
            lower, upper = intervals[scheme]
            if scheme == "character":
                level_counts = {"characters": 180, "fonts": 4, "mapping_seeds": 5}
                resampled_level_count = 180
            elif scheme == "font":
                level_counts = {"characters": 180, "fonts": 4, "mapping_seeds": 5}
                resampled_level_count = 4
            elif scheme == "mapping_seed":
                level_counts = {"characters": 180, "fonts": 4, "mapping_seeds": 5}
                resampled_level_count = 5
            else:
                level_counts = {"characters": 180, "fonts": 4, "mapping_seeds": 5}
                resampled_level_count = None
            scheme_results.append(
                {
                    "scheme": scheme,
                    "description": SCHEME_DESCRIPTIONS[scheme],
                    "iterations": ITERATIONS,
                    "seed": SCHEME_SEEDS[scheme],
                    "level_counts": level_counts,
                    "resampled_level_count": resampled_level_count,
                    "mean_accuracy_difference_pp": point_estimate_pp,
                    "ci95_pp": [
                        float(lower[comparison_index] * 100.0),
                        float(upper[comparison_index] * 100.0),
                    ],
                    "analysis_status": (
                        "RECOMPUTATION_OF_PREREGISTERED_PRIMARY"
                        if comparison["role"] == "primary" and scheme == "character"
                        else "POST_HOC_NOT_PREREGISTERED"
                    ),
                }
            )
        results.append({**comparison, "resampling_results": scheme_results})

    return {
        "status": "POST_HOC_SENSITIVITY_ANALYSIS_COMPLETE",
        "analysis_status": "POST_HOC_NOT_PREREGISTERED",
        "warning": (
            "This sensitivity package was generated after the formal E1 analysis. "
            "Except for recomputing the preregistered primary character-cluster "
            "bootstrap, these intervals and ablation contrasts are post hoc and "
            "must not be presented as preregistered confirmatory tests."
        ),
        "source": {
            "predictions_path": "outputs/e1_confirmation/predictions.csv",
            "expected_sha256": EXPECTED_PREDICTIONS_SHA256,
            "observed_sha256": EXPECTED_PREDICTIONS_SHA256,
            "formal_artifacts_modified": False,
        },
        "design": {
            "condition": CONDITION,
            "true_labels": common_labels,
            "characters_in_formal_cluster_order": [
                character_by_label[int(label)] for label in character_order
            ],
            "fonts": common_fonts,
            "mapping_seeds": common_mapping_seeds,
            "paired_cells_per_comparison": int(np.prod(tensor_array.shape[1:])),
        },
        "bootstrap": {
            "interval": "two-sided percentile 95%",
            "iterations_per_scheme": ITERATIONS,
            "scheme_seeds": SCHEME_SEEDS,
            "shared_resamples_across_comparisons_within_scheme": True,
        },
        "comparisons": results,
        "interpretation_limits": [
            (
                "The character-only interval conditions on the four fonts and five "
                "mapping seeds observed in E1."
            ),
            (
                "The crossed interval treats characters, fonts, and mapping seeds as "
                "sampled factors, but only four fonts and five seeds are available."
            ),
            (
                "The +1/+2/+4 representations form one frozen nested mapping sequence; "
                "mapping count is therefore confounded with mapping identity and order."
            ),
            (
                "The ideal-anchor comparison also changes the number of training "
                "representations (five vs four), so it does not isolate anchor identity."
            ),
        ],
    }


def _csv_text(report: dict[str, Any]) -> str:
    fieldnames = [
        "comparison_id",
        "label",
        "role",
        "condition",
        "method",
        "reference_method",
        "scheme",
        "mean_accuracy_difference_pp",
        "ci95_lower_pp",
        "ci95_upper_pp",
        "iterations",
        "seed",
        "analysis_status",
        "source_predictions_sha256",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for comparison in report["comparisons"]:
        for result in comparison["resampling_results"]:
            writer.writerow(
                {
                    "comparison_id": comparison["comparison_id"],
                    "label": comparison["label"],
                    "role": comparison["role"],
                    "condition": CONDITION,
                    "method": comparison["method"],
                    "reference_method": comparison["reference_method"],
                    "scheme": result["scheme"],
                    "mean_accuracy_difference_pp": (
                        f"{result['mean_accuracy_difference_pp']:.9f}"
                    ),
                    "ci95_lower_pp": f"{result['ci95_pp'][0]:.9f}",
                    "ci95_upper_pp": f"{result['ci95_pp'][1]:.9f}",
                    "iterations": result["iterations"],
                    "seed": result["seed"],
                    "analysis_status": result["analysis_status"],
                    "source_predictions_sha256": EXPECTED_PREDICTIONS_SHA256,
                }
            )
    return buffer.getvalue()


def _markdown_text(report: dict[str, Any]) -> str:
    lines = [
        "# E1 post-hoc resampling sensitivity analysis",
        "",
        "> **POST HOC — NOT PREREGISTERED.** This supplement was generated after "
        "the formal E1 analysis from the frozen prediction table. The primary "
        "character-cluster row is a deterministic recomputation of the preregistered "
        "primary analysis; every additional scheme and ablation interval is post hoc.",
        "",
        "The script verified the frozen prediction SHA-256 before parsing:",
        "",
        f"`{EXPECTED_PREDICTIONS_SHA256}`",
        "",
        "No formal output, lock, configuration, source asset, or prediction was changed.",
        "",
        "## Results",
        "",
        "All entries are paired accuracy differences in percentage points with "
        "100,000-resample percentile 95% confidence intervals.",
        "",
        "| Comparison | Resampling scheme | Difference (pp) | 95% CI (pp) | Status |",
        "|---|---:|---:|---:|---|",
    ]
    scheme_labels = {
        "character": "Character clusters",
        "font": "Font clusters",
        "mapping_seed": "Mapping-seed clusters",
        "crossed_character_font_mapping_seed": "Crossed character × font × seed",
    }
    for comparison in report["comparisons"]:
        for result in comparison["resampling_results"]:
            lower, upper = result["ci95_pp"]
            lines.append(
                f"| {comparison['label']} | {scheme_labels[result['scheme']]} | "
                f"{result['mean_accuracy_difference_pp']:+.3f} | "
                f"[{lower:+.3f}, {upper:+.3f}] | "
                f"{result['analysis_status']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- The preregistered primary comparison remains `ideal + four moderate "
            "mappings` versus `ideal only` under the moderate condition.",
            "- The character-cluster interval treats the four E1 fonts and five E1 "
            "mapping seeds as fixed repeated cells. The crossed sensitivity interval "
            "independently resamples all three sampled factors.",
            "- Only four fonts and five mapping seeds are available, so population-level "
            "generalization across these factors remains imprecise.",
            "- The +1, +2, and +4 representations are one frozen nested sequence. The "
            "dose trend cannot establish that arbitrary sets of four mappings outperform "
            "arbitrary singleton mappings because count, identity, and order are confounded.",
            "- `ideal + four mappings` contains five training representations whereas "
            "`four mappings without ideal` contains four. Their contrast therefore does "
            "not isolate an ideal-anchor identity effect at equal training-set size.",
            "",
            "## Reproduction",
            "",
            "From the E1 project root, run:",
            "",
            "```text",
            "python publication_materials/scripts/generate_e1_posthoc_sensitivity.py",
            "```",
            "",
            "The script refuses to run if the prediction SHA-256 differs from the "
            "expected frozen digest.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.exists() and path.read_bytes() == encoded:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    predictions_path = project_root / "outputs" / "e1_confirmation" / "predictions.csv"
    supplement_dir = project_root / "publication_materials" / "supplements"
    rows = _read_frozen_rows(predictions_path)
    report = _build_report(rows)

    json_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _write_if_changed(supplement_dir / "e1_posthoc_sensitivity.json", json_text)
    _write_if_changed(supplement_dir / "e1_posthoc_sensitivity.csv", _csv_text(report))
    _write_if_changed(supplement_dir / "e1_posthoc_sensitivity.md", _markdown_text(report))

    primary = report["comparisons"][0]
    crossed = next(
        result
        for result in primary["resampling_results"]
        if result["scheme"] == "crossed_character_font_mapping_seed"
    )
    lower, upper = crossed["ci95_pp"]
    print("POST_HOC_SENSITIVITY_ANALYSIS_COMPLETE")
    print(f"source_predictions_sha256={EXPECTED_PREDICTIONS_SHA256}")
    print(
        "primary_crossed_difference_pp="
        f"{crossed['mean_accuracy_difference_pp']:.6f}"
    )
    print(f"primary_crossed_ci95_pp=[{lower:.6f}, {upper:.6f}]")


if __name__ == "__main__":
    main()
