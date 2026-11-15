"""Limited paired multi-axis sensitivity audit from frozen prediction CSVs.

This does not train or run inference, and does not replace historical intervals.
E1 reuses the archived historical crossed sampler. C1 and StudyB reuse the
independent audit's multinomial product-bootstrap path, not a historical replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

# Set before importing NumPy; this audit has no need for more than two threads.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "2"
sys.dont_write_bytecode = True
import numpy as np

ITERATIONS = 100_000
SOURCES = {
    "C1": ("whole_glyph/C1/predictions.csv", "01460997e7cc3b898241486d6881137e19225ea9f38f3fc9c027577be312f3d3"),
    "E1": ("whole_glyph/E1/predictions.csv", "87ce2ec00a86a373237abb305a7a1ab7931b9fda58bda5cdaff69aa1d5dbbde8"),
    "StudyB": ("whole_glyph/StudyB/merged_predictions_with_truth.csv", "65a07e70c520d7651b18faf5bdf997974e0326c400140878e492c1ee4dabd0f3"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(root: Path, panel: str) -> tuple[list[dict[str, str]], dict]:
    relative, expected = SOURCES[panel]
    path = root / relative
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen source digest mismatch: {relative}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {"relative_path": relative, "sha256": actual, "rows_read": len(rows)}


def difference_cube(rows: list[dict[str, str]], panel: str, budget: int | None = None) -> tuple[np.ndarray, dict]:
    reference = "ideal_only"
    treatment = "ideal_plus_multi4"
    if panel == "E1":
        reference, treatment = "proto_skeleton_ideal_only", "proto_skeleton_ideal_plus_multi4"
    cells = {reference: {}, treatment: {}}
    for row in rows:
        method = row["method"]
        if method not in cells or (budget is not None and int(row["budget"]) != budget):
            continue
        if panel == "E1" and row["condition"] != "moderate":
            continue
        truth = int(next(row[k] for k in ("class_id", "true_label", "y_true") if k in row))
        predicted = int(next(row[k] for k in ("predicted_class_id", "predicted_label", "y_pred") if k in row))
        if not (0 <= truth < 180 and 0 <= predicted < 180):
            raise ValueError("Class identifiers must be in 0..179")
        correct = int(truth == predicted)
        if "correct" in row and int(row["correct"]) != correct:
            raise ValueError("Correctness flag disagrees with frozen prediction")
        key = (truth, row["font_id"], int(row.get("mapping_seed", row.get("seed", ""))))
        if key in cells[method]:
            raise ValueError(f"Duplicate paired key: {panel}/{budget}/{method}/{key}")
        cells[method][key] = correct
    if not cells[reference] or cells[reference].keys() != cells[treatment].keys():
        raise ValueError("Treatment and reference paired keys differ or are empty")
    labels = sorted({key[0] for key in cells[reference]})
    fonts = sorted({key[1] for key in cells[reference]})
    maps = sorted({key[2] for key in cells[reference]})
    if labels != list(range(180)) or len(fonts) != (2 if panel == "C1" else 4) or len(maps) != 5:
        raise ValueError("Unexpected frozen panel dimensions")
    expected_keys = {(c, f, m) for c in labels for f in fonts for m in maps}
    if cells[reference].keys() != expected_keys:
        raise ValueError("Incomplete Cartesian grid")
    cube = np.asarray([[[cells[treatment][c, f, m] - cells[reference][c, f, m]
                         for m in maps] for f in fonts] for c in labels], dtype=np.float64)
    return cube, {"characters": 180, "fonts": fonts, "mapping_seeds": maps,
                  "paired_cells": int(cube.size), "complete_paired_cartesian_grid": True,
                  "reference": reference, "treatment": treatment}


def independent_product_bootstrap(cube: np.ndarray, seed: int) -> np.ndarray:
    """Identical to the reviewed audit's C1/StudyB path: 50 batches x 2,000.

    Each axis is independently resampled with replacement; the same product
    weights multiply paired treatment-minus-reference cells. Budget integration
    is done first within each cell, so all budgets retain the same resamples.
    """
    rng = np.random.default_rng(seed)
    nc, nf, nm = cube.shape
    distribution = []
    for _ in range(50):
        cw = rng.multinomial(nc, np.full(nc, 1 / nc), size=2000)
        fw = rng.multinomial(nf, np.full(nf, 1 / nf), size=2000)
        mw = rng.multinomial(nm, np.full(nm, 1 / nm), size=2000)
        distribution.append(np.einsum("bc,bf,bm,cfm->b", cw, fw, mw, cube,
                                      optimize=True) / (nc * nf * nm))
    return np.quantile(np.concatenate(distribution), [0.025, 0.975])


def result_row(cube: np.ndarray, ci: np.ndarray, historical: list[float],
               scale: float, unit: str, replay: str, seed: int, details: dict) -> dict:
    scaled = ci * scale
    return {"point": float(cube.mean() * scale), "ci95": scaled.tolist(), "unit": unit,
            "historical_reported_ci95": historical,
            "historical_endpoints_replaced": False, "replay_status": replay,
            "iterations": ITERATIONS, "seed": seed, **details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True,
                        help="Evidence directory containing whole_glyph/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New report directory outside the frozen evidence tree")
    args = parser.parse_args()
    bundle = args.bundle_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output == bundle or bundle in output.parents:
        raise ValueError("Output must be outside the frozen evidence tree")
    start = time.perf_counter()
    archive = Path(__file__).resolve().parent / "sensitivity_sources"
    archive_manifest = json.loads((archive / "source_manifest.json").read_text(encoding="utf8"))
    for entry in archive_manifest["archived_files"]:
        # Step 4: verify redacted candidate bytes while retaining the original digest.
        if sha256(archive / entry["archive_name"]) != entry.get("candidate_sha256", entry["sha256"]):
            raise ValueError(f"Archived-source digest mismatch: {entry['archive_name']}")
    spec = importlib.util.spec_from_file_location("e1_historical", archive / "generate_e1_posthoc_sensitivity.py")
    historical_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical_module)
    report = {"scope": "Limited frozen-prediction sensitivity recalculation only; no training or inference",
              "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                              "thread_environment_limit": 2},
              "bootstrap": "Paired crossed product empirical percentile bootstrap; 100000 replicates per panel",
              "sources": {}, "results": {}, "checks": [], "historical_intervals_unchanged": True}

    rows, report["sources"]["C1"] = read_rows(bundle, "C1")
    cube, design = difference_cube(rows, "C1")
    ci = independent_product_bootstrap(cube, 20260825)
    report["results"]["C1"] = result_row(cube, ci, [9.83, 21.00], 100, "percentage_points",
        "independent_recalculation_not_historical_rng_replay", 20260825,
        {"design": design, "sampler": "multinomial", "batch_size": 2000,
         "ordering": "numeric class id, lexicographic font id, numeric mapping seed",
         "boundary": "Historical C1 multi-axis invocation path was not recovered; do not interpret endpoint differences as a failed exact-replay test."})
    report["checks"].append({"name": "C1_matches_review_independent_path",
                            "passed": bool(np.allclose(ci, [0.09888888888888889, 0.21], atol=1e-12, rtol=0))})
    print("C1 independent sensitivity complete", flush=True)

    rows, report["sources"]["E1"] = read_rows(bundle, "E1")
    cube, design = difference_cube(rows, "E1")
    characters = {}
    for row in rows:
        label = int(row["true_label"])
        if characters.setdefault(label, row["character"]) != row["character"]:
            raise ValueError("Inconsistent character labels")
    order = sorted(range(180), key=lambda label: characters[label])
    cube = cube[order]
    lo, hi = historical_module._crossed_bootstrap(cube[None, ...], iterations=ITERATIONS, seed=20260829)
    ci = np.array([lo[0], hi[0]])
    report["results"]["E1"] = result_row(cube, ci, [9.94, 20.19], 100, "percentage_points",
        "historical_primary_crossed_sampler_replayed", 20260829,
        {"design": design, "sampler": "sequential integers: character, font, mapping in each batch",
         "batch_size": 1000, "ordering": "character string, lexicographic font id, numeric mapping seed",
         "historical_function": "sensitivity_sources/generate_e1_posthoc_sensitivity.py::_crossed_bootstrap",
         "boundary": "Only the primary multi4-versus-ideal crossed sensitivity is rerun, not all historical ablations or one-axis resampling schemes."})
    report["checks"].append({"name": "E1_matches_historical_crossed_endpoints",
                            "passed": bool(np.allclose(ci, [0.09944444444444445, 0.20194444444444445], atol=1e-12, rtol=0))})
    print("E1 historical primary crossed sampler complete", flush=True)

    rows, report["sources"]["StudyB"] = read_rows(bundle, "StudyB")
    budgets = [32, 48, 64, 96, 128]
    weights = np.array([8, 16, 24, 32, 16]) / 96
    cubes = [difference_cube(rows, "StudyB", b) for b in budgets]
    if any(design != cubes[0][1] for _, design in cubes[1:]):
        raise ValueError("Budget grids must retain identical paired units")
    cube = sum(w * item[0] for w, item in zip(weights, cubes))
    ci = independent_product_bootstrap(cube, 20260829)
    report["results"]["StudyB"] = result_row(cube, ci, [0.0274, 0.1693], 1, "delta_nAUBC",
        "independent_recalculation_not_historical_rng_replay", 20260829,
        {"design": cubes[0][1], "sampler": "multinomial", "batch_size": 2000,
         "budgets": budgets, "normalized_trapezoid_weights": weights.tolist(),
         "ordering": "numeric class id, lexicographic font id, numeric mapping seed",
         "boundary": "Agreement at four reported decimals is not a claim of exact historical random-call replay."})
    report["checks"].append({"name": "StudyB_matches_review_independent_path",
                            "passed": bool(np.allclose(ci, [0.027407407407407405, 0.16932870370370368], atol=1e-12, rtol=0))})
    print("StudyB independent sensitivity complete", flush=True)
    report["interpretation_limits"] = [
        "All multi-axis sensitivity intervals are post hoc, not newly preregistered confirmatory tests.",
        "Only 2 fonts for C1 and 4 for E1/StudyB, with 5 mapping seeds per panel, limit population-level generalization.",
        "The primary character-cluster intervals remain separate and are not replaced by this audit.",
        "Independent C1 and StudyB sampler paths are explicitly not exact historical RNG replay.",
    ]
    report["elapsed_seconds"] = time.perf_counter() - start
    report["entry_script_sha256"] = sha256(Path(__file__))
    report["all_checks_passed"] = all(item["passed"] for item in report["checks"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "sensitivity_recomputed.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "elapsed_seconds": report["elapsed_seconds"],
                      "intervals": {k: v["ci95"] for k, v in report["results"].items()}}, ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
