"""Replay the frozen primary character-cluster analyses; do not refit models."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compare(actual, expected, path=""):
    failures = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [{"path": path, "problem": "type mismatch"}]
        for key in expected:
            if key not in actual:
                failures.append({"path": path + "/" + key, "problem": "missing key"})
            else:
                failures.extend(compare(actual[key], expected[key], path + "/" + key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [{"path": path, "problem": "list size mismatch"}]
        for i, (a, e) in enumerate(zip(actual, expected)):
            failures.extend(compare(a, e, path + "/" + str(i)))
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isclose(actual, expected, atol=1e-12, rtol=0):
            failures.append({"path": path, "actual": actual, "archived": expected})
    elif actual != expected:
        failures.append({"path": path, "actual": actual, "archived": expected})
    return failures


def c1():
    from src.v2 import mma_confirmation as m
    rows = read_rows(ROOT / "evidence/whole_glyph/C1/predictions.csv")
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["mapping_seed"]), []).append(row)
    methods = {}
    for method in m.METHODS:
        values = {seed: m._metrics(np.array([int(x["class_id"]) for x in rs], dtype=np.int64),
                                  np.array([int(x["predicted_class_id"]) for x in rs], dtype=np.int64))
                  for (name, seed), rs in groups.items() if name == method}
        methods[method] = {"mean_accuracy": float(np.mean([v["accuracy"] for v in values.values()])),
                           "std_accuracy": float(np.std([v["accuracy"] for v in values.values()])),
                           "mean_macro_f1": float(np.mean([v["macro_f1"] for v in values.values()])),
                           "per_seed_accuracy": {s: v["accuracy"] for s, v in values.items()}}
    cells = {name: {} for name in (m.PRIMARY_BASELINE, m.PRIMARY_METHOD)}
    for row in rows:
        if row["method"] in cells:
            key = (int(row["class_id"]), row["font_id"], row["mapping_seed"])
            if key in cells[row["method"]]:
                raise ValueError("Duplicate C1 cell")
            cells[row["method"]][key] = int(row["class_id"] == row["predicted_class_id"])
    if cells[m.PRIMARY_METHOD].keys() != cells[m.PRIMARY_BASELINE].keys():
        raise ValueError("Unpaired C1 cells")
    differences = np.array([np.mean([cells[m.PRIMARY_METHOD][k] - cells[m.PRIMARY_BASELINE][k]
                                     for k in cells[m.PRIMARY_BASELINE] if k[0] == c]) for c in range(180)])
    actual = {"methods": methods, "paired_bootstrap": m._bootstrap(differences, iterations=100000, seed=20260825, cells_per_character=10),
              "primary_difference_pp": float(differences.mean() * 100)}
    archived = read_json(ROOT / "evidence/whole_glyph/C1/summary.json")
    expected = {key: archived[key] for key in actual}
    return actual, expected, len(rows)


def e1():
    from src.e1 import statistics as m
    rows = read_rows(ROOT / "evidence/whole_glyph/E1/predictions.csv")
    archived = read_json(ROOT / "evidence/whole_glyph/E1/statistics_summary.json")
    comparisons = [(r["condition"], r["method"], r["reference_method"]) for r in archived["paired_comparisons"]]
    recomputed = m.build_statistics_report(rows, comparisons=comparisons)
    keys = ["validation", "bootstrap_parameters", "conditions", "seed_metrics", "paired_comparisons", "primary_comparison"]
    return {key: recomputed[key] for key in keys}, {key: archived[key] for key in keys}, len(rows)


def study_b():
    from src.multibudget import analysis as m
    rows = read_rows(ROOT / "evidence/whole_glyph/StudyB/merged_predictions_with_truth.csv")
    spec = m.AnalysisSpec(reference_method="ideal_only", treatment_method="ideal_plus_multi4",
                         budgets=(32,48,64,96,128), seeds=(4101,4102,4103,4104,4105),
                         font_ids=tuple(sorted({r["font_id"] for r in rows})),
                         char_ids=tuple(sorted({r["char_id"] for r in rows})))
    validated = m.load_released_predictions(read_json(ROOT / "evidence/whole_glyph/StudyB/release_manifest.json"),
                                           lambda: rows, spec)
    result = m.analyze_predictions(validated.rows, spec)
    archived = read_json(ROOT / "evidence/whole_glyph/StudyB/main_results.json")
    bs = result.bootstrap
    actual = {"accuracy_budget": [asdict(r) for r in result.accuracy],
              "delta_budget": [asdict(r) for r in result.delta],
              "naubc": [asdict(r) for r in result.naubc], "delta_naubc": result.delta_naubc,
              "b_eq64": asdict(result.b_eq64), "bootstrap": {
                  "accuracy": [{"method": r["method"], "budget": r["budget"], **asdict(bs.accuracy[r["method"],r["budget"]])}
                               for r in archived["bootstrap"]["accuracy"]],
                  "delta": [{"budget": r["budget"], **asdict(bs.delta[r["budget"]])} for r in archived["bootstrap"]["delta"]],
                  "naubc": [{"method": r["method"], **asdict(bs.naubc[r["method"]])} for r in archived["bootstrap"]["naubc"]],
                  "delta_naubc": asdict(bs.delta_naubc), "replicates": bs.replicates, "seed": bs.seed,
                  "confidence_level": bs.confidence_level, "cluster": bs.cluster}}
    return actual, {key: archived[key] for key in actual}, len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, choices=["C1", "E1", "StudyB"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT / "checks" not in output.parents:
        raise ValueError("Output must be within checks/")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "code/whole_glyph" / args.panel))
    start = time.perf_counter()
    actual, expected, count = {"C1": c1, "E1": e1, "StudyB": study_b}[args.panel]()
    failures = compare(actual, expected)
    report = {"panel": args.panel, "passed": not failures, "failures": failures,
              "rows_read": count, "scope": "Primary conditional character-cluster bootstrap (100000 draws) and archived descriptive statistics replay; not new inference.",
              "statistical_unit": "180 character clusters; font/mapping cells remain paired and fixed in these conditional intervals",
              "normality_assumption": "Not required for the frozen empirical cluster percentile bootstrap; no normality-based test substituted.",
              "numpy_version": np.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "elapsed_seconds": time.perf_counter() - start, "recomputed": actual}
    (output / "primary_statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("panel", "passed", "rows_read", "failures", "elapsed_seconds")}), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
