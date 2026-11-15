"""Recompute internal development controls and RanDumb saved-prediction reports."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
import numpy as np
from sklearn.metrics import f1_score
from recompute_primary_statistics import compare
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bootstrap(vectors, seed):
    # Batch only for memory; each vector uses the same 100000 character draws.
    rng = np.random.default_rng(seed)
    matrix = np.stack(vectors, axis=1)
    estimates = []
    for _ in range(100):
        weights = rng.multinomial(180, np.full(180, 1 / 180), size=1000) / 180
        estimates.append(weights @ matrix * 100)
    return np.quantile(np.concatenate(estimates), [.025, .975], axis=0).T


def internal():
    rows = read_rows(ROOT / "evidence/internal_hog/results/hog/predictions.csv")
    archived = read_json(ROOT / "evidence/internal_hog/results/hog/summary.json")
    groups = {}
    cells = {}
    for row in rows:
        key = (row["font"], row["mapping_seed"], int(row["class_id"]))
        method = row["method"]
        target = cells.setdefault(method, {})
        if key in target:
            raise ValueError("Duplicate internal comparison key")
        target[key] = int(row["class_id"] == row["predicted"])
    for method, values in cells.items():
        if len(values) != 5400 or values.keys() != cells["ideal_only"].keys():
            raise ValueError("Internal controls not fully paired")
        groups[method] = np.array([np.mean([v for k, v in values.items() if k[2] == c]) for c in range(180)])
    vectors = [groups[r["a"]] - groups[r["b"]] for r in archived["comparisons"]]
    intervals = bootstrap(vectors, 20260909)
    actual = {"accuracies": {k: float(v.mean() * 100) for k, v in groups.items()},
              "comparisons": [{"a": item["a"], "b": item["b"], "difference_pp": float(v.mean() * 100),
                               "character_conditional_95ci_pp": interval.tolist()}
                              for item, v, interval in zip(archived["comparisons"], vectors, intervals)],
              "training_group_mma_accuracy": [float(groups[f"mma_g{i}"].mean() * 100) for i in range(5)]}
    return actual, compare(actual, {k: archived[k] for k in actual}), len(rows)


def randumb():
    rows = read_rows(ROOT / "evidence/randumb/predictions.csv")
    archived = read_json(ROOT / "evidence/randumb/results_summary.json")
    grouped = {}
    for row in rows:
        grouped.setdefault((row["panel"], row["method"], row["projection_seed"]), []).append(row)
    panels, all_vectors = {}, []
    correctness = {}
    for panel, expected in archived["panels"].items():
        methods = {}
        for method in expected["methods"]:
            seeds = sorted({s for p, m, s in grouped if p == panel and m == method}, key=lambda s: -1 if s == "fixed" else int(s))
            correct_by_seed, f1s = [], []
            for seed in seeds:
                cells = grouped[panel, method, seed]
                keys = [(r["font_id"], r["mapping_seed"], int(r["class_id"])) for r in cells]
                if len(set(keys)) != len(keys):
                    raise ValueError("Duplicate random projection prediction key")
                correct = {(r["font_id"], r["mapping_seed"], int(r["class_id"])): int(r["class_id"] == r["predicted_class"]) for r in cells}
                correct_by_seed.append(correct)
                mappings = sorted({r["mapping_seed"] for r in cells})
                f1s.append(float(np.mean([f1_score([int(r["class_id"]) for r in cells if r["mapping_seed"] == s],
                                                 [int(r["predicted_class"]) for r in cells if r["mapping_seed"] == s],
                                                 labels=np.arange(180), average="macro", zero_division=0) for s in mappings])))
            keys = sorted(correct_by_seed[0])
            if any(c.keys() != correct_by_seed[0].keys() for c in correct_by_seed):
                raise ValueError("Projection seeds use different evaluation units")
            values = np.array([[c[k] for k in keys] for c in correct_by_seed], dtype=np.float64)
            acc = values.mean(1)
            methods[method] = {"accuracy_percent": float(acc.mean() * 100), "seed_accuracy_percent": (acc * 100).tolist(),
                               "seed_sd_percent": float(acc.std(ddof=1) * 100) if len(acc) > 1 else None,
                               "macro_f1": float(np.mean(f1s))}
            correctness[panel, method] = {k: values[:, i].mean() for i, k in enumerate(keys)}
        comparisons = []
        for item in expected["comparisons"]:
            a, b = correctness[panel, item["a"]], correctness[panel, item["b"]]
            if a.keys() != b.keys():
                raise ValueError("RanDumb comparison is not paired")
            vector = np.array([np.mean([a[k] - b[k] for k in a if k[2] == c]) for c in range(180)])
            all_vectors.append(vector)
            comparisons.append({"a": item["a"], "b": item["b"], "difference_pp": float(vector.mean() * 100), "primary": item["primary"]})
        panels[panel] = {"methods": methods, "comparisons": comparisons}
    intervals = iter(bootstrap(all_vectors, 20260928))
    for panel in panels.values():
        for item in panel["comparisons"]:
            item["conditional95ci_pp"] = next(intervals).tolist()
    return {"panels": panels}, compare(panels, archived["panels"]), len(rows)


def table_checks(internal_report, random_report):
    table2 = read_rows(ROOT / "tables/reference/Table2.csv")
    t2 = random_report["panels"]
    methods = ["H0", "H1", "R0", "R1"]
    checks = []
    for index, method in enumerate(methods):
        for panel, column in [("C1", "Basic accuracy, %"), ("E1", "Extended accuracy, %")]:
            item = t2[panel]["methods"][method]
            text = f"{item['accuracy_percent']:.2f}"
            if item["seed_sd_percent"] is not None:
                text += f" ± {item['seed_sd_percent']:.2f}"
            checks.append({"table": "Table2", "row": index + 1, "column": column,
                           "actual": text, "archived": table2[index][column], "passed": text == table2[index][column]})
    table3 = read_rows(ROOT / "tables/reference/Table3.csv")
    controls = ["ideal_only", "ordinary_affine4", "ensemble_mean_g0", "ensemble_max_g0", "mma_g0"]
    for index, control in enumerate(controls):
        value = f"{internal_report['accuracies'][control]:.2f}"
        checks.append({"table": "Table3", "row": index + 1, "actual": value,
                       "archived": table3[index]["Accuracy, %"], "passed": value == table3[index]["Accuracy, %"]})
        if control == "mma_g0":
            continue
        item = next(r for r in internal_report["comparisons"] if {r["a"], r["b"]} == {"mma_g0", control})
        delta = item["difference_pp"] if item["a"] == "mma_g0" else -item["difference_pp"]
        interval = item["character_conditional_95ci_pp"]
        if item["a"] != "mma_g0":
            interval = [-interval[1], -interval[0]]
        for column, value in [("MMA minus control, pp", f"{delta:+.2f}"),
                              (list(table3[index])[-1], f"[{interval[0]:.2f}, {interval[1]:.2f}]")]:
            checks.append({"table": "Table3", "row": index + 1, "column": column, "actual": value,
                           "archived": table3[index][column], "passed": value == table3[index][column]})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT / "checks" not in output.parents:
        raise ValueError("Output must be within checks/")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    i, fi, ni = internal()
    r, fr, nr = randumb()
    tables = table_checks(i, r)
    report = {"passed": not (fi + fr) and all(v["passed"] for v in tables), "failures": fi + fr,
              "rows_read": {"internal_hog": ni, "randumb": nr}, "internal_recomputed": i, "randumb_recomputed": r,
              "main_table_checks": tables, "scope": "Frozen-prediction descriptive statistics and character-cluster bootstrap only; original development/exposed-panel roles retained; unadjusted secondary comparisons remain exploratory.",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "elapsed_seconds": time.perf_counter() - start}
    (output / "comparator_statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "statistical_failures": fi + fr,
                      "failed_table_cells": [v for v in tables if not v["passed"]], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
