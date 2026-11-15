"""Recount Point-window predictions, replay intervals and summarize saved costs.

Only historical arrays are read. No Point checkpoint is loaded and no timing
measurement is performed on this computer.
"""
from __future__ import annotations
import argparse
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
from recompute_primary_statistics import compare
ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def point():
    root = ROOT / "evidence/point"
    protocol = read_json(root / "source/protocol.json")
    expected = read_json(root / "results/summary.json")
    vectors, details = {}, {}
    reference = {}
    for key in expected["methods"]:
        if key.startswith("point_"):
            continue
        with np.load(root / f"results/{key}.npz", allow_pickle=False) as archive:
            labels, predictions, fonts = archive["labels"], archive["predictions"], archive["fonts"]
            if predictions.shape != (5, 720) or labels.shape != (720,):
                raise ValueError("Point fixed panel shape changed")
            correct = predictions == labels[None]
            vectors[key] = np.array([correct[:, labels == c].mean() for c in range(180)])
            details[key] = {"correct": int(correct.sum()), "n": 3600,
                            "accuracy_percent": 100 * float(correct.mean()),
                            "per_font_percent": {str(f): 100 * float(correct[:, fonts == f].mean()) for f in np.unique(fonts)},
                            "per_mapping_percent": (100 * correct.mean(1)).tolist()}
            reference[key] = predictions.reshape(-1)
    for condition in ["ideal", "multi"]:
        vv = np.stack([vectors[f"{condition}_seed{s}"] for s in protocol["initialization_seeds"]])
        vectors["point_" + condition] = vv.mean(0)
        details["point_" + condition] = {"mean_accuracy_percent": 100 * float(vv.mean()),
                                         "seed_accuracy_percent": (100 * vv.mean(1)).tolist(),
                                         "seed_sample_sd_pp": float((100 * vv.mean(1)).std(ddof=1)), "ensemble": False}
    diffs = {"primary_skeleton_mma_minus_point_multi": vectors["skeleton_mma"] - vectors["point_multi"],
             "point_multi_minus_ideal": vectors["point_multi"] - vectors["point_ideal"],
             "skeleton_mma_gain": vectors["skeleton_mma"] - vectors["skeleton_ideal"],
             "canny_mma_gain": vectors["canny_mma"] - vectors["canny_ideal"]}
    rng = np.random.default_rng(protocol["statistics"]["seed"])
    boot = {key: [] for key in diffs}
    for _ in range(100):
        weights = rng.multinomial(180, np.full(180, 1 / 180), size=1000) / 180
        for key, vector in diffs.items():
            boot[key].extend((weights @ vector * 100).tolist())
    result = {"status": "complete_fixed_panel", "methods": details,
              "contrasts": {key: {"difference_pp": float(v.mean() * 100),
                                   "ci95_pp": np.quantile(boot[key], [.025, .975]).tolist()} for key, v in diffs.items()},
              "statistics": protocol["statistics"], "scope": protocol["scope"]}
    return result, compare(result, expected), reference


def costs(reference):
    root = ROOT / "evidence/cost_server"
    raw = root / "raw/step4/reports"
    expected = read_json(root / "cost_summary.json")
    actual = {"latency": {}, "memory": {}, "components": {}, "cpu_point_over_mma_ratios": {}}
    stream_checks, failures = [], []
    for section, mode in [("latency", "latency"), ("memory", "memory"), ("components", "components")]:
        for config in expected[section]:
            rounds = range(3) if mode == "latency" else range(1)
            latency = []
            for round_number in rounds:
                stem = f"{mode}_r{round_number}_{config}"
                meta = read_json(raw / (stem + ".json"))
                with np.load(raw / (stem + ".npz"), allow_pickle=False) as arr:
                    order, predictions = arr["order"], arr["predictions"]
                    if not np.array_equal(np.sort(order), np.arange(3600)):
                        raise ValueError("Cost order is not a full permutation")
                    model = config.removesuffix("_cpu").removesuffix("_cuda")
                    if model.startswith("point_"):
                        model = "multi_seed" + model.removeprefix("point_")
                    mismatches = int(np.count_nonzero(predictions != reference[model][order]))
                    stream_checks.append({"file": stem + ".npz", "rows": 3600, "prediction_mismatches": mismatches})
                    if mismatches:
                        failures.append({"path": stem, "problem": "raw predictions differ from fixed-window predictions", "count": mismatches})
                    if mode == "latency":
                        values = arr["latency_ns"].astype(np.float64) / 1e6
                        entry = {"round": round_number, "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}
                        failures.extend(compare(entry, {k: meta[k] for k in entry}, stem))
                        latency.append(entry)
                    elif mode == "components":
                        ns = arr["component_ns"]
                        actual[section][config] = {"component_medians_ms": np.median(ns, axis=0).tolist(),
                                                   "without_render_median_ms": float(np.median(ns[:, 0] + ns[:, 2]))}
                        actual[section][config]["component_medians_ms"] = [v / 1e6 for v in actual[section][config]["component_medians_ms"]]
                        actual[section][config]["without_render_median_ms"] /= 1e6
                    else:
                        actual[section][config] = {key: meta[key] for key in expected[section][config]}
            if mode == "latency":
                medians = [r["median_ms"] for r in latency]
                actual[section][config] = {"rounds": latency, "median_of_round_medians_ms": float(np.median(medians)),
                                           "round_median_range_ms": [min(medians), max(medians)]}
    for seed in expected["cpu_point_over_mma_ratios"]:
        ratios = [p["median_ms"] / h["median_ms"]
                  for p, h in zip(actual["latency"][f"point_{seed}_cpu"]["rounds"], actual["latency"]["skeleton_mma_cpu"]["rounds"])]
        actual["cpu_point_over_mma_ratios"][seed] = {"rounds": ratios, "median": float(np.median(ratios)), "range": [min(ratios), max(ratios)]}
    failures.extend(compare(actual, expected, "/cost_summary"))
    return actual, failures, stream_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if ROOT / "checks" not in output.parents:
        raise ValueError("Output must be within checks/")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    p, pf, reference = point()
    c, cf, streams = costs(reference)
    report = {"passed": not (pf + cf), "failures": pf + cf, "point_recomputed": p, "cost_recomputed": c,
              "raw_cost_prediction_checks": streams, "raw_cost_arrays_checked": len(streams),
              "scope": "Saved Point prediction intervals and historical latency/component arrays recomputed; memory copied and cross-checked against per-run raw JSON. No original neural model inference, new training or new hardware measurement.",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "numpy_version": np.__version__,
              "elapsed_seconds": time.perf_counter() - start}
    (output / "point_and_costs.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "failures", "raw_cost_arrays_checked", "elapsed_seconds")}), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
