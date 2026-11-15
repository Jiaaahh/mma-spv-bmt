"""Verify that the multibudget code preserves the frozen C1 B=64 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("audits/B64_REPLAY_REPORT.json"),
    )
    arguments = parser.parse_args()
    reference = arguments.reference_dir.resolve()
    replay = arguments.replay_dir.resolve()
    for directory in (reference, replay):
        if not directory.is_dir():
            raise SystemExit(f"ERROR: result directory does not exist: {directory}")

    reference_summary = _load_json(reference / "summary.json")
    replay_summary = _load_json(replay / "summary.json")
    reference_manifest = _load_json(reference / "manifest.json")
    replay_manifest = _load_json(replay / "manifest.json")
    reference_summary_without_id = dict(reference_summary)
    replay_summary_without_id = dict(replay_summary)
    reference_summary_without_id.pop("experiment_id", None)
    replay_summary_without_id.pop("experiment_id", None)
    comparisons = {
        "predictions_csv_exact": (
            _sha256(reference / "predictions.csv")
            == _sha256(replay / "predictions.csv")
        ),
        "fold_metrics_csv_exact": (
            _sha256(reference / "fold_metrics.csv")
            == _sha256(replay / "fold_metrics.csv")
        ),
        "methods_exact": reference_summary.get("methods")
        == replay_summary.get("methods"),
        "primary_difference_exact": reference_summary.get("primary_difference_pp")
        == replay_summary.get("primary_difference_pp"),
        "primary_per_seed_exact": reference_summary.get(
            "primary_per_seed_difference_pp"
        )
        == replay_summary.get("primary_per_seed_difference_pp"),
        "paired_bootstrap_exact": reference_summary.get("paired_bootstrap")
        == replay_summary.get("paired_bootstrap"),
        "summary_except_experiment_id_exact": reference_summary_without_id
        == replay_summary_without_id,
        "train_selection_hash_exact": reference_manifest.get(
            "train_selections_sha256"
        )
        == replay_manifest.get("train_selections_sha256"),
        "test_selection_hash_exact": reference_manifest.get(
            "test_selections_sha256"
        )
        == replay_manifest.get("test_selections_sha256"),
    }
    expected_anchor = {
        "ideal_only_accuracy": 0.4955555555555556,
        "mma_accuracy": 0.6472222222222221,
        "difference_pp": 15.166666666666655,
        "ci95_pp": [11.833333333333332, 18.555555555555557],
    }
    observed_anchor = {
        "ideal_only_accuracy": replay_summary["methods"]["ideal_only"][
            "mean_accuracy"
        ],
        "mma_accuracy": replay_summary["methods"]["ideal_plus_multi4"][
            "mean_accuracy"
        ],
        "difference_pp": replay_summary["primary_difference_pp"],
        "ci95_pp": replay_summary["paired_bootstrap"]["ci95_pp"],
    }
    comparisons["expected_anchor_exact"] = observed_anchor == expected_anchor
    comparisons["expected_selection_hashes_exact"] = {
        "train": replay_manifest.get("train_selections_sha256"),
        "test": replay_manifest.get("test_selections_sha256"),
    } == {
        "train": "b3ac9c362d2e52f1d9242f8f2bc04c6de86c6fb7f523c93f8b9c0bbcd8ffac0f",
        "test": "08e96e9323b05fd26f8ac455fd358b50e8580c94ce9d7007b2413292c2a582f5",
    }
    passed = all(comparisons.values())
    report = {
        "schema_version": 1,
        "status": "PASS_B64_EXACT_REGRESSION" if passed else "FAIL_B64_REGRESSION",
        "reference_dir": str(reference),
        "replay_dir": str(replay),
        "comparisons": comparisons,
        "expected_anchor": expected_anchor,
        "observed_anchor": observed_anchor,
        "reference_hashes": {
            name: _sha256(reference / name)
            for name in ("predictions.csv", "fold_metrics.csv", "summary.json")
        },
        "replay_hashes": {
            name: _sha256(replay / name)
            for name in ("predictions.csv", "fold_metrics.csv", "summary.json")
        },
    }
    _atomic_json(arguments.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("ERROR: B=64 replay differs from the frozen C1 result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
