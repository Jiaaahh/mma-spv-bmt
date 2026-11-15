"""Read-only Stage-10A audit and deterministic final-run plan construction.

No function in this module imports or invokes an encoder, SPV forward model,
HOG extractor, classifier, predictor, or metric implementation.  The formal
glyph archive is opened only for non-pixel metadata arrays; ``images`` is never
indexed or loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.deterministic_io import sha256_file, tree_sha256
from src.locked_test_guard import validate_formal_freeze_integrity


BENCHMARK_ID = "eb_sse_hanzi180_font8_v1"
METHODS = (
    "raw_topk",
    "edge_topk",
    "skeleton_topk",
    "eb_topk",
    "eb_sse",
)
BUDGETS = (32, 64, 128)
FORMAL_MODERATE_SEEDS = (2026, 2027, 2028, 2029, 2030)
IDEAL_AUXILIARY_BUDGET = 64
DEVELOPMENT_FONTS = (
    "noto_sans_sc_thin",
    "noto_serif_sc_extralight",
    "ma_shan_zheng",
    "zcool_qingke_huangyou",
    "zcool_kuaile",
    "liu_jian_mao_cao",
)
LOCKED_TEST_FONTS = ("long_cang", "zhi_mang_xing")
EXPECTED_TRAIN_SAMPLES = 1080
EXPECTED_TEST_SAMPLES = 360

STAGE8_MANIFEST_SHA256 = (
    "452b514b4a2c7e5dd7b10c13a37656571cf56181f0115b0801ef89287ea369ab"
)
STAGE5_5_TRUTH_TABLE_SHA256 = (
    "3559d22eba27bd195bef99a29169307c21f0898b2869b113ff1c09af9dace97e"
)
FORMAL_PARAMETERS_SHA256 = (
    "8ea8e25da16cc9ccdcb96f650b5ed5a9ec4f2d59bacbba2505f48a997398dd2c"
)
FORMAL_EXPERIMENT_CONFIG_SHA256 = (
    "6cf4e77d441b932841d7a981c2b0320346b4cadb714b65c8139e1cbefd58a2c2"
)
ENCODER_BUNDLE_SHA256 = (
    "4183a48cf6c9e9001fceb0ad0bc39223f4bf16e06328d27db24e9cc8c4baa367"
)

PLAN_RELATIVE_PATH = Path("plans/stage10_final_run_plan.json")
PLAN_SHA_RELATIVE_PATH = Path("plans/stage10_final_run_plan.sha256")
PREFLIGHT_AUDIT_RELATIVE_PATH = Path("outputs/stage10a/preflight_audit.json")
FORMAL_OUTPUT_ROOT = Path("outputs/stage10/formal_runs")
STAGE9_VALIDATION_PREDICTIONS = Path("outputs/stage9/validation_predictions")


class Stage10APreflightError(ValueError):
    """Raised before a protocol mismatch can enter the final run plan."""


@dataclass(frozen=True, slots=True)
class MetadataAudit:
    train_sample_count: int
    test_sample_count: int
    train_font_ids: tuple[str, ...]
    test_font_ids: tuple[str, ...]
    arrays_read: tuple[str, ...]
    images_array_read: bool
    train_classes: int
    test_classes: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage10APreflightError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Stage10APreflightError(f"JSON root must be an object: {path}")
    return value


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="ascii", newline="\n")
    temporary.replace(path)


def encoder_bundle_paths(project_root: Path) -> list[Path]:
    root = project_root.resolve()
    return [
        *sorted((root / "src" / "encoders").glob("*.py")),
        root / "src" / "coverage.py",
        root / "src" / "eb_drive.py",
        root / "src" / "eb_representation.py",
        root / "src" / "topology.py",
        root / "src" / "spv_forward.py",
    ]


def _formal_mapping_paths(project_root: Path) -> list[Path]:
    root = project_root.resolve()
    return [
        root / "data" / "mappings" / "formal" / f"moderate_seed_{seed}.npz"
        for seed in FORMAL_MODERATE_SEEDS
    ]


def frozen_readonly_snapshot(project_root: str | Path) -> dict[str, Any]:
    """Hash every frozen input/ledger without opening performance arrays."""

    root = Path(project_root).resolve()
    file_paths = {
        "stage8_manifest": root / "data" / "benchmark" / "benchmark_manifest.json",
        "stage5_5_truth_table": root / "protocols" / "STAGE5_5_CODE_TRUTH_TABLE.yaml",
        "formal_parameters": root / "configs" / "formal_parameters_frozen.json",
        "formal_parameters_sidecar": root / "configs" / "formal_parameters_frozen.sha256",
        "formal_experiment_config": root / "configs" / "formal_experiment_frozen.json",
        "formal_performance_ledger": root / "data" / "benchmark" / "formal_performance_access_ledger.json",
        "stage8_mapping_ledger": root / "data" / "mappings" / "formal" / "mapping_access_ledger.json",
        "formal_glyph_archive": root / "data" / "processed" / "benchmark" / "glyphs.npz",
        "font_split": root / "data" / "registry" / "font_split_formal.json",
    }
    snapshot = {
        name: sha256_file(path)
        for name, path in file_paths.items()
    }
    snapshot["encoder_implementation_bundle"] = tree_sha256(
        root, encoder_bundle_paths(root)
    )
    snapshot["formal_mapping_files"] = {
        path.stem.removeprefix("moderate_seed_"): sha256_file(path)
        for path in _formal_mapping_paths(root)
    }
    return snapshot


def load_formal_sample_metadata(project_root: str | Path) -> MetadataAudit:
    """Read labels/font/split IDs only; the locked-test pixel array stays unopened."""

    root = Path(project_root).resolve()
    glyph_path = root / "data" / "processed" / "benchmark" / "glyphs.npz"
    arrays_read = ("class_ids", "font_ids", "sample_ids", "splits")
    with np.load(glyph_path, allow_pickle=False) as archive:
        required = {*arrays_read, "images"}
        if not required <= set(archive.files):
            raise Stage10APreflightError("formal glyph archive is missing required arrays")
        class_ids = np.asarray(archive["class_ids"])
        font_ids = np.asarray(archive["font_ids"])
        sample_ids = np.asarray(archive["sample_ids"])
        splits = np.asarray(archive["splits"])
        # Deliberately no access to archive["images"].

    if any(array.shape != (1440,) for array in (class_ids, font_ids, sample_ids, splits)):
        raise Stage10APreflightError("formal sample metadata must contain 1440 rows")
    if class_ids.dtype != np.int64:
        raise Stage10APreflightError("formal class IDs must use int64")
    if len(set(map(str, sample_ids))) != 1440:
        raise Stage10APreflightError("formal sample IDs are not unique")

    train_mask = np.isin(font_ids, DEVELOPMENT_FONTS)
    test_mask = np.isin(font_ids, LOCKED_TEST_FONTS)
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise Stage10APreflightError("development and locked-test metadata overlap or omit samples")
    if not np.all(np.isin(splits[train_mask], ("train", "validation"))):
        raise Stage10APreflightError("final refit metadata has an unexpected split label")
    if not np.all(splits[test_mask] == "test"):
        raise Stage10APreflightError("locked-test metadata has an unexpected split label")
    train_count = int(train_mask.sum())
    test_count = int(test_mask.sum())
    train_classes = len(set(map(int, class_ids[train_mask])))
    test_classes = len(set(map(int, class_ids[test_mask])))
    return MetadataAudit(
        train_sample_count=train_count,
        test_sample_count=test_count,
        train_font_ids=tuple(sorted(set(map(str, font_ids[train_mask])))),
        test_font_ids=tuple(sorted(set(map(str, font_ids[test_mask])))),
        arrays_read=arrays_read,
        images_array_read=False,
        train_classes=train_classes,
        test_classes=test_classes,
    )


def formal_output_root_state(project_root: str | Path) -> str:
    root = Path(project_root).resolve() / FORMAL_OUTPUT_ROOT
    if not root.exists():
        return "absent"
    if not root.is_dir():
        raise Stage10APreflightError("formal output root exists but is not a directory")
    if any(root.iterdir()):
        raise Stage10APreflightError("formal output root is not empty")
    return "empty"


def _run_id(method: str, budget: int, mapping_type: str, seed: int) -> str:
    if mapping_type == "moderate":
        return f"stage10_moderate_s{seed}_{method}_b{budget:03d}"
    if mapping_type == "ideal":
        return f"stage10_ideal_s0000_{method}_b{budget:03d}"
    raise Stage10APreflightError(f"unknown mapping type: {mapping_type}")


def build_conditions() -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for method in METHODS:
        for budget in BUDGETS:
            for seed in FORMAL_MODERATE_SEEDS:
                run_id = _run_id(method, budget, "moderate", seed)
                conditions.append(
                    {
                        "run_id": run_id,
                        "condition_role": "formal_main",
                        "method": method,
                        "budget": budget,
                        "mapping_type": "moderate",
                        "mapping_seed": seed,
                        "train_font_ids": list(DEVELOPMENT_FONTS),
                        "test_font_ids": list(LOCKED_TEST_FONTS),
                        "formal_parameters_sha256": FORMAL_PARAMETERS_SHA256,
                        "encoder_implementation_bundle_sha256": ENCODER_BUNDLE_SHA256,
                        "benchmark_manifest_sha256": STAGE8_MANIFEST_SHA256,
                        "expected_train_sample_count": EXPECTED_TRAIN_SAMPLES,
                        "expected_test_sample_count": EXPECTED_TEST_SAMPLES,
                        "planned_output_path": (
                            FORMAL_OUTPUT_ROOT / run_id
                        ).as_posix(),
                    }
                )
    for method in METHODS:
        budget = IDEAL_AUXILIARY_BUDGET
        run_id = _run_id(method, budget, "ideal", 0)
        conditions.append(
            {
                "run_id": run_id,
                "condition_role": "formal_auxiliary",
                "method": method,
                "budget": budget,
                "mapping_type": "ideal",
                "mapping_seed": 0,
                "train_font_ids": list(DEVELOPMENT_FONTS),
                "test_font_ids": list(LOCKED_TEST_FONTS),
                "formal_parameters_sha256": FORMAL_PARAMETERS_SHA256,
                "encoder_implementation_bundle_sha256": ENCODER_BUNDLE_SHA256,
                "benchmark_manifest_sha256": STAGE8_MANIFEST_SHA256,
                "expected_train_sample_count": EXPECTED_TRAIN_SAMPLES,
                "expected_test_sample_count": EXPECTED_TEST_SAMPLES,
                "planned_output_path": (
                    FORMAL_OUTPUT_ROOT / run_id
                ).as_posix(),
            }
        )
    return conditions


def _result_builder_contract() -> dict[str, Any]:
    return {
        "status": "frozen_contract_not_executed_in_stage10a",
        "allowed_performance_input_glob": (
            FORMAL_OUTPUT_ROOT / "*" / "predictions.csv"
        ).as_posix(),
        "allowed_nonperformance_metadata": [PLAN_RELATIVE_PATH.as_posix()],
        "forbidden_input_paths": [
            STAGE9_VALIDATION_PREDICTIONS.as_posix(),
            "outputs/stage4_5",
            "outputs/stage5",
            "outputs/stage6",
            "outputs/stage7",
            "data/processed/benchmark/glyphs.npz",
        ],
        "forbidden_operations": [
            "encoder_execution",
            "spv_generation",
            "hog_extraction",
            "classifier_fit",
            "classifier_predict",
        ],
        "prediction_source_requirement": "stage10b_saved_formal_predictions_only",
        "retraining_allowed": False,
        "encoder_rerun_allowed": False,
    }


def build_run_plan(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    conditions = build_conditions()
    return {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "status": "FROZEN_STAGE10A_FINAL_RUN_PLAN_NO_PERFORMANCE_ACCESS",
        "stage10a_scope": {
            "audit_only": True,
            "locked_test_pixels_read": False,
            "spv_generated": False,
            "hog_generated": False,
            "classifier_trained": False,
            "predictions_generated": False,
            "accuracy_generated": False,
            "automatic_stage10b_entry": False,
        },
        "hash_bindings": {
            "benchmark_manifest_sha256": STAGE8_MANIFEST_SHA256,
            "stage5_5_truth_table_sha256": STAGE5_5_TRUTH_TABLE_SHA256,
            "formal_parameters_sha256": FORMAL_PARAMETERS_SHA256,
            "formal_experiment_config_sha256": FORMAL_EXPERIMENT_CONFIG_SHA256,
            "encoder_implementation_bundle_sha256": ENCODER_BUNDLE_SHA256,
            "stage10a_preflight_implementation_sha256": sha256_file(
                root / "src" / "stage10a_preflight.py"
            ),
        },
        "frozen_parameters": {
            "alpha_s": 1.0,
            "alpha_e": 0.5,
            "alpha_j": 0.5,
        },
        "methods": list(METHODS),
        "budgets": list(BUDGETS),
        "mapping_policy": {
            "formal_main": {
                "type": "moderate",
                "seeds": list(FORMAL_MODERATE_SEEDS),
                "budgets": list(BUDGETS),
            },
            "formal_auxiliary": {
                "type": "ideal",
                "seed": 0,
                "budgets": [IDEAL_AUXILIARY_BUDGET],
            },
        },
        "data_split": {
            "final_refit_font_ids": list(DEVELOPMENT_FONTS),
            "final_test_font_ids": list(LOCKED_TEST_FONTS),
            "expected_train_sample_count": EXPECTED_TRAIN_SAMPLES,
            "expected_test_sample_count": EXPECTED_TEST_SAMPLES,
            "development_locked_test_overlap": [],
        },
        "condition_counts": {
            "formal_moderate": 75,
            "ideal_auxiliary": 5,
            "total": 80,
        },
        "formal_output_root": FORMAL_OUTPUT_ROOT.as_posix(),
        "data_lineage": {
            "final_training_source": "data/processed/benchmark/glyphs.npz",
            "final_training_font_allowlist": list(DEVELOPMENT_FONTS),
            "stage9_validation_predictions_excluded": True,
            "stage9_validation_predictions_path": STAGE9_VALIDATION_PREDICTIONS.as_posix(),
            "locked_test_used_only_by_stage10b_after_access_commit": True,
        },
        "final_result_builder_contract": _result_builder_contract(),
        "conditions": conditions,
    }


def validate_run_plan(plan: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if plan.get("benchmark_id") != BENCHMARK_ID:
        raise Stage10APreflightError("run plan benchmark ID changed")
    if plan.get("status") != "FROZEN_STAGE10A_FINAL_RUN_PLAN_NO_PERFORMANCE_ACCESS":
        raise Stage10APreflightError("run plan status changed")
    if plan.get("methods") != list(METHODS) or plan.get("budgets") != list(BUDGETS):
        raise Stage10APreflightError("run plan methods or budgets changed")
    expected_hashes = {
        "benchmark_manifest_sha256": STAGE8_MANIFEST_SHA256,
        "stage5_5_truth_table_sha256": STAGE5_5_TRUTH_TABLE_SHA256,
        "formal_parameters_sha256": FORMAL_PARAMETERS_SHA256,
        "formal_experiment_config_sha256": FORMAL_EXPERIMENT_CONFIG_SHA256,
        "encoder_implementation_bundle_sha256": ENCODER_BUNDLE_SHA256,
        "stage10a_preflight_implementation_sha256": sha256_file(
            root / "src" / "stage10a_preflight.py"
        ),
    }
    if plan.get("hash_bindings") != expected_hashes:
        raise Stage10APreflightError("run plan hash bindings changed")
    conditions = plan.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 80:
        raise Stage10APreflightError("run plan must contain exactly 80 conditions")
    run_ids = [item.get("run_id") for item in conditions]
    output_paths = [item.get("planned_output_path") for item in conditions]
    if len(set(run_ids)) != 80 or len(set(output_paths)) != 80:
        raise Stage10APreflightError("run IDs or planned output paths are not unique")
    moderate = [item for item in conditions if item.get("mapping_type") == "moderate"]
    ideal = [item for item in conditions if item.get("mapping_type") == "ideal"]
    if len(moderate) != 75 or len(ideal) != 5:
        raise Stage10APreflightError("formal condition matrix is not 75+5")
    expected_moderate = {
        (method, budget, seed)
        for method in METHODS
        for budget in BUDGETS
        for seed in FORMAL_MODERATE_SEEDS
    }
    actual_moderate = {
        (item["method"], item["budget"], item["mapping_seed"])
        for item in moderate
    }
    if actual_moderate != expected_moderate:
        raise Stage10APreflightError("moderate condition Cartesian product changed")
    if {
        (item["method"], item["budget"], item["mapping_seed"])
        for item in ideal
    } != {(method, 64, 0) for method in METHODS}:
        raise Stage10APreflightError("ideal auxiliary conditions changed")
    for item in conditions:
        required = {
            "method",
            "budget",
            "mapping_type",
            "mapping_seed",
            "train_font_ids",
            "test_font_ids",
            "formal_parameters_sha256",
            "encoder_implementation_bundle_sha256",
            "benchmark_manifest_sha256",
            "expected_train_sample_count",
            "expected_test_sample_count",
            "planned_output_path",
            "run_id",
        }
        if not required <= set(item):
            raise Stage10APreflightError("a condition is missing required fields")
        if item["train_font_ids"] != list(DEVELOPMENT_FONTS):
            raise Stage10APreflightError("a condition changed the final refit fonts")
        if item["test_font_ids"] != list(LOCKED_TEST_FONTS):
            raise Stage10APreflightError("a condition changed the locked-test fonts")
        if item["formal_parameters_sha256"] != FORMAL_PARAMETERS_SHA256:
            raise Stage10APreflightError("a condition changed the parameter hash")
        if item["encoder_implementation_bundle_sha256"] != ENCODER_BUNDLE_SHA256:
            raise Stage10APreflightError("a condition changed the encoder hash")
        if item["benchmark_manifest_sha256"] != STAGE8_MANIFEST_SHA256:
            raise Stage10APreflightError("a condition changed the benchmark hash")
        if (
            item["expected_train_sample_count"],
            item["expected_test_sample_count"],
        ) != (EXPECTED_TRAIN_SAMPLES, EXPECTED_TEST_SAMPLES):
            raise Stage10APreflightError("a condition changed expected sample counts")
        if (root / item["planned_output_path"]).exists():
            raise Stage10APreflightError("a planned formal output path already exists")

    lineage = plan.get("data_lineage")
    if not isinstance(lineage, Mapping):
        raise Stage10APreflightError("run plan lacks data-lineage policy")
    if lineage.get("stage9_validation_predictions_excluded") is not True:
        raise Stage10APreflightError("Stage-9 validation predictions are not excluded")
    if lineage.get("final_training_font_allowlist") != list(DEVELOPMENT_FONTS):
        raise Stage10APreflightError("final training allowlist changed")
    builder = plan.get("final_result_builder_contract")
    if not isinstance(builder, Mapping):
        raise Stage10APreflightError("run plan lacks result-builder contract")
    expected_glob = (FORMAL_OUTPUT_ROOT / "*" / "predictions.csv").as_posix()
    if builder.get("allowed_performance_input_glob") != expected_glob:
        raise Stage10APreflightError("result builder input glob changed")
    if STAGE9_VALIDATION_PREDICTIONS.as_posix() not in builder.get(
        "forbidden_input_paths", []
    ):
        raise Stage10APreflightError("result builder does not forbid Stage-9 predictions")
    if builder.get("retraining_allowed") is not False or builder.get(
        "encoder_rerun_allowed"
    ) is not False:
        raise Stage10APreflightError("result builder could rerun training or encoding")
    return {
        "condition_count": 80,
        "moderate_condition_count": 75,
        "ideal_condition_count": 5,
        "unique_run_id_count": len(set(run_ids)),
        "unique_output_path_count": len(set(output_paths)),
    }


def collect_preflight_state(project_root: str | Path) -> dict[str, Any]:
    """Perform all read-only checks needed before writing the plan."""

    root = Path(project_root).resolve()
    snapshot = frozen_readonly_snapshot(root)
    if snapshot["stage8_manifest"] != STAGE8_MANIFEST_SHA256:
        raise Stage10APreflightError("Stage-8 manifest hash mismatch")
    if snapshot["stage5_5_truth_table"] != STAGE5_5_TRUTH_TABLE_SHA256:
        raise Stage10APreflightError("Stage-5.5 truth-table hash mismatch")
    if snapshot["formal_parameters"] != FORMAL_PARAMETERS_SHA256:
        raise Stage10APreflightError("formal parameter hash mismatch")
    if snapshot["formal_experiment_config"] != FORMAL_EXPERIMENT_CONFIG_SHA256:
        raise Stage10APreflightError("formal experiment config hash mismatch")
    if snapshot["encoder_implementation_bundle"] != ENCODER_BUNDLE_SHA256:
        raise Stage10APreflightError("encoder implementation bundle changed")

    parameters = _read_json(root / "configs" / "formal_parameters_frozen.json")
    experiment = _read_json(root / "configs" / "formal_experiment_frozen.json")
    manifest = _read_json(root / "data" / "benchmark" / "benchmark_manifest.json")
    if (
        parameters.get("alpha_s"),
        parameters.get("alpha_e"),
        parameters.get("alpha_j"),
    ) != (1.0, 0.5, 0.5):
        raise Stage10APreflightError("formal alpha values changed")
    if parameters.get("final_methods") != list(METHODS):
        raise Stage10APreflightError("formal method list changed")
    if experiment.get("methods") != list(METHODS):
        raise Stage10APreflightError("formal experiment method list changed")
    if manifest["protocol"].get("methods") != list(METHODS):
        raise Stage10APreflightError("Stage-8 manifest method list changed")
    if parameters.get("formal_budgets") != list(BUDGETS):
        raise Stage10APreflightError("formal budget list changed")
    if experiment.get("budgets") != list(BUDGETS):
        raise Stage10APreflightError("formal experiment budgets changed")
    if parameters.get("formal_moderate_mapping_seeds") != list(FORMAL_MODERATE_SEEDS):
        raise Stage10APreflightError("formal moderate seed list changed")
    if parameters.get("ideal_auxiliary_budget") != IDEAL_AUXILIARY_BUDGET:
        raise Stage10APreflightError("ideal auxiliary budget changed")
    if parameters.get("final_refit_fonts") != list(DEVELOPMENT_FONTS):
        raise Stage10APreflightError("final refit fonts changed")
    if parameters.get("final_test_fonts") != list(LOCKED_TEST_FONTS):
        raise Stage10APreflightError("final test fonts changed")
    if set(DEVELOPMENT_FONTS) & set(LOCKED_TEST_FONTS):
        raise Stage10APreflightError("development and locked-test font identities overlap")

    metadata = load_formal_sample_metadata(root)
    if metadata.train_sample_count != EXPECTED_TRAIN_SAMPLES:
        raise Stage10APreflightError("final refit sample count is not 1080")
    if metadata.test_sample_count != EXPECTED_TEST_SAMPLES:
        raise Stage10APreflightError("locked-test sample count is not 360")
    if metadata.train_classes != 180 or metadata.test_classes != 180:
        raise Stage10APreflightError("final refit/test class coverage is not 180/180")

    output_state = formal_output_root_state(root)
    guard = validate_formal_freeze_integrity(root)
    if guard["locked_test_performance_access_count"] != 0:
        raise Stage10APreflightError("locked-test performance access count is nonzero")
    if guard["formal_mapping_performance_access_count"] != 0:
        raise Stage10APreflightError("formal mapping performance access count is nonzero")
    performance_ledger = _read_json(
        root / "data" / "benchmark" / "formal_performance_access_ledger.json"
    )
    formal_seed_counts = {
        str(seed): int(
            performance_ledger["formal_mapping_seeds"][str(seed)][
                "performance_access_count"
            ]
        )
        for seed in FORMAL_MODERATE_SEEDS
    }
    if any(formal_seed_counts.values()):
        raise Stage10APreflightError("a formal seed performance count is nonzero")
    return {
        "snapshot": snapshot,
        "parameters": parameters,
        "experiment": experiment,
        "metadata": {
            "train_sample_count": metadata.train_sample_count,
            "test_sample_count": metadata.test_sample_count,
            "train_font_ids": list(metadata.train_font_ids),
            "test_font_ids": list(metadata.test_font_ids),
            "arrays_read": list(metadata.arrays_read),
            "images_array_read": metadata.images_array_read,
            "train_classes": metadata.train_classes,
            "test_classes": metadata.test_classes,
        },
        "formal_output_root_state": output_state,
        "guard": guard,
        "formal_seed_performance_access_counts": formal_seed_counts,
    }


def build_numbered_checks(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_audit: Mapping[str, Any],
    *,
    historical_test_evidence: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return the user's 25 requested checks in the same numbered order."""

    parameters = state["parameters"]
    experiment = state["experiment"]
    metadata = state["metadata"]
    snapshot = state["snapshot"]
    conditions = plan["conditions"]
    builder = plan["final_result_builder_contract"]
    ledger = state["formal_seed_performance_access_counts"]

    evidence = [
        (snapshot["stage8_manifest"] == STAGE8_MANIFEST_SHA256, snapshot["stage8_manifest"]),
        (snapshot["stage5_5_truth_table"] == STAGE5_5_TRUTH_TABLE_SHA256, snapshot["stage5_5_truth_table"]),
        (snapshot["encoder_implementation_bundle"] == ENCODER_BUNDLE_SHA256, snapshot["encoder_implementation_bundle"]),
        (snapshot["formal_parameters"] == FORMAL_PARAMETERS_SHA256, snapshot["formal_parameters"]),
        (snapshot["formal_experiment_config"] == FORMAL_EXPERIMENT_CONFIG_SHA256, snapshot["formal_experiment_config"]),
        ((parameters["alpha_s"], parameters["alpha_e"], parameters["alpha_j"]) == (1.0, 0.5, 0.5), [parameters["alpha_s"], parameters["alpha_e"], parameters["alpha_j"]]),
        (parameters["final_methods"] == list(METHODS) == experiment["methods"], parameters["final_methods"]),
        (parameters["formal_budgets"] == list(BUDGETS) == experiment["budgets"], parameters["formal_budgets"]),
        (parameters["formal_moderate_mapping_seeds"] == list(FORMAL_MODERATE_SEEDS), parameters["formal_moderate_mapping_seeds"]),
        (parameters["ideal_auxiliary_budget"] == 64, parameters["ideal_auxiliary_budget"]),
        (parameters["final_refit_fonts"] == list(DEVELOPMENT_FONTS), parameters["final_refit_fonts"]),
        (parameters["final_test_fonts"] == list(LOCKED_TEST_FONTS), parameters["final_test_fonts"]),
        (metadata["train_sample_count"] == 1080, metadata["train_sample_count"]),
        (metadata["test_sample_count"] == 360, metadata["test_sample_count"]),
        (not (set(DEVELOPMENT_FONTS) & set(LOCKED_TEST_FONTS)), []),
        (plan_audit["moderate_condition_count"] == 75, plan_audit["moderate_condition_count"]),
        (plan_audit["ideal_condition_count"] == 5, plan_audit["ideal_condition_count"]),
        (plan_audit["condition_count"] == 80, plan_audit["condition_count"]),
        (plan_audit["unique_run_id_count"] == 80, plan_audit["unique_run_id_count"]),
        (state["formal_output_root_state"] in {"absent", "empty"}, state["formal_output_root_state"]),
        (state["guard"]["locked_test_performance_access_count"] == 0, 0),
        (all(count == 0 for count in ledger.values()), ledger),
        (
            plan["data_lineage"]["stage9_validation_predictions_excluded"] is True
            and all(
                STAGE9_VALIDATION_PREDICTIONS.as_posix()
                not in json.dumps(condition, ensure_ascii=False)
                for condition in conditions
            ),
            "Stage-9 predictions excluded from all 80 conditions",
        ),
        (
            builder["prediction_source_requirement"]
            == "stage10b_saved_formal_predictions_only"
            and builder["retraining_allowed"] is False
            and builder["encoder_rerun_allowed"] is False,
            builder,
        ),
        (
            historical_test_evidence is not None
            and historical_test_evidence.get("exit_code") == 0
            and int(historical_test_evidence.get("passed", 0)) >= 87,
            historical_test_evidence or "PENDING",
        ),
    ]
    names = [
        "Stage 8 benchmark manifest hash",
        "Stage 5.5 truth table hash",
        "encoder implementation bundle hash",
        "formal_parameters_frozen.json hash",
        "formal_experiment_frozen.json hash",
        "formal alpha tuple",
        "five fixed methods",
        "three fixed budgets",
        "five formal moderate seeds",
        "ideal auxiliary B=64 only",
        "six final refit fonts",
        "two final test fonts",
        "expected final refit samples",
        "expected locked-test samples",
        "font-role disjointness",
        "75 moderate conditions",
        "5 ideal auxiliary conditions",
        "80 total conditions",
        "80 unique run IDs",
        "formal output root clean",
        "locked-test performance access count zero",
        "all formal seed performance counts zero",
        "Stage-9 predictions excluded",
        "final result builder input/operation boundary",
        "historical and current tests",
    ]
    checks: list[dict[str, Any]] = []
    for index, (name, (passed, detail)) in enumerate(zip(names, evidence, strict=True), start=1):
        status = "PASS" if passed else ("PENDING" if index == 25 and historical_test_evidence is None else "FAIL")
        if status == "FAIL":
            raise Stage10APreflightError(f"preflight check {index} failed: {name}")
        checks.append(
            {
                "check_id": index,
                "name": name,
                "status": status,
                "detail": detail,
            }
        )
    return checks
