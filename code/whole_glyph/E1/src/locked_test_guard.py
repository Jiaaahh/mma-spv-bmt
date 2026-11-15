"""Mandatory integrity and single-access gate for locked-test performance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.deterministic_io import sha256_file, tree_sha256
from src.formal_benchmark import BENCHMARK_ID, FORMAL_FONT_SPECS


TECHNICAL_QC_OPERATIONS = frozenset(
    {
        "file_hash",
        "character_coverage",
        "glyph_render_qc",
        "shape_range_empty_check",
        "contact_sheet",
    }
)
PERFORMANCE_OPERATIONS = frozenset(
    {
        "classifier_evaluation",
        "accuracy",
        "macro_f1",
        "confusion_matrix",
        "method_comparison",
    }
)
LOCKED_FONT_IDS = frozenset(
    spec.font_id for spec in FORMAL_FONT_SPECS if spec.split_role == "locked_test"
)
PARAMETER_FREEZE_FLAG = Path("configs/formal_parameters_frozen.json")
PARAMETER_FREEZE_SHA = Path("configs/formal_parameters_frozen.sha256")
FORMAL_EXPERIMENT_CONFIG = Path("configs/formal_experiment_frozen.json")
STAGE8_MANIFEST = Path("data/benchmark/benchmark_manifest.json")
STAGE5_5_TRUTH_TABLE = Path("protocols/STAGE5_5_CODE_TRUTH_TABLE.yaml")
FORMAL_PERFORMANCE_LEDGER = Path(
    "data/benchmark/formal_performance_access_ledger.json"
)
STAGE8_MAPPING_LEDGER = Path("data/mappings/formal/mapping_access_ledger.json")
FORMAL_PARAMETER_APPLICATION_MODULE = Path("src/stage9_development.py")


class LockedTestAccessError(PermissionError):
    """Raised before a forbidden locked-test performance operation can run."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockedTestAccessError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise LockedTestAccessError(f"{label} must be a JSON object")
    return value


def _encoder_bundle_paths(project_root: Path) -> list[Path]:
    return [
        *sorted((project_root / "src" / "encoders").glob("*.py")),
        project_root / "src" / "coverage.py",
        project_root / "src" / "eb_drive.py",
        project_root / "src" / "eb_representation.py",
        project_root / "src" / "topology.py",
        project_root / "src" / "spv_forward.py",
    ]


def validate_formal_freeze_integrity(project_root: str | Path) -> dict[str, Any]:
    """Validate every Stage-9 hash binding before the one final test access."""

    root = Path(project_root).resolve()
    parameter_path = root / PARAMETER_FREEZE_FLAG
    parameter_sha_path = root / PARAMETER_FREEZE_SHA
    if not parameter_path.is_file() or not parameter_sha_path.is_file():
        raise LockedTestAccessError(
            "formal parameter-freeze file or SHA sidecar is missing"
        )
    expected_parameter_sha = parameter_sha_path.read_text(
        encoding="ascii"
    ).split()[0]
    actual_parameter_sha = sha256_file(parameter_path)
    if actual_parameter_sha != expected_parameter_sha:
        raise LockedTestAccessError("formal parameter-freeze SHA-256 mismatch")
    value = _read_json(parameter_path, "formal parameter-freeze flag")
    required = {
        "benchmark_id": BENCHMARK_ID,
        "status": "formal_parameters_frozen",
        "guard_state": "PARAMETERS_FROZEN_FINAL_EVALUATION_NOT_YET_RUN",
        "locked_test_access_enabled": True,
        "frozen_before_locked_test": True,
        "final_evaluation_run": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise LockedTestAccessError(
                f"formal parameter-freeze flag has invalid {key!r}"
            )
    alpha_e = value.get("alpha_e")
    alpha_j = value.get("alpha_j")
    if alpha_e not in {0.5, 1.0, 2.0} or alpha_j not in {0.5, 1.0, 2.0}:
        raise LockedTestAccessError("formal alpha_E/alpha_J are absent or invalid")
    manifest_hash = value.get("benchmark_manifest_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise LockedTestAccessError("formal flag lacks a benchmark manifest SHA-256")

    manifest_path = root / STAGE8_MANIFEST
    if sha256_file(manifest_path) != manifest_hash:
        raise LockedTestAccessError("Stage-8 benchmark manifest hash changed")
    manifest = _read_json(manifest_path, "Stage-8 benchmark manifest")
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise LockedTestAccessError("Stage-8 benchmark ID changed")

    truth_hash = value.get("stage5_5_truth_table_sha256")
    if sha256_file(root / STAGE5_5_TRUTH_TABLE) != truth_hash:
        raise LockedTestAccessError("Stage-5.5 truth-table hash changed")
    expected_encoder_hash = value.get("encoder_implementation_bundle_sha256")
    actual_encoder_hash = tree_sha256(root, _encoder_bundle_paths(root))
    if actual_encoder_hash != expected_encoder_hash:
        raise LockedTestAccessError("encoder implementation bundle hash changed")

    application_hash = value.get("formal_parameter_application_module_sha256")
    if sha256_file(root / FORMAL_PARAMETER_APPLICATION_MODULE) != application_hash:
        raise LockedTestAccessError("formal EB parameter-application module changed")
    formal_config_path = root / FORMAL_EXPERIMENT_CONFIG
    formal_config_hash = value.get("formal_experiment_config_sha256")
    if sha256_file(formal_config_path) != formal_config_hash:
        raise LockedTestAccessError("formal experiment config hash changed")
    formal_config = _read_json(formal_config_path, "formal experiment config")
    if formal_config.get("guard_state") != required["guard_state"]:
        raise LockedTestAccessError("formal experiment config guard state changed")
    formal_parameters = formal_config.get("formal_parameters")
    if not isinstance(formal_parameters, dict):
        raise LockedTestAccessError("formal experiment config lacks parameters")
    if (
        formal_parameters.get("alpha_e") != alpha_e
        or formal_parameters.get("alpha_j") != alpha_j
        or formal_parameters.get("shared_by") != ["eb_topk", "eb_sse"]
    ):
        raise LockedTestAccessError("formal experiment config parameter binding changed")

    performance_ledger = _read_json(
        root / FORMAL_PERFORMANCE_LEDGER,
        "formal performance access ledger",
    )
    if performance_ledger.get("formal_parameters_sha256") != actual_parameter_sha:
        raise LockedTestAccessError("performance ledger parameter hash mismatch")
    if performance_ledger.get("formal_experiment_config_sha256") != formal_config_hash:
        raise LockedTestAccessError("performance ledger experiment config hash mismatch")
    if performance_ledger.get("guard_state") != required["guard_state"]:
        raise LockedTestAccessError("performance ledger guard state changed")
    if int(performance_ledger.get("locked_test_performance_access_count", -1)) != 0:
        raise LockedTestAccessError("locked-test final performance has already been accessed")
    if int(performance_ledger.get("formal_mapping_performance_access_count", -1)) != 0:
        raise LockedTestAccessError("formal mapping performance has already been accessed")
    if performance_ledger.get("final_evaluation_events"):
        raise LockedTestAccessError("formal final-evaluation ledger is not empty")

    stage8_mapping_ledger = _read_json(
        root / STAGE8_MAPPING_LEDGER,
        "Stage-8 mapping access ledger",
    )
    if int(stage8_mapping_ledger.get("formal_mapping_performance_access_count", -1)) != 0:
        raise LockedTestAccessError("Stage-8 formal mapping access ledger is nonzero")
    return {
        "benchmark_id": BENCHMARK_ID,
        "guard_state": required["guard_state"],
        "formal_parameters_sha256": actual_parameter_sha,
        "benchmark_manifest_sha256": manifest_hash,
        "encoder_implementation_bundle_sha256": actual_encoder_hash,
        "formal_experiment_config_sha256": formal_config_hash,
        "locked_test_performance_access_count": 0,
        "formal_mapping_performance_access_count": 0,
    }


def _validate_parameter_freeze_flag(path: Path) -> None:
    """Backward-compatible helper retained for callers that hold the flag path."""

    resolved = path.resolve()
    try:
        project_root = resolved.parents[1]
    except IndexError as exc:
        raise LockedTestAccessError("cannot resolve project root from formal flag") from exc
    validate_formal_freeze_integrity(project_root)


def assert_locked_test_access(
    *,
    operation: str,
    font_ids: Iterable[str],
    project_root: str | Path,
) -> None:
    """Allow technical QC, but block locked-test performance until a valid flag exists."""

    operation_name = str(operation).strip().lower()
    if operation_name not in TECHNICAL_QC_OPERATIONS | PERFORMANCE_OPERATIONS:
        raise LockedTestAccessError(f"unknown formal access operation: {operation!r}")
    targets = frozenset(map(str, font_ids))
    unknown = targets - {spec.font_id for spec in FORMAL_FONT_SPECS}
    if unknown:
        raise LockedTestAccessError(f"unknown formal font IDs: {sorted(unknown)}")
    locked_targets = targets & LOCKED_FONT_IDS
    if not locked_targets:
        return
    if operation_name in TECHNICAL_QC_OPERATIONS:
        return
    flag_path = Path(project_root).resolve() / PARAMETER_FREEZE_FLAG
    if not flag_path.is_file():
        raise LockedTestAccessError(
            "locked-test performance access refused: formal alpha_E/alpha_J have not "
            "been selected and frozen on development fonts"
        )
    validate_formal_freeze_integrity(Path(project_root).resolve())
