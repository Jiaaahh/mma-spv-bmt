"""Stage-10B formal-session preparation and crash-safe checkpoint contracts.

This module never imports or invokes an encoder, the SPV forward model, HOG,
the classifier, or a metric implementation.  Preparation hashes frozen files
as opaque bytes; it never indexes ``glyphs.npz`` or reads its ``images`` array.

The actual Stage-10B runner may later use ``create_run_staging_dir``,
``record_formal_run_started`` and ``commit_completed_run``.  None of those
functions is called by restart preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import secrets
import sys
from typing import Any, Mapping, Sequence

from src.deterministic_io import sha256_bytes, sha256_file, tree_sha256
from src.locked_test_guard import validate_formal_freeze_integrity
from src.stage10a_preflight import (
    BENCHMARK_ID,
    Stage10APreflightError,
    encoder_bundle_paths,
    validate_run_plan,
)


RUN_PLAN_PATH = Path("plans/stage10_final_run_plan.json")
RUN_PLAN_SHA_PATH = Path("plans/stage10_final_run_plan.sha256")
ANALYSIS_PLAN_PATH = Path("plans/stage10_final_analysis_plan.json")
ANALYSIS_PLAN_SHA_PATH = Path("plans/stage10_final_analysis_plan.sha256")
STAGE10_ROOT = Path("outputs/stage10")
SESSION_ROOT = STAGE10_ROOT / "sessions"
ACTIVE_SESSION_POINTER = STAGE10_ROOT / "ACTIVE_SESSION.json"
FORMAL_RUNS_ROOT = STAGE10_ROOT / "formal_runs"
WORK_ROOT = STAGE10_ROOT / "work"

EXPECTED_RUN_PLAN_SHA256 = (
    "4dcedc5fc538f2256ab7a9ec578e1c95a6b4d085c42acc3b2298c770c8332697"
)
EXPECTED_ENCODER_BUNDLE_SHA256 = (
    "4183a48cf6c9e9001fceb0ad0bc39223f4bf16e06328d27db24e9cc8c4baa367"
)

EXPECTED_FROZEN_FILES: dict[str, tuple[Path, str]] = {
    "benchmark_manifest": (
        Path("data/benchmark/benchmark_manifest.json"),
        "452b514b4a2c7e5dd7b10c13a37656571cf56181f0115b0801ef89287ea369ab",
    ),
    "stage5_5_truth_table": (
        Path("protocols/STAGE5_5_CODE_TRUTH_TABLE.yaml"),
        "3559d22eba27bd195bef99a29169307c21f0898b2869b113ff1c09af9dace97e",
    ),
    "formal_parameters": (
        Path("configs/formal_parameters_frozen.json"),
        "8ea8e25da16cc9ccdcb96f650b5ed5a9ec4f2d59bacbba2505f48a997398dd2c",
    ),
    "formal_parameters_sidecar": (
        Path("configs/formal_parameters_frozen.sha256"),
        "34a5ba4c532c6946124410a704114843bfbc9c80ccd91c28f1161c1e6582092e",
    ),
    "formal_experiment_config": (
        Path("configs/formal_experiment_frozen.json"),
        "6cf4e77d441b932841d7a981c2b0320346b4cadb714b65c8139e1cbefd58a2c2",
    ),
    "formal_parameter_application_module": (
        Path("src/stage9_development.py"),
        "0c7d2ec3a420c40430a0e966fe0657f004219386e1553951cbccc2eac0903fbd",
    ),
    "formal_glyph_archive": (
        Path("data/processed/benchmark/glyphs.npz"),
        "8427140836b2358aa48ff9c7bfc07cfdbe0f02ee166fb8b2e671ff7cb8a8d582",
    ),
    "font_split": (
        Path("data/registry/font_split_formal.json"),
        "15d1c65919160943e628d7b4d5ef084338a509e9990a15db0d84cdbbe200d3e4",
    ),
    "mapping_bank_manifest": (
        Path("data/mappings/formal/mapping_bank_manifest.json"),
        "093e46f0c6553f39efe6809db1fafd9132c1fb91bbda5c771e8c6b3a3e05c1d3",
    ),
    "ideal_grid": (
        Path("data/mappings/formal/ideal_grid.npz"),
        "f5fafa93355ea3daca51a0532ba1bc0508f56da3572d6f77b24e44daa8b999af",
    ),
    "ideal_mapping": (
        Path("data/mappings/formal/ideal_mapping.npz"),
        "267cd5ebfaa9ff4093f950fbe6df66442cf3f5115419c677808c1db34c617d5a",
    ),
    "moderate_mapping_2026": (
        Path("data/mappings/formal/moderate_seed_2026.npz"),
        "46b543400ff709568e462c61003c57b6fc29f39ddccbc2030df7ef0464079b62",
    ),
    "moderate_mapping_2027": (
        Path("data/mappings/formal/moderate_seed_2027.npz"),
        "b9d11eb0b9ba96b02f1caca911f80f66430296c5a3a12c2b371ade53c7b74ddd",
    ),
    "moderate_mapping_2028": (
        Path("data/mappings/formal/moderate_seed_2028.npz"),
        "30a076f1a70044bd3fc93f5bae13108b93db074372bb3661cd72d433936b1d92",
    ),
    "moderate_mapping_2029": (
        Path("data/mappings/formal/moderate_seed_2029.npz"),
        "4448dca4eb15848124f30398f855d75a0f9dae202e0a7cc115411dfac91d1925",
    ),
    "moderate_mapping_2030": (
        Path("data/mappings/formal/moderate_seed_2030.npz"),
        "f3c9330b31d973680c099cd0f46675b56c8009b0e6781e3a039b4cdc79be9a99",
    ),
    "stage10a_formal_performance_ledger": (
        Path("data/benchmark/formal_performance_access_ledger.json"),
        "1961c5b8db6f5e968c7bfc5ed61580dadfe9d865ddd21edb712805c682c4dac4",
    ),
    "stage8_mapping_access_ledger": (
        Path("data/mappings/formal/mapping_access_ledger.json"),
        "725ffa4d5f2013fdefaf0bf88d0e6bec19511ab78395020da1d685b7a6c53a4b",
    ),
}

REQUIRED_PREDICTION_COLUMNS = (
    "benchmark_id",
    "run_id",
    "method",
    "budget",
    "mapping_type",
    "mapping_seed",
    "char_id",
    "font_id",
    "true_label",
    "pred_label",
    "correct",
)
REQUIRED_COMPLETED_RUN_FILES = (
    "predictions.csv",
    "metrics.json",
    "run_manifest.json",
)
SESSION_ID_PATTERN = re.compile(
    r"^stage10b_restart_\d{8}T\d{6}Z_[0-9a-f]{8}_[0-9a-f]{8}$"
)


class Stage10BSessionError(ValueError):
    """Raised before an invalid session or checkpoint can be used."""


@dataclass(frozen=True, slots=True)
class ResumeState:
    formal_session_id: str
    completed_count: int
    total_count: int
    next_run_id: str | None
    completed_run_ids: tuple[str, ...]
    remaining_run_ids: tuple[str, ...]
    safe_to_resume: bool
    status: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _strict_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(path, _strict_json_bytes(value))


def write_sha_sidecar(path: Path, sidecar: Path | None = None) -> str:
    digest = sha256_file(path)
    destination = sidecar or path.with_name(path.name + ".sha256")
    _write_bytes_atomic(destination, f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise Stage10BSessionError(f"non-finite JSON constant {value!r}: {path}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage10BSessionError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise Stage10BSessionError(f"JSON root must be an object: {path}")
    return value


def _verify_sidecar(path: Path, sidecar: Path | None = None) -> str:
    digest = sha256_file(path)
    checksum_path = sidecar or path.with_name(path.name + ".sha256")
    try:
        expected = checksum_path.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as exc:
        raise Stage10BSessionError(f"missing SHA sidecar: {checksum_path}") from exc
    if digest != expected:
        raise Stage10BSessionError(f"SHA mismatch: {path}")
    return digest


def _environment_snapshot(project_root: Path) -> dict[str, Any]:
    package_versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "scikit-image", "scikit-learn", "Pillow"):
        try:
            package_versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            package_versions[name] = "NOT_INSTALLED"
    requirements = project_root / "requirements.txt"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable_name": Path(sys.executable).name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_only_protocol": True,
        "package_versions": package_versions,
        "requirements_sha256": sha256_file(requirements),
    }


def build_analysis_plan() -> dict[str, Any]:
    """Return the result-independent analysis plan frozen before Stage-10B."""

    return {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "status": "FROZEN_BEFORE_STAGE10B_RESTART_NO_PERFORMANCE_ACCESS",
        "run_plan_sha256": EXPECTED_RUN_PLAN_SHA256,
        "primary_results": {
            "mapping_type": "moderate",
            "condition_count": 75,
            "methods": [
                "raw_topk",
                "edge_topk",
                "skeleton_topk",
                "eb_topk",
                "eb_sse",
            ],
            "budgets": [32, 64, 128],
            "mapping_seeds": [2026, 2027, 2028, 2029, 2030],
            "primary_metric": "top1_accuracy",
            "auxiliary_metric": "macro_f1",
            "report_all_conditions_without_selection": True,
        },
        "auxiliary_results": {
            "mapping_type": "ideal",
            "budget": 64,
            "condition_count": 5,
            "reported_separately_from_moderate": True,
        },
        "preregistered_comparisons": [
            {
                "comparison": "eb_topk_minus_skeleton_topk",
                "interpretation": "increment_of_full_eb_topology_over_skeleton_only",
            },
            {
                "comparison": "eb_sse_minus_eb_topk",
                "interpretation": "increment_of_coverage_greedy_over_topk_with_same_eb_drive_and_utility",
            },
            {
                "comparison": "eb_sse_vs_raw_edge_skeleton",
                "interpretation": "overall_method_comparison",
            },
        ],
        "uncertainty": {
            "method": "paired_character_level_bootstrap_percentile_ci",
            "resampling_unit": "char_id",
            "paired_observations_per_character": {
                "locked_test_fonts": 2,
                "moderate_mappings": 5,
            },
            "repetitions": 5000,
            "random_seed": 2026,
            "confidence_level": 0.95,
            "quantity": "accuracy_difference",
            "additional_p_value_system": False,
        },
        "selection_prohibitions": {
            "best_seed_only": False,
            "best_budget_only": False,
            "drop_unfavorable_conditions": False,
            "change_analysis_after_results": False,
        },
    }


def ensure_analysis_plan(project_root: str | Path) -> str:
    root = Path(project_root).resolve()
    path = root / ANALYSIS_PLAN_PATH
    sidecar = root / ANALYSIS_PLAN_SHA_PATH
    expected = build_analysis_plan()
    if path.exists() != sidecar.exists():
        raise Stage10BSessionError("analysis plan and SHA sidecar are incomplete")
    if not path.exists():
        write_json_atomic(path, expected)
        write_sha_sidecar(path, sidecar)
    else:
        if _read_json(path) != expected:
            raise Stage10BSessionError("saved analysis plan differs from frozen plan")
        _verify_sidecar(path, sidecar)
    return sha256_file(path)


def validate_frozen_state(
    project_root: str | Path,
    *,
    require_clean_formal_runs: bool = True,
) -> dict[str, Any]:
    """Validate every Stage-10A binding without reading glyph arrays.

    ``require_clean_formal_runs`` is true for restart preparation, where any
    pre-existing condition output is forbidden.  Resume validation sets it to
    false because completed outputs are checked separately against the
    hash-chained completed-run manifest later in :func:`load_resume_state`.
    """

    root = Path(project_root).resolve()
    run_plan_path = root / RUN_PLAN_PATH
    actual_run_plan_sha = _verify_sidecar(run_plan_path, root / RUN_PLAN_SHA_PATH)
    if actual_run_plan_sha != EXPECTED_RUN_PLAN_SHA256:
        raise Stage10BSessionError("Stage-10 run-plan SHA changed")
    plan = _read_json(run_plan_path)
    try:
        if require_clean_formal_runs:
            plan_audit = validate_run_plan(plan, root)
        else:
            # The exact plan bytes are already bound by the frozen SHA above.
            # Stage-10A's validator additionally asserts that every planned
            # output path is absent, which is valid only before execution.
            conditions = plan.get("conditions")
            if not isinstance(conditions, list) or len(conditions) != 80:
                raise Stage10BSessionError("frozen run plan condition count changed")
            run_ids = [item.get("run_id") for item in conditions]
            output_paths = [item.get("planned_output_path") for item in conditions]
            if len(set(run_ids)) != 80 or len(set(output_paths)) != 80:
                raise Stage10BSessionError(
                    "frozen run IDs or planned output paths are not unique"
                )
            plan_audit = {
                "condition_count": 80,
                "moderate_condition_count": sum(
                    item.get("mapping_type") == "moderate" for item in conditions
                ),
                "ideal_condition_count": sum(
                    item.get("mapping_type") == "ideal" for item in conditions
                ),
                "unique_run_id_count": 80,
                "unique_output_path_count": 80,
            }
            if (
                plan_audit["moderate_condition_count"],
                plan_audit["ideal_condition_count"],
            ) != (75, 5):
                raise Stage10BSessionError("frozen run plan is not the 75+5 matrix")
        guard = validate_formal_freeze_integrity(root)
    except (Stage10APreflightError, PermissionError, ValueError) as exc:
        raise Stage10BSessionError(f"Stage-10A frozen guard failed: {exc}") from exc

    hashes: dict[str, dict[str, Any]] = {}
    for name, (relative, expected) in EXPECTED_FROZEN_FILES.items():
        path = root / relative
        if not path.is_file():
            raise Stage10BSessionError(f"missing frozen file: {relative}")
        actual = sha256_file(path)
        hashes[name] = {
            "path": relative.as_posix(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        }
        if actual != expected:
            raise Stage10BSessionError(f"frozen hash mismatch: {relative}")

    encoder_hash = tree_sha256(root, encoder_bundle_paths(root))
    if encoder_hash != EXPECTED_ENCODER_BUNDLE_SHA256:
        raise Stage10BSessionError("encoder implementation bundle hash changed")

    benchmark = _read_json(root / EXPECTED_FROZEN_FILES["benchmark_manifest"][0])
    mapping_manifest = _read_json(
        root / EXPECTED_FROZEN_FILES["mapping_bank_manifest"][0]
    )
    if benchmark.get("benchmark_id") != BENCHMARK_ID:
        raise Stage10BSessionError("benchmark identity changed")
    if mapping_manifest.get("formal_seeds") != [2026, 2027, 2028, 2029, 2030]:
        raise Stage10BSessionError("formal mapping seed set changed")
    if mapping_manifest.get("status") != "generated_geometry_only_no_performance_access":
        raise Stage10BSessionError("formal mapping manifest status changed")

    performance_ledger = _read_json(
        root / EXPECTED_FROZEN_FILES["stage10a_formal_performance_ledger"][0]
    )
    mapping_ledger = _read_json(
        root / EXPECTED_FROZEN_FILES["stage8_mapping_access_ledger"][0]
    )
    if performance_ledger.get("locked_test_performance_access_count") != 0:
        raise Stage10BSessionError("Stage-10A locked-test ledger is nonzero")
    if performance_ledger.get("formal_mapping_performance_access_count") != 0:
        raise Stage10BSessionError("Stage-10A formal-mapping ledger is nonzero")
    if mapping_ledger.get("formal_mapping_performance_access_count") != 0:
        raise Stage10BSessionError("Stage-8 mapping ledger is nonzero")

    if FORMAL_RUNS_ROOT.joinpath().is_absolute():
        raise AssertionError("formal output root must remain project-relative")
    formal_runs = root / FORMAL_RUNS_ROOT
    if formal_runs.exists() and not formal_runs.is_dir():
        raise Stage10BSessionError("formal output root is not a directory")
    if (
        require_clean_formal_runs
        and formal_runs.exists()
        and any(formal_runs.iterdir())
    ):
        raise Stage10BSessionError("formal output root is not clean")

    return {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "verification_scope": "opaque_file_hashes_and_nonperformance_metadata_only",
        "locked_test_images_array_read": False,
        "run_plan": {
            "path": RUN_PLAN_PATH.as_posix(),
            "sha256": actual_run_plan_sha,
            **plan_audit,
        },
        "guard": guard,
        "encoder_implementation_bundle": {
            "expected_sha256": EXPECTED_ENCODER_BUNDLE_SHA256,
            "actual_sha256": encoder_hash,
            "match": True,
        },
        "frozen_files": hashes,
        "mapping_configuration": {
            "mapping_bank_manifest_sha256": hashes["mapping_bank_manifest"][
                "actual_sha256"
            ],
            "formal_seeds": [2026, 2027, 2028, 2029, 2030],
            "ideal_seed": 0,
            "all_hashes_match": True,
        },
        "all_frozen_bindings_match": True,
    }


def _session_paths(session_id: str) -> dict[str, Path]:
    base = SESSION_ROOT / session_id
    return {
        "base": base,
        "session": base / "session.json",
        "snapshot": base / "frozen_snapshot.json",
        "ledger": base / "access_ledger.json",
        "completed_manifest": base / "completed_runs.jsonl",
        "checkpoint": base / "checkpoint.json",
    }


def _event(
    *,
    sequence: int,
    event_type: str,
    timestamp_utc: str,
    previous_event_sha256: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "sequence": sequence,
        "event_type": event_type,
        "timestamp_utc": timestamp_utc,
        "previous_event_sha256": previous_event_sha256,
        "payload": dict(payload),
    }
    value["event_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


def _validate_event_chain(events: Sequence[Mapping[str, Any]]) -> str | None:
    previous: str | None = None
    for index, item in enumerate(events):
        if item.get("sequence") != index:
            raise Stage10BSessionError("access-ledger event sequence is invalid")
        if item.get("previous_event_sha256") != previous:
            raise Stage10BSessionError("access-ledger event chain is broken")
        unsigned = {key: value for key, value in item.items() if key != "event_sha256"}
        actual = sha256_bytes(canonical_json_bytes(unsigned))
        if item.get("event_sha256") != actual:
            raise Stage10BSessionError("access-ledger event hash is invalid")
        previous = actual
    return previous


def _make_session_id(created_utc: str, nonce: str | None = None) -> str:
    timestamp = datetime.fromisoformat(created_utc.replace("Z", "+00:00")).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    suffix = nonce or secrets.token_hex(4)
    if not re.fullmatch(r"[0-9a-f]{8}", suffix):
        raise Stage10BSessionError("session nonce must contain eight lowercase hex digits")
    return (
        f"stage10b_restart_{timestamp}_{EXPECTED_RUN_PLAN_SHA256[:8]}_{suffix}"
    )


def _load_plan(
    project_root: Path,
    *,
    require_clean_output_paths: bool = False,
) -> dict[str, Any]:
    plan = _read_json(project_root / RUN_PLAN_PATH)
    actual_sha = _verify_sidecar(
        project_root / RUN_PLAN_PATH,
        project_root / RUN_PLAN_SHA_PATH,
    )
    if actual_sha != EXPECTED_RUN_PLAN_SHA256:
        raise Stage10BSessionError("Stage-10 run-plan SHA changed")
    if require_clean_output_paths:
        validate_run_plan(plan, project_root)
    else:
        conditions = plan.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 80:
            raise Stage10BSessionError("frozen run plan must contain 80 conditions")
    return plan


def initialize_restart_session(
    project_root: str | Path,
    *,
    created_utc: str | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Create a new zero-access session and initial 0/80 checkpoint."""

    root = Path(project_root).resolve()
    active_pointer = root / ACTIVE_SESSION_POINTER
    if active_pointer.exists():
        raise Stage10BSessionError("an active formal session already exists")
    session_parent = root / SESSION_ROOT
    if session_parent.exists() and any(session_parent.iterdir()):
        raise Stage10BSessionError(
            "session directory is non-empty without an active-session pointer"
        )

    frozen = validate_frozen_state(root)
    analysis_sha = ensure_analysis_plan(root)
    timestamp = created_utc or utc_now()
    session_id = _make_session_id(timestamp, nonce)
    paths = _session_paths(session_id)
    absolute = {name: root / path for name, path in paths.items()}
    if absolute["base"].exists():
        raise Stage10BSessionError("generated formal session ID already exists")
    absolute["base"].mkdir(parents=True, exist_ok=False)

    snapshot_document = {
        **frozen,
        "formal_session_id": session_id,
        "verified_at_utc": timestamp,
    }
    write_json_atomic(absolute["snapshot"], snapshot_document)
    snapshot_sha = write_sha_sidecar(absolute["snapshot"])

    plan = _load_plan(root, require_clean_output_paths=True)
    conditions = plan["conditions"]
    run_ids = [str(item["run_id"]) for item in conditions]
    session_document = {
        "schema_version": 1,
        "formal_session_id": session_id,
        "benchmark_id": BENCHMARK_ID,
        "session_kind": "stage10b_restart_after_unrecoverable_prior_session",
        "status": "PREPARED_AWAITING_USER_CONFIRMATION_NO_CONDITIONS_RUN",
        "created_utc": timestamp,
        "run_plan": {
            "path": RUN_PLAN_PATH.as_posix(),
            "sha256": EXPECTED_RUN_PLAN_SHA256,
            "condition_count": 80,
        },
        "analysis_plan": {
            "path": ANALYSIS_PLAN_PATH.as_posix(),
            "sha256": analysis_sha,
            "frozen_before_formal_predictions": True,
        },
        "frozen_snapshot": {
            "path": paths["snapshot"].as_posix(),
            "sha256": snapshot_sha,
        },
        "prior_session_disposition": {
            "status": "UNRECOVERABLE_NOT_IMPORTED",
            "reported_progress": "55/80",
            "imported_completed_runs": 0,
            "prior_outputs_reused": False,
            "prior_metrics_reused": False,
        },
        "planned_run_ids": run_ids,
        "execution_policy": {
            "run_order": "exact_frozen_run_plan_order",
            "resume_only_same_session": True,
            "session_reset_forbidden_after_first_access": True,
            "scientific_code_change_requires_abort": True,
            "cross_method_summary_before_80_complete": False,
            "condition_executed_during_preparation": False,
            "locked_test_images_array_read_during_preparation": False,
        },
        "paths": {
            "access_ledger": paths["ledger"].as_posix(),
            "completed_run_manifest": paths["completed_manifest"].as_posix(),
            "checkpoint": paths["checkpoint"].as_posix(),
            "formal_runs_root": FORMAL_RUNS_ROOT.as_posix(),
            "staging_root": (WORK_ROOT / session_id).as_posix(),
        },
        "environment": _environment_snapshot(root),
    }
    write_json_atomic(absolute["session"], session_document)
    session_sha = write_sha_sidecar(absolute["session"])

    prepared_event = _event(
        sequence=0,
        event_type="SESSION_PREPARED_NO_PERFORMANCE_ACCESS",
        timestamp_utc=timestamp,
        previous_event_sha256=None,
        payload={
            "formal_session_id": session_id,
            "run_plan_sha256": EXPECTED_RUN_PLAN_SHA256,
            "analysis_plan_sha256": analysis_sha,
            "condition_executed": False,
            "locked_test_images_array_read": False,
        },
    )
    ledger = {
        "schema_version": 1,
        "formal_session_id": session_id,
        "benchmark_id": BENCHMARK_ID,
        "status": "PREPARED_NO_FORMAL_PERFORMANCE_ACCESS",
        "created_utc": timestamp,
        "run_plan_sha256": EXPECTED_RUN_PLAN_SHA256,
        "analysis_plan_sha256": analysis_sha,
        "session_record_sha256": session_sha,
        "performance_access_committed": False,
        "locked_test_performance_access_count": 0,
        "formal_mapping_performance_access_count": 0,
        "formal_condition_started_count": 0,
        "formal_condition_completed_count": 0,
        "formal_mapping_seeds": {
            str(seed): {"performance_access_count": 0, "run_ids": []}
            for seed in (2026, 2027, 2028, 2029, 2030)
        },
        "locked_test_fonts": {
            "long_cang": {"performance_access_count": 0},
            "zhi_mang_xing": {"performance_access_count": 0},
        },
        "events": [prepared_event],
        "event_chain_head_sha256": prepared_event["event_sha256"],
    }
    write_json_atomic(absolute["ledger"], ledger)
    write_sha_sidecar(absolute["ledger"])

    _write_bytes_atomic(absolute["completed_manifest"], b"")
    completed_manifest_sha = sha256_file(absolute["completed_manifest"])
    checkpoint = {
        "schema_version": 1,
        "formal_session_id": session_id,
        "benchmark_id": BENCHMARK_ID,
        "status": "READY_NOT_STARTED_AWAITING_CONFIRMATION",
        "updated_utc": timestamp,
        "checkpoint_generation": 0,
        "run_plan_sha256": EXPECTED_RUN_PLAN_SHA256,
        "session_record_sha256": session_sha,
        "total_condition_count": 80,
        "completed_condition_count": 0,
        "completed_run_ids": [],
        "next_condition_index": 0,
        "next_run_id": run_ids[0],
        "completed_run_manifest": {
            "path": paths["completed_manifest"].as_posix(),
            "sha256": completed_manifest_sha,
            "size_bytes": 0,
            "record_count": 0,
        },
        "safe_to_resume_after_confirmation": True,
        "condition_executed_during_preparation": False,
    }
    write_json_atomic(absolute["checkpoint"], checkpoint)
    write_sha_sidecar(absolute["checkpoint"])

    pointer = {
        "schema_version": 1,
        "formal_session_id": session_id,
        "session_record_path": paths["session"].as_posix(),
        "session_record_sha256": session_sha,
        "created_utc": timestamp,
        "status": "ACTIVE_PREPARED_AWAITING_CONFIRMATION",
    }
    write_json_atomic(active_pointer, pointer)
    write_sha_sidecar(active_pointer)
    return validate_restart_preparation(root)


def _active_session(project_root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    pointer_path = project_root / ACTIVE_SESSION_POINTER
    _verify_sidecar(pointer_path)
    pointer = _read_json(pointer_path)
    session_id = pointer.get("formal_session_id")
    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
        raise Stage10BSessionError("active formal session ID is invalid")
    paths = _session_paths(session_id)
    session_path = project_root / paths["session"]
    session_sha = _verify_sidecar(session_path)
    if pointer.get("session_record_sha256") != session_sha:
        raise Stage10BSessionError("active-session pointer does not bind session record")
    return _read_json(session_path), paths


def _read_completed_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Stage10BSessionError("completed-run manifest is unreadable") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise Stage10BSessionError(
                f"blank line in completed-run manifest at {line_number}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Stage10BSessionError(
                f"invalid completed-run record at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise Stage10BSessionError("completed-run record must be an object")
        unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
        expected = sha256_bytes(canonical_json_bytes(unsigned))
        if value.get("record_sha256") != expected:
            raise Stage10BSessionError("completed-run record hash is invalid")
        records.append(value)
    return records


def _verify_output_manifest(project_root: Path, record: Mapping[str, Any]) -> None:
    output_path = project_root / str(record.get("output_path", ""))
    if not output_path.is_dir():
        raise Stage10BSessionError(f"completed output directory is missing: {output_path}")
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise Stage10BSessionError("completed-run file manifest is empty")
    for item in files:
        if not isinstance(item, dict):
            raise Stage10BSessionError("completed-run file entry is invalid")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise Stage10BSessionError("completed-run file path escapes output root")
        path = output_path / relative
        if not path.is_file():
            raise Stage10BSessionError(f"completed-run file is missing: {path}")
        if path.stat().st_size != item.get("size_bytes"):
            raise Stage10BSessionError(f"completed-run file size changed: {path}")
        if sha256_file(path) != item.get("sha256"):
            raise Stage10BSessionError(f"completed-run file hash changed: {path}")


def load_resume_state(project_root: str | Path) -> ResumeState:
    """Validate the active session and return the exact next frozen run."""

    root = Path(project_root).resolve()
    session, paths = _active_session(root)
    if session.get("run_plan", {}).get("sha256") != EXPECTED_RUN_PLAN_SHA256:
        raise Stage10BSessionError("session run-plan binding changed")
    # Completed output directories are expected after the first condition.
    # Their exact path, size, and SHA-256 bindings are verified below against
    # completed_runs.jsonl; only restart preparation requires an empty root.
    validate_frozen_state(root, require_clean_formal_runs=False)
    if _verify_sidecar(root / ANALYSIS_PLAN_PATH, root / ANALYSIS_PLAN_SHA_PATH) != session.get(
        "analysis_plan", {}
    ).get("sha256"):
        raise Stage10BSessionError("session analysis-plan binding changed")

    snapshot_path = root / paths["snapshot"]
    snapshot = _read_json(snapshot_path)
    _verify_sidecar(snapshot_path)
    if not snapshot.get("all_frozen_bindings_match"):
        raise Stage10BSessionError("saved frozen snapshot is not a PASS snapshot")

    ledger_path = root / paths["ledger"]
    ledger = _read_json(ledger_path)
    _verify_sidecar(ledger_path)
    events = ledger.get("events")
    if not isinstance(events, list):
        raise Stage10BSessionError("access-ledger events are invalid")
    chain_head = _validate_event_chain(events)
    if ledger.get("event_chain_head_sha256") != chain_head:
        raise Stage10BSessionError("access-ledger chain head is invalid")

    manifest_path = root / paths["completed_manifest"]
    records = _read_completed_records(manifest_path)
    plan = _load_plan(root)
    run_ids = [str(item["run_id"]) for item in plan["conditions"]]
    completed_ids = [str(item.get("run_id")) for item in records]
    if completed_ids != run_ids[: len(completed_ids)]:
        raise Stage10BSessionError("completed runs are not an exact run-plan prefix")
    if len(set(completed_ids)) != len(completed_ids):
        raise Stage10BSessionError("completed-run manifest contains duplicates")
    for record in records:
        _verify_output_manifest(root, record)

    checkpoint_path = root / paths["checkpoint"]
    checkpoint = _read_json(checkpoint_path)
    _verify_sidecar(checkpoint_path)
    manifest_sha = sha256_file(manifest_path)
    manifest_state = checkpoint.get("completed_run_manifest", {})
    if manifest_state.get("sha256") != manifest_sha:
        raise Stage10BSessionError("checkpoint does not bind completed-run manifest")
    if manifest_state.get("record_count") != len(records):
        raise Stage10BSessionError("checkpoint completed-record count changed")
    if checkpoint.get("completed_run_ids") != completed_ids:
        raise Stage10BSessionError("checkpoint completed-run list changed")
    if checkpoint.get("completed_condition_count") != len(records):
        raise Stage10BSessionError("checkpoint completed count changed")
    next_id = run_ids[len(records)] if len(records) < len(run_ids) else None
    if checkpoint.get("next_run_id") != next_id:
        raise Stage10BSessionError("checkpoint next run does not match run plan")

    expected_completed_outputs = {
        str(item["planned_output_path"])
        for item in plan["conditions"][: len(records)]
    }
    formal_root = root / FORMAL_RUNS_ROOT
    actual_outputs = set()
    if formal_root.exists():
        actual_outputs = {
            path.relative_to(root).as_posix()
            for path in formal_root.iterdir()
            if path.is_dir()
        }
    unexpected = actual_outputs - expected_completed_outputs
    if unexpected:
        raise Stage10BSessionError(
            f"uncommitted or extra formal output directories exist: {sorted(unexpected)}"
        )
    return ResumeState(
        formal_session_id=str(session["formal_session_id"]),
        completed_count=len(records),
        total_count=len(run_ids),
        next_run_id=next_id,
        completed_run_ids=tuple(completed_ids),
        remaining_run_ids=tuple(run_ids[len(records) :]),
        safe_to_resume=True,
        status=str(checkpoint["status"]),
    )


def validate_restart_preparation(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    state = load_resume_state(root)
    session, paths = _active_session(root)
    ledger = _read_json(root / paths["ledger"])
    checkpoint = _read_json(root / paths["checkpoint"])
    if state.completed_count != 0 or state.total_count != 80:
        raise Stage10BSessionError("restart preparation is not at 0/80")
    zero_fields = (
        "locked_test_performance_access_count",
        "formal_mapping_performance_access_count",
        "formal_condition_started_count",
        "formal_condition_completed_count",
    )
    if any(ledger.get(name) != 0 for name in zero_fields):
        raise Stage10BSessionError("restart-preparation access ledger is nonzero")
    if ledger.get("performance_access_committed") is not False:
        raise Stage10BSessionError("formal performance access was already committed")
    if checkpoint.get("condition_executed_during_preparation") is not False:
        raise Stage10BSessionError("checkpoint reports a preparation-time condition")
    return {
        "formal_session_id": state.formal_session_id,
        "status": session["status"],
        "run_plan_sha256": EXPECTED_RUN_PLAN_SHA256,
        "analysis_plan_sha256": session["analysis_plan"]["sha256"],
        "frozen_snapshot_sha256": session["frozen_snapshot"]["sha256"],
        "completed_condition_count": 0,
        "remaining_condition_count": 80,
        "next_run_id": state.next_run_id,
        "access_counts": {name: ledger[name] for name in zero_fields},
        "performance_access_committed": False,
        "locked_test_images_array_read": False,
        "formal_runs_root_state": (
            "absent" if not (root / FORMAL_RUNS_ROOT).exists() else "empty"
        ),
        "safe_to_start_after_user_confirmation": True,
    }


def _append_ledger_event(
    project_root: Path,
    paths: Mapping[str, Path],
    *,
    event_type: str,
    payload: Mapping[str, Any],
    counter_updates: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    ledger_path = project_root / paths["ledger"]
    ledger = _read_json(ledger_path)
    _verify_sidecar(ledger_path)
    events = ledger.get("events")
    if not isinstance(events, list):
        raise Stage10BSessionError("access-ledger events are invalid")
    previous = _validate_event_chain(events)
    event = _event(
        sequence=len(events),
        event_type=event_type,
        timestamp_utc=utc_now(),
        previous_event_sha256=previous,
        payload=payload,
    )
    events.append(event)
    ledger["events"] = events
    ledger["event_chain_head_sha256"] = event["event_sha256"]
    for key, value in (counter_updates or {}).items():
        ledger[key] = int(ledger.get(key, 0)) + int(value)
    write_json_atomic(ledger_path, ledger)
    write_sha_sidecar(ledger_path)
    return ledger


def create_run_staging_dir(project_root: str | Path, run_id: str) -> Path:
    """Reserve the next run's staging directory after a resume audit."""

    root = Path(project_root).resolve()
    state = load_resume_state(root)
    if run_id != state.next_run_id:
        raise Stage10BSessionError("only the exact next frozen run may be staged")
    path = root / WORK_ROOT / state.formal_session_id / f"{run_id}.partial"
    path.mkdir(parents=True, exist_ok=False)
    return path


def record_formal_run_started(project_root: str | Path, run_id: str) -> None:
    """Commit first formal access and append an irreversible run-start event."""

    root = Path(project_root).resolve()
    state = load_resume_state(root)
    if run_id != state.next_run_id:
        raise Stage10BSessionError("run-start ID is not the next frozen condition")
    session, paths = _active_session(root)
    plan = _load_plan(root)
    condition = plan["conditions"][state.completed_count]
    ledger = _read_json(root / paths["ledger"])
    first_access = not bool(ledger.get("performance_access_committed"))
    counter_updates = {"formal_condition_started_count": 1}
    if first_access:
        counter_updates.update(
            {
                "locked_test_performance_access_count": 1,
                "formal_mapping_performance_access_count": 1,
            }
        )
    ledger = _append_ledger_event(
        root,
        paths,
        event_type="FORMAL_RUN_STARTED",
        payload={
            "formal_session_id": session["formal_session_id"],
            "run_id": run_id,
            "method": condition["method"],
            "budget": condition["budget"],
            "mapping_type": condition["mapping_type"],
            "mapping_seed": condition["mapping_seed"],
            "first_formal_performance_access": first_access,
        },
        counter_updates=counter_updates,
    )
    if first_access:
        ledger["performance_access_committed"] = True
        ledger["status"] = "FORMAL_PERFORMANCE_ACCESS_COMMITTED"
        for font_id in ("long_cang", "zhi_mang_xing"):
            ledger["locked_test_fonts"][font_id]["performance_access_count"] = 1
    if condition["mapping_type"] == "moderate":
        seed = str(condition["mapping_seed"])
        seed_state = ledger["formal_mapping_seeds"][seed]
        if seed_state["performance_access_count"] == 0:
            seed_state["performance_access_count"] = 1
        seed_state["run_ids"].append(run_id)
    write_json_atomic(root / paths["ledger"], ledger)
    write_sha_sidecar(root / paths["ledger"])


def _validate_prediction_file(path: Path, condition: Mapping[str, Any]) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REQUIRED_PREDICTION_COLUMNS:
                raise Stage10BSessionError("predictions.csv columns changed")
            rows = list(reader)
    except OSError as exc:
        raise Stage10BSessionError("predictions.csv is unreadable") from exc
    if len(rows) != 360:
        raise Stage10BSessionError("predictions.csv must contain exactly 360 rows")
    identities = {(row["char_id"], row["font_id"]) for row in rows}
    if len(identities) != 360:
        raise Stage10BSessionError("predictions.csv contains missing or duplicate samples")
    expected_constants = {
        "benchmark_id": BENCHMARK_ID,
        "run_id": str(condition["run_id"]),
        "method": str(condition["method"]),
        "budget": str(condition["budget"]),
        "mapping_type": str(condition["mapping_type"]),
        "mapping_seed": str(condition["mapping_seed"]),
    }
    for row in rows:
        if any(row[key] != value for key, value in expected_constants.items()):
            raise Stage10BSessionError("prediction row condition binding changed")
        if row["font_id"] not in {"long_cang", "zhi_mang_xing"}:
            raise Stage10BSessionError("prediction row contains a non-locked-test font")
        if row["correct"] not in {"0", "1", "False", "True", "false", "true"}:
            raise Stage10BSessionError("prediction row has an invalid correct flag")


def _file_manifest(directory: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(directory).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def commit_completed_run(project_root: str | Path, run_id: str) -> dict[str, Any]:
    """Atomically promote the next complete staged run and advance checkpoint."""

    root = Path(project_root).resolve()
    state = load_resume_state(root)
    if run_id != state.next_run_id:
        raise Stage10BSessionError("only the next frozen run may be committed")
    session, paths = _active_session(root)
    plan = _load_plan(root)
    condition = plan["conditions"][state.completed_count]
    staging = root / WORK_ROOT / state.formal_session_id / f"{run_id}.partial"
    if not staging.is_dir():
        raise Stage10BSessionError("run staging directory is missing")
    missing = [name for name in REQUIRED_COMPLETED_RUN_FILES if not (staging / name).is_file()]
    if missing:
        raise Stage10BSessionError(f"run output is incomplete: missing {missing}")
    _validate_prediction_file(staging / "predictions.csv", condition)
    metrics = _read_json(staging / "metrics.json")
    manifest = _read_json(staging / "run_manifest.json")
    required_bindings = {
        "benchmark_id": BENCHMARK_ID,
        "formal_session_id": state.formal_session_id,
        "run_id": run_id,
        "run_plan_sha256": EXPECTED_RUN_PLAN_SHA256,
        "method": condition["method"],
        "budget": condition["budget"],
        "mapping_type": condition["mapping_type"],
        "mapping_seed": condition["mapping_seed"],
    }
    if any(manifest.get(key) != value for key, value in required_bindings.items()):
        raise Stage10BSessionError("run_manifest.json does not bind the frozen condition")
    if metrics.get("train_sample_count") != 1080 or metrics.get("test_sample_count") != 360:
        raise Stage10BSessionError("metrics sample counts changed")

    files = _file_manifest(staging)
    output_relative = Path(str(condition["planned_output_path"]))
    final_output = root / output_relative
    final_output.parent.mkdir(parents=True, exist_ok=True)
    if final_output.exists():
        raise Stage10BSessionError("formal output path already exists")
    os.replace(staging, final_output)

    completed_at = utc_now()
    unsigned_record = {
        "record_index": state.completed_count,
        "formal_session_id": state.formal_session_id,
        "run_id": run_id,
        "completed_at_utc": completed_at,
        "output_path": output_relative.as_posix(),
        "files": files,
    }
    record = {
        **unsigned_record,
        "record_sha256": sha256_bytes(canonical_json_bytes(unsigned_record)),
    }
    manifest_path = root / paths["completed_manifest"]
    existing = manifest_path.read_bytes()
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(
        "utf-8"
    ) + b"\n"
    _write_bytes_atomic(manifest_path, existing + line)

    completed_ids = [*state.completed_run_ids, run_id]
    next_index = len(completed_ids)
    next_run_id = (
        str(plan["conditions"][next_index]["run_id"])
        if next_index < len(plan["conditions"])
        else None
    )
    checkpoint_path = root / paths["checkpoint"]
    old_checkpoint = _read_json(checkpoint_path)
    checkpoint = {
        **old_checkpoint,
        "status": "COMPLETE_80_OF_80" if next_run_id is None else "IN_PROGRESS",
        "updated_utc": completed_at,
        "checkpoint_generation": int(old_checkpoint["checkpoint_generation"]) + 1,
        "completed_condition_count": len(completed_ids),
        "completed_run_ids": completed_ids,
        "next_condition_index": next_index,
        "next_run_id": next_run_id,
        "completed_run_manifest": {
            "path": paths["completed_manifest"].as_posix(),
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "record_count": len(completed_ids),
        },
    }
    write_json_atomic(checkpoint_path, checkpoint)
    write_sha_sidecar(checkpoint_path)
    _append_ledger_event(
        root,
        paths,
        event_type="FORMAL_RUN_COMPLETED_AND_CHECKPOINTED",
        payload={
            "run_id": run_id,
            "record_sha256": record["record_sha256"],
            "completed_condition_count": len(completed_ids),
        },
        counter_updates={"formal_condition_completed_count": 1},
    )
    return record
