"""Frozen protocol and zero-of-fifty preflight for MMA multi-budget Study B.

This module is deliberately performance-blind.  It builds and validates the
scientific run plan, but it neither imports a classifier nor reads predictions.
The formal prediction runner and the post-50/50 analysis release are separate
future stages.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class MultiBudgetProtocolError(ValueError):
    """Raised before an invalid multi-budget protocol can be frozen or used."""


PROTOCOL_SCHEMA = "mma-multibudget-v1-protocol"
RUN_PLAN_SCHEMA = "mma-multibudget-v1-run-plan"
PREFLIGHT_SCHEMA = "mma-multibudget-v1-preflight"
SCHEMA_VERSION = 1

REQUIRED_METHODS = ("ideal_only", "ideal_plus_multi4")
REQUIRED_BUDGETS = (32, 48, 64, 96, 128)
REQUIRED_MAPPING_SEEDS = (4101, 4102, 4103, 4104, 4105)
REQUIRED_CONDITION = "moderate"
REQUIRED_CLASS_COUNT = 180
REQUIRED_FONT_FAMILY_COUNT = 4
REQUIRED_RUN_COUNT = 50
REQUIRED_RUN_ORDER = "budget_seed_method"

RUN_PLAN_STATUS = "FROZEN_50_CONDITION_RUN_PLAN_NO_FORMAL_PREDICTIONS"
PREFLIGHT_STATUS = (
    "PASS_PROTOCOL_PREFLIGHT_ZERO_OF_50_AWAITING_CONFIRMATION_ASSETS"
)
ANALYSIS_RELEASE_STATUS = "ANALYSIS_RELEASED"
ASSET_STATUS = "PENDING_BLINDED_GENERATION"


@dataclass(frozen=True, slots=True)
class MultiBudgetProtocol:
    """Validated scientific protocol and project-relative artifact locations."""

    source_path: Path
    project_root: Path
    study_id: str
    condition: str
    methods: tuple[str, ...]
    budgets: tuple[int, ...]
    mapping_seeds: tuple[int, ...]
    run_order: str
    class_count: int
    confirmation_font_family_count: int
    method_training_mappings: Mapping[str, tuple[str, ...]]
    encoder: Mapping[str, Any]
    feature: Mapping[str, Any]
    classifier: Mapping[str, Any]
    analysis: Mapping[str, Any]
    asset_manifest_path: Path
    asset_status: str
    run_plan_path: Path
    run_plan_sha256_path: Path
    preflight_report_path: Path
    preflight_report_sha256_path: Path
    formal_runs_root: Path

    @property
    def expected_samples_per_condition(self) -> int:
        return self.class_count * self.confirmation_font_family_count

    @property
    def expected_conditions(self) -> int:
        return len(self.methods) * len(self.budgets) * len(self.mapping_seeds)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used by every digest."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise MultiBudgetProtocolError(f"required file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MultiBudgetProtocolError(f"{name} must be a mapping")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MultiBudgetProtocolError(f"{name} must be a non-empty list")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise MultiBudgetProtocolError(f"{name} contains empty or duplicate values")
    return result


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise MultiBudgetProtocolError(f"{name} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise MultiBudgetProtocolError(f"{name} must contain integers only")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise MultiBudgetProtocolError(f"{name} contains duplicate values")
    return result


def _safe_token(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result or not all(character.isalnum() or character in "-_" for character in result):
        raise MultiBudgetProtocolError(f"{name} must be a filesystem-safe token")
    return result


def _project_path(project_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MultiBudgetProtocolError(f"{name} must be a non-empty relative path")
    candidate = Path(value.strip())
    if candidate.is_absolute():
        raise MultiBudgetProtocolError(f"{name} must be project-relative")
    resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise MultiBudgetProtocolError(f"{name} escapes the project root") from error
    return resolved


def _exact(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise MultiBudgetProtocolError(
            f"{name} differs from the frozen design: {actual!r} != {expected!r}"
        )


def load_protocol(path: str | Path) -> MultiBudgetProtocol:
    """Load and strictly validate the frozen 2 x 5 x 5 scientific design."""

    source = Path(path).resolve()
    if not source.is_file():
        raise MultiBudgetProtocolError(f"protocol config does not exist: {source}")
    try:
        document = _mapping(
            yaml.safe_load(source.read_text(encoding="utf-8")), "protocol config"
        )
    except yaml.YAMLError as error:
        raise MultiBudgetProtocolError(f"invalid protocol YAML: {source}") from error
    _exact(document.get("schema"), PROTOCOL_SCHEMA, "schema")
    _exact(document.get("schema_version"), SCHEMA_VERSION, "schema_version")

    project_root = source.parent.parent.resolve()
    design = _mapping(document.get("design"), "design")
    methods = _string_tuple(design.get("methods"), "design.methods")
    budgets = _integer_tuple(design.get("budgets"), "design.budgets")
    seeds = _integer_tuple(design.get("mapping_seeds"), "design.mapping_seeds")
    condition = str(design.get("condition", "")).strip()
    run_order = str(design.get("run_order", "")).strip()
    class_count = int(design.get("class_count", -1))
    font_count = int(design.get("confirmation_font_family_count", -1))

    _exact(methods, REQUIRED_METHODS, "design.methods")
    _exact(budgets, REQUIRED_BUDGETS, "design.budgets")
    _exact(seeds, REQUIRED_MAPPING_SEEDS, "design.mapping_seeds")
    _exact(condition, REQUIRED_CONDITION, "design.condition")
    _exact(run_order, REQUIRED_RUN_ORDER, "design.run_order")
    _exact(class_count, REQUIRED_CLASS_COUNT, "design.class_count")
    _exact(
        font_count,
        REQUIRED_FONT_FAMILY_COUNT,
        "design.confirmation_font_family_count",
    )
    _exact(int(design.get("expected_conditions", -1)), REQUIRED_RUN_COUNT, "design.expected_conditions")

    raw_training = _mapping(
        design.get("method_training_mappings"), "design.method_training_mappings"
    )
    training = {
        method: _string_tuple(
            raw_training.get(method), f"design.method_training_mappings.{method}"
        )
        for method in methods
    }
    if set(raw_training) != set(methods):
        raise MultiBudgetProtocolError(
            "method_training_mappings must contain exactly the two frozen methods"
        )
    _exact(training["ideal_only"], ("ideal",), "ideal_only training mappings")
    _exact(
        training["ideal_plus_multi4"],
        (
            "ideal",
            "moderate_seed2026",
            "moderate_seed2027",
            "moderate_seed2028",
            "moderate_seed2029",
        ),
        "ideal_plus_multi4 training mappings",
    )

    encoder = dict(_mapping(document.get("encoder"), "encoder"))
    _exact(encoder.get("name"), "skeleton_topk", "encoder.name")
    _exact(encoder.get("selection_mapping"), "ideal", "encoder.selection_mapping")
    _exact(
        encoder.get("budget_contract"),
        "common_deterministic_ranking_prefix",
        "encoder.budget_contract",
    )
    _exact(encoder.get("tie_break"), "candidate_index_ascending", "encoder.tie_break")

    feature = dict(_mapping(document.get("feature"), "feature"))
    _exact(feature.get("name"), "hog", "feature.name")
    _exact(feature.get("orientations"), 9, "feature.orientations")
    _exact(feature.get("pixels_per_cell"), [8, 8], "feature.pixels_per_cell")
    _exact(feature.get("cells_per_block"), [2, 2], "feature.cells_per_block")
    _exact(feature.get("block_norm"), "L2-Hys", "feature.block_norm")

    classifier = dict(_mapping(document.get("classifier"), "classifier"))
    _exact(
        classifier.get("name"),
        "multimapping_cosine_prototype",
        "classifier.name",
    )
    _exact(classifier.get("parameter_search"), False, "classifier.parameter_search")

    assets = _mapping(document.get("confirmation_assets"), "confirmation_assets")
    _exact(assets.get("status"), ASSET_STATUS, "confirmation_assets.status")
    _exact(
        int(assets.get("required_font_family_count", -1)),
        REQUIRED_FONT_FAMILY_COUNT,
        "confirmation_assets.required_font_family_count",
    )
    _exact(
        tuple(assets.get("required_mapping_seeds", [])),
        REQUIRED_MAPPING_SEEDS,
        "confirmation_assets.required_mapping_seeds",
    )
    if assets.get("font_family_ids") != []:
        raise MultiBudgetProtocolError(
            "future confirmation font IDs must remain empty until blinded asset freeze"
        )

    analysis = dict(_mapping(document.get("analysis"), "analysis"))
    _exact(
        analysis.get("release_status_required"),
        ANALYSIS_RELEASE_STATUS,
        "analysis.release_status_required",
    )
    _exact(
        analysis.get("release_completed_conditions"),
        REQUIRED_RUN_COUNT,
        "analysis.release_completed_conditions",
    )
    _exact(
        analysis.get("primary_endpoint"),
        "delta_normalized_area_under_accuracy_budget_curve",
        "analysis.primary_endpoint",
    )
    _exact(
        analysis.get("bootstrap_unit"),
        "character_cluster_retaining_all_paired_cells",
        "analysis.bootstrap_unit",
    )
    _exact(analysis.get("bootstrap_iterations"), 100_000, "analysis.bootstrap_iterations")

    outputs = _mapping(document.get("outputs"), "outputs")
    protocol = MultiBudgetProtocol(
        source_path=source,
        project_root=project_root,
        study_id=_safe_token(document.get("study_id"), "study_id"),
        condition=condition,
        methods=methods,
        budgets=budgets,
        mapping_seeds=seeds,
        run_order=run_order,
        class_count=class_count,
        confirmation_font_family_count=font_count,
        method_training_mappings=training,
        encoder=encoder,
        feature=feature,
        classifier=classifier,
        analysis=analysis,
        asset_manifest_path=_project_path(
            project_root,
            assets.get("future_manifest_path"),
            "confirmation_assets.future_manifest_path",
        ),
        asset_status=str(assets["status"]),
        run_plan_path=_project_path(
            project_root, outputs.get("run_plan"), "outputs.run_plan"
        ),
        run_plan_sha256_path=_project_path(
            project_root,
            outputs.get("run_plan_sha256"),
            "outputs.run_plan_sha256",
        ),
        preflight_report_path=_project_path(
            project_root,
            outputs.get("preflight_report"),
            "outputs.preflight_report",
        ),
        preflight_report_sha256_path=_project_path(
            project_root,
            outputs.get("preflight_report_sha256"),
            "outputs.preflight_report_sha256",
        ),
        formal_runs_root=_project_path(
            project_root, outputs.get("formal_runs_root"), "outputs.formal_runs_root"
        ),
    )
    if protocol.expected_conditions != REQUIRED_RUN_COUNT:
        raise MultiBudgetProtocolError("protocol does not expand to exactly 50 conditions")
    if protocol.expected_samples_per_condition != 720:
        raise MultiBudgetProtocolError("each condition must contain 4 fonts x 180 classes")
    artifact_paths = (
        protocol.run_plan_path,
        protocol.run_plan_sha256_path,
        protocol.preflight_report_path,
        protocol.preflight_report_sha256_path,
        protocol.formal_runs_root,
        protocol.asset_manifest_path,
    )
    if len(set(artifact_paths)) != len(artifact_paths):
        raise MultiBudgetProtocolError("protocol artifact paths must be distinct")
    return protocol


def _relative(protocol: MultiBudgetProtocol, path: Path) -> str:
    return path.relative_to(protocol.project_root).as_posix()


def _run_id(method: str, budget: int, seed: int) -> str:
    return f"mbv1_moderate_b{budget:03d}_s{seed}_{method}"


def build_run_plan(protocol: MultiBudgetProtocol) -> dict[str, Any]:
    """Build the deterministic budget-major, seed-major, method-major plan."""

    conditions: list[dict[str, Any]] = []
    ordinal = 1
    formal_root = _relative(protocol, protocol.formal_runs_root)
    for budget in protocol.budgets:
        for seed in protocol.mapping_seeds:
            for method in protocol.methods:
                run_id = _run_id(method, budget, seed)
                conditions.append(
                    {
                        "ordinal": ordinal,
                        "run_id": run_id,
                        "method": method,
                        "training_mapping_ids": list(
                            protocol.method_training_mappings[method]
                        ),
                        "budget": budget,
                        "mapping_condition": protocol.condition,
                        "mapping_seed": seed,
                        "encoder": "skeleton_topk",
                        "selection_prefix_length": budget,
                        "expected_font_families": (
                            protocol.confirmation_font_family_count
                        ),
                        "expected_classes": protocol.class_count,
                        "expected_prediction_rows": (
                            protocol.expected_samples_per_condition
                        ),
                        "output_dir": f"{formal_root}/{run_id}",
                        "prediction_only_before_analysis_release": True,
                        "per_condition_metrics_allowed": False,
                    }
                )
                ordinal += 1
    run_ids = [item["run_id"] for item in conditions]
    if len(conditions) != REQUIRED_RUN_COUNT or len(set(run_ids)) != REQUIRED_RUN_COUNT:
        raise MultiBudgetProtocolError("run-plan expansion is not exactly 50 unique IDs")
    return {
        "schema": RUN_PLAN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": RUN_PLAN_STATUS,
        "study_id": protocol.study_id,
        "protocol_config": _relative(protocol, protocol.source_path),
        "protocol_config_sha256": sha256_file(protocol.source_path),
        "design": {
            "condition": protocol.condition,
            "methods": list(protocol.methods),
            "budgets": list(protocol.budgets),
            "mapping_seeds": list(protocol.mapping_seeds),
            "run_order": protocol.run_order,
            "expected_conditions": protocol.expected_conditions,
            "expected_prediction_rows_per_condition": (
                protocol.expected_samples_per_condition
            ),
            "expected_prediction_rows_total": (
                protocol.expected_conditions
                * protocol.expected_samples_per_condition
            ),
        },
        "selection_nesting_contract": {
            "ranking": "one_common_deterministic_skeleton_ranking_per_glyph",
            "prefix_budgets": list(protocol.budgets),
            "required_relation": "S32_subset_S48_subset_S64_subset_S96_subset_S128",
            "exact_unique_positions_per_budget": True,
        },
        "confirmation_assets": {
            "status": protocol.asset_status,
            "future_manifest_path": _relative(
                protocol, protocol.asset_manifest_path
            ),
            "required_font_family_count": (
                protocol.confirmation_font_family_count
            ),
            "required_mapping_seeds": list(protocol.mapping_seeds),
            "formal_asset_hashes_verified": False,
        },
        "analysis_gate": {
            "release_status_required": ANALYSIS_RELEASE_STATUS,
            "release_completed_conditions": REQUIRED_RUN_COUNT,
            "release_unexpected_conditions": 0,
            "release_all_run_hashes_verified": True,
            "release_formal_asset_hashes_verified": True,
            "aggregate_analysis_allowed_now": False,
        },
        "frozen_run_ids": run_ids,
        "conditions": conditions,
    }


def validate_run_plan(
    protocol: MultiBudgetProtocol, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Require byte-level scientific equivalence to the deterministic plan."""

    expected = build_run_plan(protocol)
    if dict(document) != expected:
        raise MultiBudgetProtocolError(
            "frozen run plan differs from the current protocol expansion"
        )
    return {
        "status": "PASS_FROZEN_RUN_PLAN_VALID",
        "expected_conditions": REQUIRED_RUN_COUNT,
        "frozen_run_ids": list(expected["frozen_run_ids"]),
    }


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise MultiBudgetProtocolError(f"refusing to overwrite frozen artifact: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise MultiBudgetProtocolError(f"{name} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MultiBudgetProtocolError(f"{name} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise MultiBudgetProtocolError(f"{name} root must be an object")
    return value


def _verify_sidecar(path: Path, sidecar: Path) -> str:
    actual = sha256_file(path)
    if not sidecar.is_file():
        raise MultiBudgetProtocolError(f"SHA-256 sidecar does not exist: {sidecar}")
    expected = sidecar.read_text(encoding="ascii").strip().lower()
    if expected != actual:
        raise MultiBudgetProtocolError(f"SHA-256 sidecar mismatch: {path}")
    return actual


def _formal_root_state(protocol: MultiBudgetProtocol) -> dict[str, Any]:
    root = protocol.formal_runs_root
    if not root.exists():
        return {"state": "ABSENT", "run_directory_count": 0, "run_ids": []}
    if not root.is_dir():
        raise MultiBudgetProtocolError("formal_runs_root exists but is not a directory")
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    if entries:
        raise MultiBudgetProtocolError(
            "formal run output exists; zero-of-fifty preflight is no longer valid"
        )
    return {"state": "EMPTY", "run_directory_count": 0, "run_ids": []}


def _build_preflight_report(
    protocol: MultiBudgetProtocol,
    plan: Mapping[str, Any],
    run_plan_sha256: str,
) -> dict[str, Any]:
    root_state = _formal_root_state(protocol)
    frozen_run_ids = list(plan["frozen_run_ids"])
    return {
        "schema": PREFLIGHT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "study_id": protocol.study_id,
        "status": PREFLIGHT_STATUS,
        "expected_conditions": REQUIRED_RUN_COUNT,
        "completed_conditions": 0,
        "unexpected_conditions": 0,
        "hashes_verified": True,
        "verified_hash_scope": ["protocol_config", "frozen_run_plan"],
        "protocol_config": {
            "path": _relative(protocol, protocol.source_path),
            "sha256": sha256_file(protocol.source_path),
        },
        "frozen_run_plan": {
            "path": _relative(protocol, protocol.run_plan_path),
            "sha256": run_plan_sha256,
            "condition_count": REQUIRED_RUN_COUNT,
        },
        "frozen_run_ids": frozen_run_ids,
        "run_ids": list(root_state["run_ids"]),
        "formal_runs_root": {
            "path": _relative(protocol, protocol.formal_runs_root),
            "state": root_state["state"],
            "run_directory_count": root_state["run_directory_count"],
        },
        "confirmation_assets": {
            "status": protocol.asset_status,
            "manifest_path": _relative(protocol, protocol.asset_manifest_path),
            "required_font_family_count": (
                protocol.confirmation_font_family_count
            ),
            "required_mapping_seeds": list(protocol.mapping_seeds),
            "formal_asset_hashes_verified": False,
        },
        "prediction_executed": False,
        "per_condition_metrics_generated": False,
        "accuracy_generated": False,
        "accuracy_read": False,
        "analysis_gate": {
            "status": "LOCKED",
            "release_status_required": ANALYSIS_RELEASE_STATUS,
            "released": False,
            "completed_conditions_required": REQUIRED_RUN_COUNT,
            "unexpected_conditions_required": 0,
            "all_run_hashes_verified_required": True,
            "formal_asset_hashes_verified_required": True,
        },
        "formal_execution_ready": False,
        "blocking_requirement": (
            "Freeze and hash four new font families and moderate mapping seeds "
            "4101-4105, then create the separate formal execution lock."
        ),
    }


def _validate_preflight_report(
    protocol: MultiBudgetProtocol,
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_plan_sha256: str,
) -> None:
    expected = _build_preflight_report(protocol, plan, run_plan_sha256)
    if dict(report) != expected:
        raise MultiBudgetProtocolError(
            "preflight report differs from the current zero-of-fifty state"
        )


def prepare_preflight(
    config_path: str | Path, *, validate_only: bool = False
) -> dict[str, Any]:
    """Freeze or read-only validate the 50-run plan and 0/50 report."""

    protocol = load_protocol(config_path)
    if validate_only:
        plan = _read_json(protocol.run_plan_path, "frozen run plan")
        validate_run_plan(protocol, plan)
        plan_sha256 = _verify_sidecar(
            protocol.run_plan_path, protocol.run_plan_sha256_path
        )
        report = _read_json(protocol.preflight_report_path, "preflight report")
        _validate_preflight_report(protocol, report, plan, plan_sha256)
        _verify_sidecar(
            protocol.preflight_report_path, protocol.preflight_report_sha256_path
        )
        return dict(report) | {"validation_mode": "READ_ONLY"}

    targets = (
        protocol.run_plan_path,
        protocol.run_plan_sha256_path,
        protocol.preflight_report_path,
        protocol.preflight_report_sha256_path,
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise MultiBudgetProtocolError(
            "preflight artifacts already exist; use --validate: " + ", ".join(existing)
        )
    _formal_root_state(protocol)
    plan = build_run_plan(protocol)
    validate_run_plan(protocol, plan)
    _atomic_text(
        protocol.run_plan_path,
        canonical_json_bytes(plan).decode("utf-8"),
    )
    plan_sha256 = sha256_file(protocol.run_plan_path)
    _atomic_text(protocol.run_plan_sha256_path, plan_sha256 + "\n")
    _verify_sidecar(protocol.run_plan_path, protocol.run_plan_sha256_path)

    report = _build_preflight_report(protocol, plan, plan_sha256)
    _atomic_text(
        protocol.preflight_report_path,
        canonical_json_bytes(report).decode("utf-8"),
    )
    report_sha256 = sha256_file(protocol.preflight_report_path)
    _atomic_text(protocol.preflight_report_sha256_path, report_sha256 + "\n")
    _verify_sidecar(
        protocol.preflight_report_path, protocol.preflight_report_sha256_path
    )
    return report


def run_ids(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Return frozen run IDs after validating their basic sequence contract."""

    values = plan.get("frozen_run_ids")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise MultiBudgetProtocolError("frozen_run_ids must be a sequence")
    result = tuple(str(item) for item in values)
    if len(result) != REQUIRED_RUN_COUNT or len(set(result)) != REQUIRED_RUN_COUNT:
        raise MultiBudgetProtocolError("frozen_run_ids must contain 50 unique IDs")
    return result


__all__ = [
    "ANALYSIS_RELEASE_STATUS",
    "MultiBudgetProtocol",
    "MultiBudgetProtocolError",
    "PREFLIGHT_STATUS",
    "REQUIRED_BUDGETS",
    "REQUIRED_CONDITION",
    "REQUIRED_MAPPING_SEEDS",
    "REQUIRED_METHODS",
    "REQUIRED_RUN_COUNT",
    "build_run_plan",
    "canonical_json_bytes",
    "load_protocol",
    "prepare_preflight",
    "run_ids",
    "sha256_file",
    "validate_run_plan",
]
