"""Frozen formal-execution and crash-safe 50-run session for Study B.

Freezing, session preparation, and dry-run validation are performance blind:
they hash asset files as opaque bytes and never open prediction data.  The
formal runner writes only per-sample predictions plus administrative manifests;
it deliberately contains no accuracy or F1 aggregation.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import secrets
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray
import yaml
from threadpoolctl import threadpool_limits

from src.encoders.skeleton_topk import SkeletonTopKEncoder
from src.features import HogFeatureExtractor
from src.grid import make_regular_grid
from src.multibudget.protocol import (
    REQUIRED_RUN_COUNT,
    MultiBudgetProtocol,
    MultiBudgetProtocolError,
    canonical_json_bytes,
    load_protocol,
    sha256_file,
    validate_run_plan,
)
from src.spv_forward import GaussianPhospheneForwardModel, IdealGaussianUtilityProjector
from src.v2.experiment import ConditionSpec, load_mapping
from src.v2.multimapping_prototypes import MultiMappingCosinePrototypeClassifier


FORMAL_SCHEMA_VERSION = 1
FORMAL_LOCK_STATUS = "FROZEN_AWAITING_FORMAL_SESSION"
ASSET_MANIFEST_SCHEMA = "mma-multibudget-assets-v1"
ASSET_MANIFEST_STATUS = "FROZEN_INPUTS_ONLY_NO_PERFORMANCE_READ"
TRAINING_CONFIG = Path("configs/multibudget_formal_inputs.yaml")
REQUIREMENTS_FILE = Path("requirements-windows-dev.txt")
FORMAL_LOCK = Path("locks/multibudget_formal.lock.json")
FORMAL_LOCK_SHA = Path("locks/multibudget_formal.lock.sha256")
FORMAL_STATE_ROOT = Path("outputs/multibudget_formal")
SESSIONS_ROOT = FORMAL_STATE_ROOT / "sessions"
ACTIVE_SESSION = FORMAL_STATE_ROOT / "ACTIVE_SESSION.json"
WORK_ROOT = FORMAL_STATE_ROOT / "work"

PREDICTION_COLUMNS = (
    "method",
    "budget",
    "seed",
    "font_id",
    "sample_token",
    "y_pred",
)
SAMPLE_TOKEN_DOMAIN = b"mma-multibudget-v1:sample-token:v1\x00"
STATE_LOCK_NAME = ".formal_state.lock"
RUN_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "session_id",
        "run_id",
        "method",
        "budget",
        "mapping_seed",
        "prediction_row_count",
        "predictions_sha256",
        "prediction_executed",
    }
)
SESSION_ID_PATTERN = re.compile(
    r"^mma_multibudget_v1_\d{8}T\d{6}Z_[0-9a-f]{8}_[0-9a-f]{8}$"
)


class FormalExecutionError(ValueError):
    """Raised before an invalid freeze, session, resume, or commit."""


@dataclass(frozen=True, slots=True)
class ResumeState:
    session_id: str
    completed_count: int
    total_count: int
    next_run_id: str | None
    completed_run_ids: tuple[str, ...]
    remaining_run_ids: tuple[str, ...]
    status: str
    safe_to_resume: bool


@dataclass(frozen=True, slots=True)
class _Archive:
    images: NDArray[np.float32]
    class_ids: NDArray[np.int64]
    characters: NDArray[np.str_]
    font_ids: NDArray[np.str_]
    sample_ids: NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class _TrainingSpec:
    glyph_archive: Path
    fonts: tuple[str, ...]
    ideal_mapping: Path
    moderate_mappings: Mapping[int, Path]
    feature: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _read_json(path: Path, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise FormalExecutionError(f"non-finite JSON constant {value!r}: {path}")

    if not path.is_file():
        raise FormalExecutionError(f"{name} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise FormalExecutionError(f"{name} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise FormalExecutionError(f"{name} root must be an object: {path}")
    return value


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and path.exists():
            raise FormalExecutionError(f"refusing to overwrite frozen artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any, *, replace: bool = True) -> None:
    _atomic_bytes(path, canonical_json_bytes(value), replace=replace)


@contextmanager
def _state_lock(root: Path):
    """Hold the single non-blocking, crash-released OS lock for state writes."""

    lock_path = root / FORMAL_STATE_ROOT / STATE_LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        if lock_path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            raise FormalExecutionError(
                "formal shared state is locked by another process"
            ) from error
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _write_sidecar(path: Path, sidecar: Path, *, replace: bool = True) -> str:
    digest = sha256_file(path)
    _atomic_bytes(sidecar, f"{digest}  {path.name}\n".encode("ascii"), replace=replace)
    return digest


def _verify_sidecar(path: Path, sidecar: Path) -> str:
    digest = sha256_file(path)
    try:
        expected = sidecar.read_text(encoding="ascii").split()[0].lower()
    except (OSError, IndexError) as error:
        raise FormalExecutionError(f"missing SHA-256 sidecar: {sidecar}") from error
    if expected != digest:
        raise FormalExecutionError(f"SHA-256 sidecar mismatch: {path}")
    return digest


def _resolve_path(root: Path, value: Any, *, base: Path | None = None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FormalExecutionError("bound input path must be a non-empty string")
    raw = value.strip()
    if os.name != "nt" and len(raw) > 2 and raw[1:3] in {":\\", ":/"}:
        return (Path("/mnt") / raw[0].lower() / raw[3:].replace("\\", "/")).resolve()
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    return ((base or root) / candidate).resolve()


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _require_digest(value: Any, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FormalExecutionError(f"{name} must be a SHA-256 digest")
    return digest


def _bound_file(
    root: Path,
    path: Path,
    expected_sha256: Any,
    role: str,
) -> dict[str, Any]:
    expected = _require_digest(expected_sha256, f"{role}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise FormalExecutionError(f"frozen input SHA-256 mismatch for {role}: {path}")
    return {
        "role": role,
        "path": _relative_or_absolute(root, path),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }


def _load_plan(protocol: MultiBudgetProtocol) -> tuple[dict[str, Any], str]:
    plan = _read_json(protocol.run_plan_path, "frozen run plan")
    try:
        validate_run_plan(protocol, plan)
    except MultiBudgetProtocolError as error:
        raise FormalExecutionError(str(error)) from error
    plan_sha = _verify_sidecar(protocol.run_plan_path, protocol.run_plan_sha256_path)
    return plan, plan_sha


def _asset_manifest(protocol: MultiBudgetProtocol) -> tuple[dict[str, Any], str]:
    manifest = _read_json(protocol.asset_manifest_path, "formal asset manifest")
    sidecar = protocol.asset_manifest_path.with_name("assets_manifest.sha256")
    manifest_sha = _verify_sidecar(protocol.asset_manifest_path, sidecar)
    if manifest.get("schema") != ASSET_MANIFEST_SCHEMA:
        raise FormalExecutionError("formal asset manifest schema is invalid")
    if manifest.get("schema_version") != 1:
        raise FormalExecutionError("formal asset manifest schema_version must equal 1")
    if manifest.get("status") != ASSET_MANIFEST_STATUS:
        raise FormalExecutionError("formal assets are not frozen inputs-only")
    forbidden_flags = ("prediction_executed", "accuracy_generated", "accuracy_read")
    if any(manifest.get(name) is not False for name in forbidden_flags):
        raise FormalExecutionError("formal assets are not performance-naive")
    protocol_binding = manifest.get("protocol_config")
    if not isinstance(protocol_binding, Mapping):
        raise FormalExecutionError("asset manifest lacks protocol_config binding")
    if protocol_binding.get("sha256") != sha256_file(protocol.source_path):
        raise FormalExecutionError("asset manifest protocol binding changed")
    glyph = manifest.get("glyph_archive")
    fonts = manifest.get("fonts")
    mappings = manifest.get("mappings")
    if not isinstance(glyph, Mapping):
        raise FormalExecutionError("asset manifest lacks glyph_archive")
    if not isinstance(fonts, list) or len(fonts) != 4:
        raise FormalExecutionError("asset manifest must bind exactly four fonts")
    if not isinstance(mappings, list) or len(mappings) != 5:
        raise FormalExecutionError("asset manifest must bind exactly five mappings")
    if (
        glyph.get("sample_count") != 720
        or glyph.get("class_count") != 180
        or glyph.get("font_count") != 4
    ):
        raise FormalExecutionError("asset manifest glyph dimensions changed")
    seeds = tuple(sorted(int(item.get("seed", -1)) for item in mappings if isinstance(item, Mapping)))
    if seeds != protocol.mapping_seeds:
        raise FormalExecutionError("asset manifest mapping seeds changed")
    font_ids = tuple(str(item.get("font_id", "")) for item in fonts if isinstance(item, Mapping))
    if font_ids != ("MB_F01", "MB_F02", "MB_F03", "MB_F04"):
        raise FormalExecutionError("asset manifest font IDs must be exactly MB_F01..MB_F04")
    governance = manifest.get("selection_governance")
    expected_governance = {
        "font_ids_are_analyst_blinded": True,
        "font_order_rule": (
            "eligible_font_file_sha256_ascending_then_lowest_face_index"
        ),
        "performance_selection": False,
        "font_binary_copied_into_project": False,
    }
    if governance != expected_governance:
        raise FormalExecutionError("font selection governance is not performance-blind")
    expected_rights = {
        "font_binaries_copied": False,
        "font_redistribution_permitted": False,
        "glyph_archive_distribution": (
            "LOCAL_RESEARCH_ONLY_PENDING_SEPARATE_RIGHTS_REVIEW"
        ),
        "public_reproduction_requirement": (
            "separately_preregister_four_new_redistributable_fonts"
        ),
    }
    if manifest.get("rights_boundary") != expected_rights:
        raise FormalExecutionError("asset rights boundary contract changed")
    font_hashes: list[str] = []
    for entry, font_id in zip(fonts, font_ids, strict=True):
        if (
            entry.get("analyst_blinded_id") != font_id
            or entry.get("license_scope")
            != "local_system_font_not_cleared_for_redistribution"
            or entry.get("redistribution_permitted") is not False
            or entry.get("performance_selection") is not False
        ):
            raise FormalExecutionError("font identity or rights governance is incomplete")
        font_hashes.append(
            _require_digest(entry.get("file_sha256"), f"font {font_id} file_sha256")
        )
    if font_hashes != sorted(font_hashes):
        raise FormalExecutionError("font manifest order is not SHA-256 ascending")
    return manifest, manifest_sha


def _manifest_inputs(
    protocol: MultiBudgetProtocol, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    root = protocol.project_root
    base = protocol.asset_manifest_path.parent
    bindings: list[dict[str, Any]] = []

    def add(entry: Any, path_key: str, sha_key: str, role: str, *, relative_to_assets: bool = False) -> None:
        if not isinstance(entry, Mapping):
            raise FormalExecutionError(f"asset manifest lacks {role}")
        path = _resolve_path(root, entry.get(path_key), base=base if relative_to_assets else root)
        bindings.append(_bound_file(root, path, entry.get(sha_key), role))

    add(manifest.get("font_spec"), "path", "sha256", "asset_font_spec")
    add(manifest.get("exclusions"), "path", "sha256", "asset_exclusions")
    add(
        manifest.get("source_character_archive"),
        "path",
        "sha256",
        "source_character_archive",
    )
    add(
        manifest.get("glyph_archive"),
        "file",
        "file_sha256",
        "confirmation_glyph_archive",
        relative_to_assets=True,
    )
    for entry in manifest["fonts"]:
        add(
            entry,
            "source_path",
            "file_sha256",
            f"confirmation_font_{entry['font_id']}",
        )
    for entry in manifest["mappings"]:
        add(
            entry,
            "file",
            "file_sha256",
            f"confirmation_mapping_seed_{entry['seed']}",
            relative_to_assets=True,
        )
    historical = manifest.get("historical_mapping_exclusions")
    if not isinstance(historical, Mapping) or not isinstance(
        historical.get("manifests"), list
    ):
        raise FormalExecutionError("asset manifest lacks historical mapping exclusions")
    for index, entry in enumerate(historical["manifests"]):
        add(entry, "path", "sha256", f"historical_mapping_manifest_{index}")
    contact_sheets = manifest.get("contact_sheets", [])
    if not isinstance(contact_sheets, list):
        raise FormalExecutionError("asset contact_sheets must be a list")
    for index, entry in enumerate(contact_sheets):
        add(
            entry,
            "file",
            "file_sha256",
            f"contact_sheet_{index}",
            relative_to_assets=True,
        )
    paths = [item["path"] for item in bindings]
    if len(paths) != len(set(paths)):
        raise FormalExecutionError("formal asset input manifest contains duplicate paths")
    return bindings


def _training_inputs(protocol: MultiBudgetProtocol) -> list[dict[str, Any]]:
    root = protocol.project_root
    config_path = root / TRAINING_CONFIG
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FormalExecutionError(f"cannot load training template: {config_path}") from error
    if not isinstance(document, Mapping) or not isinstance(document.get("training"), Mapping):
        raise FormalExecutionError("training template lacks training section")
    if (
        document.get("schema") != "mma-multibudget-formal-inputs-v1"
        or document.get("schema_version") != 1
        or document.get("status") != "FROZEN_TRAINING_INPUTS_NO_PERFORMANCE_ACCESS"
    ):
        raise FormalExecutionError("formal input config schema or status is invalid")
    if any(
        document.get(name) is not False
        for name in ("prediction_executed", "accuracy_generated", "accuracy_read")
    ):
        raise FormalExecutionError("formal input config is not performance-naive")
    expected_feature = {
        name: protocol.feature[name]
        for name in ("orientations", "pixels_per_cell", "cells_per_block", "block_norm")
    }
    if document.get("feature") != expected_feature:
        raise FormalExecutionError("formal input feature settings differ from protocol")
    _validate_expected_environment(document.get("environment"))
    training = document["training"]
    bindings = [
        {
            "role": "training_template_config",
            "path": TRAINING_CONFIG.as_posix(),
            "sha256": sha256_file(config_path),
            "size_bytes": config_path.stat().st_size,
        },
        _bound_file(
            root,
            _resolve_path(root, training.get("glyphs_npz")),
            training.get("sha256"),
            "training_glyph_archive",
        ),
    ]
    ideal = training.get("ideal_mapping")
    if not isinstance(ideal, Mapping):
        raise FormalExecutionError("training template lacks ideal mapping")
    bindings.append(
        _bound_file(
            root,
            _resolve_path(root, ideal.get("path")),
            ideal.get("sha256"),
            "training_ideal_mapping",
        )
    )
    moderate = training.get("moderate_mappings")
    if not isinstance(moderate, Mapping) or tuple(sorted(map(int, moderate))) != (2026, 2027, 2028, 2029):
        raise FormalExecutionError("training moderate mappings changed")
    for raw_seed in sorted(moderate, key=int):
        entry = moderate[raw_seed]
        if not isinstance(entry, Mapping):
            raise FormalExecutionError("training mapping entry is invalid")
        bindings.append(
            _bound_file(
                root,
                _resolve_path(root, entry.get("path")),
                entry.get("sha256"),
                f"training_mapping_seed_{int(raw_seed)}",
            )
        )
    return bindings


def _source_manifest(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "src").rglob("*.py"))
    paths.extend(
        path
        for path in (
            root / "freeze_multibudget_formal.py",
            root / "run_multibudget_formal.py",
            root / "analyze_multibudget_formal.py",
        )
        if path.is_file()
    )
    unique = sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())
    if not unique:
        raise FormalExecutionError("formal execution source set is empty")
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in unique
    ]


def _source_digest(files: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in files:
        relative = str(item["path"]).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(str(item["sha256"])))
    return digest.hexdigest()


def _build_lock_document(protocol: MultiBudgetProtocol) -> dict[str, Any]:
    plan, plan_sha = _load_plan(protocol)
    manifest, manifest_sha = _asset_manifest(protocol)
    asset_inputs = _manifest_inputs(protocol, manifest)
    training_inputs = _training_inputs(protocol)
    requirements = protocol.project_root / REQUIREMENTS_FILE
    source_files = _source_manifest(protocol.project_root)
    all_inputs = sorted(
        [*training_inputs, *asset_inputs], key=lambda item: (item["role"], item["path"])
    )
    return {
        "schema": "mma-multibudget-v1-formal-lock",
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": FORMAL_LOCK_STATUS,
        "study_id": protocol.study_id,
        "protocol_config": {
            "path": protocol.source_path.relative_to(protocol.project_root).as_posix(),
            "sha256": sha256_file(protocol.source_path),
        },
        "run_plan": {
            "path": protocol.run_plan_path.relative_to(protocol.project_root).as_posix(),
            "sha256": plan_sha,
            "condition_count": len(plan["conditions"]),
            "frozen_run_ids": list(plan["frozen_run_ids"]),
        },
        "asset_manifest": {
            "path": protocol.asset_manifest_path.relative_to(protocol.project_root).as_posix(),
            "sha256": manifest_sha,
            "status": manifest["status"],
        },
        "frozen_inputs": all_inputs,
        "requirements": {
            "path": REQUIREMENTS_FILE.as_posix(),
            "sha256": sha256_file(requirements),
        },
        "execution_source": {
            "tree_sha256": _source_digest(source_files),
            "files": source_files,
        },
        "execution_contract": {
            "exact_run_count": REQUIRED_RUN_COUNT,
            "run_order": "exact_frozen_run_plan_prefix_only",
            "per_run_atomic_directory_commit": True,
            "completed_record_hash_chain": True,
            "resume_same_session_only": True,
            "unexpected_conditions_rejected": True,
            "prediction_columns": list(PREDICTION_COLUMNS),
            "expected_prediction_rows_per_run": 720,
            "direct_truth_columns_in_prediction_output": False,
            "sample_tokens_are_pseudonyms_not_cryptographic_confidentiality": True,
            "release_gate_threat_model": (
                "cooperative_research_process_governance_not_adversarial_file_access"
            ),
            "aggregate_metrics_forbidden": True,
            "prediction_executed_during_freeze": False,
            "accuracy_read_during_freeze": False,
        },
    }


def _lock_paths(protocol: MultiBudgetProtocol) -> tuple[Path, Path]:
    return protocol.project_root / FORMAL_LOCK, protocol.project_root / FORMAL_LOCK_SHA


def validate_formal_lock(config_path: str | Path) -> dict[str, Any]:
    protocol = load_protocol(config_path)
    lock_path, sidecar = _lock_paths(protocol)
    lock_sha = _verify_sidecar(lock_path, sidecar)
    document = _read_json(lock_path, "formal execution lock")
    expected = _build_lock_document(protocol)
    if document != expected:
        raise FormalExecutionError("formal execution lock differs from current frozen state")
    return {
        "status": "PASS_FORMAL_LOCK_VALID",
        "lock_path": FORMAL_LOCK.as_posix(),
        "lock_sha256": lock_sha,
        "run_count": REQUIRED_RUN_COUNT,
        "prediction_executed": False,
        "accuracy_read": False,
        "validation_mode": "READ_ONLY",
    }


def freeze_formal_lock(
    config_path: str | Path,
    *,
    validate_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Freeze or read-only validate all formal execution bindings."""

    if validate_only and dry_run:
        raise FormalExecutionError("--validate and --dry-run are mutually exclusive")
    if validate_only:
        return validate_formal_lock(config_path)
    protocol = load_protocol(config_path)
    document = _build_lock_document(protocol)
    lock_path, sidecar = _lock_paths(protocol)
    if dry_run:
        if lock_path.exists() or sidecar.exists():
            raise FormalExecutionError("formal lock already exists; use --validate")
        return {
            "status": "PASS_FORMAL_FREEZE_DRY_RUN_READY",
            "would_write": [FORMAL_LOCK.as_posix(), FORMAL_LOCK_SHA.as_posix()],
            "run_count": REQUIRED_RUN_COUNT,
            "frozen_input_count": len(document["frozen_inputs"]),
            "execution_source_file_count": len(document["execution_source"]["files"]),
            "prediction_executed": False,
            "accuracy_read": False,
        }
    if lock_path.exists() or sidecar.exists():
        raise FormalExecutionError("formal lock exists; refusing to replace it")
    _atomic_json(lock_path, document, replace=False)
    lock_sha = _write_sidecar(lock_path, sidecar, replace=False)
    return {
        "status": FORMAL_LOCK_STATUS,
        "lock_path": FORMAL_LOCK.as_posix(),
        "lock_sha256": lock_sha,
        "run_count": REQUIRED_RUN_COUNT,
        "prediction_executed": False,
        "accuracy_read": False,
    }


def _session_paths(session_id: str) -> dict[str, Path]:
    base = SESSIONS_ROOT / session_id
    return {
        "base": base,
        "session": base / "session.json",
        "completed": base / "completed_runs.jsonl",
        "checkpoint": base / "checkpoint.json",
    }


def _make_session_id(created_utc: str, lock_sha: str, nonce: str | None) -> str:
    timestamp = datetime.fromisoformat(created_utc.replace("Z", "+00:00")).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    suffix = nonce or secrets.token_hex(4)
    if not re.fullmatch(r"[0-9a-f]{8}", suffix):
        raise FormalExecutionError("session nonce must contain eight lowercase hex digits")
    return f"mma_multibudget_v1_{timestamp}_{lock_sha[:8]}_{suffix}"


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "PyYAML",
        "threadpoolctl",
    ):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "cpu_only": True,
        "packages": packages,
    }


def _validate_expected_environment(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise FormalExecutionError("formal input config lacks expected environment")
    actual = _environment()
    expected_python = str(
        value.get("python", value.get("python_version", ""))
    )
    if expected_python != actual["python_version"]:
        raise FormalExecutionError(
            "runtime Python differs from formal input expected environment"
        )
    packages = value.get("packages")
    if not isinstance(packages, Mapping):
        raise FormalExecutionError("formal input environment lacks package versions")
    mismatches = [
        name
        for name, actual_version in actual["packages"].items()
        if str(packages.get(name, "")) != actual_version
    ]
    if mismatches:
        raise FormalExecutionError(
            "runtime packages differ from formal input expected environment: "
            + ", ".join(mismatches)
        )


def _prepare_formal_session_unlocked(
    config_path: str | Path,
    *,
    created_utc: str | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Create a zero-of-fifty session without opening any array or prediction."""

    protocol = load_protocol(config_path)
    lock = validate_formal_lock(config_path)
    plan, plan_sha = _load_plan(protocol)
    root = protocol.project_root
    active = root / ACTIVE_SESSION
    session_parent = root / SESSIONS_ROOT
    formal_root = protocol.formal_runs_root
    work_root = root / WORK_ROOT
    if active.exists():
        raise FormalExecutionError("an active formal session already exists")
    for path, name in (
        (session_parent, "session root"),
        (formal_root, "formal run root"),
        (work_root, "formal work root"),
    ):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FormalExecutionError(f"{name} must be absent or empty before preparation")
    timestamp = created_utc or _utc_now()
    session_id = _make_session_id(timestamp, lock["lock_sha256"], nonce)
    paths = _session_paths(session_id)
    absolute = {name: root / path for name, path in paths.items()}
    absolute["base"].mkdir(parents=True, exist_ok=False)
    run_ids = list(plan["frozen_run_ids"])
    session = {
        "schema": "mma-multibudget-v1-formal-session",
        "schema_version": FORMAL_SCHEMA_VERSION,
        "session_id": session_id,
        "status": "PREPARED_ZERO_OF_50_NO_FORMAL_PREDICTIONS",
        "created_utc": timestamp,
        "formal_lock": {"path": FORMAL_LOCK.as_posix(), "sha256": lock["lock_sha256"]},
        "run_plan": {
            "path": protocol.run_plan_path.relative_to(root).as_posix(),
            "sha256": plan_sha,
        },
        "planned_run_ids": run_ids,
        "execution_policy": {
            "exact_prefix_only": True,
            "resume_same_session_only": True,
            "extra_conditions_forbidden": True,
            "per_run_atomic_commit": True,
            "completed_record_hash_chain": True,
            "aggregate_metrics_forbidden": True,
        },
        "paths": {
            "completed_runs": paths["completed"].as_posix(),
            "checkpoint": paths["checkpoint"].as_posix(),
            "formal_runs_root": protocol.formal_runs_root.relative_to(root).as_posix(),
            "work_root": WORK_ROOT.as_posix(),
        },
        "environment": _environment(),
        "prediction_executed_during_preparation": False,
        "accuracy_read_during_preparation": False,
    }
    _atomic_json(absolute["session"], session, replace=False)
    session_sha = _write_sidecar(
        absolute["session"], absolute["session"].with_suffix(".json.sha256"), replace=False
    )
    _atomic_bytes(absolute["completed"], b"", replace=False)
    completed_sha = _write_sidecar(
        absolute["completed"], absolute["completed"].with_suffix(".jsonl.sha256"), replace=False
    )
    checkpoint = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "session_id": session_id,
        "status": "READY_ZERO_OF_50",
        "generation": 0,
        "updated_utc": timestamp,
        "formal_lock_sha256": lock["lock_sha256"],
        "session_sha256": session_sha,
        "completed_manifest_sha256": completed_sha,
        "record_chain_head_sha256": None,
        "completed_count": 0,
        "completed_run_ids": [],
        "next_index": 0,
        "next_run_id": run_ids[0],
        "total_count": REQUIRED_RUN_COUNT,
        "safe_to_resume": True,
    }
    _atomic_json(absolute["checkpoint"], checkpoint, replace=False)
    checkpoint_sha = _write_sidecar(
        absolute["checkpoint"], absolute["checkpoint"].with_suffix(".json.sha256"), replace=False
    )
    pointer = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "session_id": session_id,
        "session_path": paths["session"].as_posix(),
        "session_sha256": session_sha,
        "checkpoint_path": paths["checkpoint"].as_posix(),
        "checkpoint_sha256_at_creation": checkpoint_sha,
        "status": "ACTIVE_PREPARED_ZERO_OF_50",
    }
    _atomic_json(active, pointer, replace=False)
    _write_sidecar(active, active.with_suffix(".json.sha256"), replace=False)
    return validate_formal_session(config_path)


def prepare_formal_session(
    config_path: str | Path,
    *,
    created_utc: str | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Create a zero-of-fifty session under the cross-process state lock."""

    protocol = load_protocol(config_path)
    with _state_lock(protocol.project_root):
        return _prepare_formal_session_unlocked(
            config_path, created_utc=created_utc, nonce=nonce
        )


def _active_session(protocol: MultiBudgetProtocol) -> tuple[dict[str, Any], dict[str, Path]]:
    root = protocol.project_root
    pointer_path = root / ACTIVE_SESSION
    _verify_sidecar(pointer_path, pointer_path.with_suffix(".json.sha256"))
    pointer = _read_json(pointer_path, "active session pointer")
    if (
        pointer.get("schema_version") != FORMAL_SCHEMA_VERSION
        or pointer.get("status") != "ACTIVE_PREPARED_ZERO_OF_50"
    ):
        raise FormalExecutionError("active session pointer schema or status changed")
    session_id = pointer.get("session_id")
    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
        raise FormalExecutionError("active session ID is invalid")
    paths = _session_paths(session_id)
    session_path = root / paths["session"]
    session_sha = _verify_sidecar(session_path, session_path.with_suffix(".json.sha256"))
    if pointer.get("session_path") != paths["session"].as_posix() or pointer.get("session_sha256") != session_sha:
        raise FormalExecutionError("active pointer does not bind the session record")
    return _read_json(session_path, "formal session"), paths


def _validate_session_immutable_bindings(
    protocol: MultiBudgetProtocol,
    session: Mapping[str, Any],
    paths: Mapping[str, Path],
    lock_sha256: str,
    plan_sha256: str,
    plan: Mapping[str, Any],
) -> None:
    root = protocol.project_root
    session_id = paths["base"].name
    expected_policy = {
        "exact_prefix_only": True,
        "resume_same_session_only": True,
        "extra_conditions_forbidden": True,
        "per_run_atomic_commit": True,
        "completed_record_hash_chain": True,
        "aggregate_metrics_forbidden": True,
    }
    expected_paths = {
        "completed_runs": paths["completed"].as_posix(),
        "checkpoint": paths["checkpoint"].as_posix(),
        "formal_runs_root": protocol.formal_runs_root.relative_to(root).as_posix(),
        "work_root": WORK_ROOT.as_posix(),
    }
    required = {
        "schema": "mma-multibudget-v1-formal-session",
        "schema_version": FORMAL_SCHEMA_VERSION,
        "session_id": session_id,
        "status": "PREPARED_ZERO_OF_50_NO_FORMAL_PREDICTIONS",
        "formal_lock": {"path": FORMAL_LOCK.as_posix(), "sha256": lock_sha256},
        "run_plan": {
            "path": protocol.run_plan_path.relative_to(root).as_posix(),
            "sha256": plan_sha256,
        },
        "planned_run_ids": plan["frozen_run_ids"],
        "execution_policy": expected_policy,
        "paths": expected_paths,
        "environment": _environment(),
        "prediction_executed_during_preparation": False,
        "accuracy_read_during_preparation": False,
    }
    if any(session.get(key) != value for key, value in required.items()):
        raise FormalExecutionError("formal session immutable bindings changed")


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous: str | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FormalExecutionError(f"completed-run manifest is unreadable: {path}") from error
    for index, line in enumerate(lines):
        if not line:
            raise FormalExecutionError("completed-run manifest contains a blank line")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise FormalExecutionError("completed-run manifest contains invalid JSON") from error
        if not isinstance(record, dict):
            raise FormalExecutionError("completed-run record must be an object")
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        digest = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        if (
            record.get("record_index") != index
            or record.get("previous_record_sha256") != previous
            or record.get("record_sha256") != digest
        ):
            raise FormalExecutionError("completed-run record hash chain is invalid")
        previous = digest
        records.append(record)
    return records


def _condition_by_index(plan: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    conditions = plan.get("conditions")
    if not isinstance(conditions, list) or not 0 <= index < len(conditions):
        raise FormalExecutionError("run-plan condition index is invalid")
    condition = conditions[index]
    if not isinstance(condition, Mapping):
        raise FormalExecutionError("run-plan condition is invalid")
    return condition


def _validate_prediction_file(path: Path, condition: Mapping[str, Any]) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                raise FormalExecutionError("predictions.csv columns changed")
            rows = list(reader)
    except OSError as error:
        raise FormalExecutionError(f"predictions.csv is unreadable: {path}") from error
    if len(rows) != 720:
        raise FormalExecutionError("predictions.csv must contain exactly 720 rows")
    ordering = [(row["font_id"], row["sample_token"]) for row in rows]
    if ordering != sorted(ordering):
        raise FormalExecutionError(
            "predictions.csv rows must be sorted by font_id and sample_token"
        )
    identities = {(row["font_id"], row["sample_token"]) for row in rows}
    if len(identities) != 720:
        raise FormalExecutionError("predictions.csv contains duplicate sample identities")
    counts = {
        font_id: len({row["sample_token"] for row in rows if row["font_id"] == font_id})
        for font_id in ("MB_F01", "MB_F02", "MB_F03", "MB_F04")
    }
    if counts != {font_id: 180 for font_id in counts}:
        raise FormalExecutionError(
            "predictions.csv must contain 180 unique tokens for each MB_F01..MB_F04"
        )
    constants = {
        "method": str(condition["method"]),
        "budget": str(condition["budget"]),
        "seed": str(condition["mapping_seed"]),
    }
    for row in rows:
        if any(row[name] != value for name, value in constants.items()):
            raise FormalExecutionError("prediction row does not bind the frozen condition")
        try:
            y_pred = int(row["y_pred"])
        except ValueError as error:
            raise FormalExecutionError("predicted class value is not an integer") from error
        token = row["sample_token"]
        if (
            len(token) != 64
            or any(character not in "0123456789abcdef" for character in token)
            or not 0 <= y_pred < 180
        ):
            raise FormalExecutionError("prediction class value is outside the frozen domain")


def _validate_output_directory(
    output: Path, condition: Mapping[str, Any], session_id: str
) -> list[dict[str, Any]]:
    if not output.is_dir():
        raise FormalExecutionError(f"formal output directory is missing: {output}")
    names = {path.name for path in output.iterdir()}
    if names != {"predictions.csv", "run_manifest.json"}:
        raise FormalExecutionError("formal run directory has missing or extra files")
    prediction_path = output / "predictions.csv"
    _validate_prediction_file(prediction_path, condition)
    manifest = _read_json(output / "run_manifest.json", "run manifest")
    if set(manifest) != RUN_MANIFEST_KEYS:
        raise FormalExecutionError("run manifest has missing or extra keys")
    required = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": "PASS_COMMITTED_PREDICTIONS_ONLY",
        "session_id": session_id,
        "run_id": condition["run_id"],
        "method": condition["method"],
        "budget": condition["budget"],
        "mapping_seed": condition["mapping_seed"],
        "prediction_row_count": 720,
        "predictions_sha256": sha256_file(prediction_path),
        "prediction_executed": True,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise FormalExecutionError("run manifest does not bind the committed prediction file")
    return [
        {
            "path": "predictions.csv",
            "size_bytes": prediction_path.stat().st_size,
            "sha256": sha256_file(prediction_path),
        },
        {
            "path": "run_manifest.json",
            "size_bytes": (output / "run_manifest.json").stat().st_size,
            "sha256": sha256_file(output / "run_manifest.json"),
        },
    ]


def _verify_record_outputs(
    root: Path,
    record: Mapping[str, Any],
    condition: Mapping[str, Any],
    session_id: str,
) -> None:
    output = _resolve_path(root, record.get("output_dir"))
    expected = _validate_output_directory(output, condition, session_id)
    if record.get("files") != expected:
        raise FormalExecutionError("completed-run record file hashes changed")


def load_resume_state(config_path: str | Path) -> ResumeState:
    """Read-only verify lock, session, exact prefix, hashes, and no extras."""

    protocol = load_protocol(config_path)
    lock = validate_formal_lock(config_path)
    plan, plan_sha = _load_plan(protocol)
    session, paths = _active_session(protocol)
    root = protocol.project_root
    _validate_session_immutable_bindings(
        protocol,
        session,
        paths,
        lock["lock_sha256"],
        plan_sha,
        plan,
    )
    completed_path = root / paths["completed"]
    _verify_sidecar(completed_path, completed_path.with_suffix(".jsonl.sha256"))
    records = _read_records(completed_path)
    run_ids = list(plan["frozen_run_ids"])
    completed_ids = [str(record.get("run_id")) for record in records]
    if completed_ids != run_ids[: len(completed_ids)]:
        raise FormalExecutionError("completed runs are not an exact frozen-plan prefix")
    if len(records) > REQUIRED_RUN_COUNT:
        raise FormalExecutionError("completed-run manifest contains extra conditions")
    for index, record in enumerate(records):
        _verify_record_outputs(
            root,
            record,
            _condition_by_index(plan, index),
            str(session["session_id"]),
        )
    expected_dirs = set(completed_ids)
    actual_dirs: set[str] = set()
    if protocol.formal_runs_root.exists():
        if not protocol.formal_runs_root.is_dir():
            raise FormalExecutionError("formal_runs_root is not a directory")
        entries = list(protocol.formal_runs_root.iterdir())
        if any(not entry.is_dir() for entry in entries):
            raise FormalExecutionError("formal_runs_root contains unexpected files")
        actual_dirs = {entry.name for entry in entries}
    if actual_dirs != expected_dirs:
        raise FormalExecutionError("formal_runs_root contains uncommitted or extra conditions")
    work_session = root / WORK_ROOT / str(session["session_id"])
    if work_session.exists():
        partial_names = {path.name for path in work_session.iterdir() if path.is_dir()}
        expected_partial = (
            {f"{run_ids[len(records)]}.partial"} if len(records) < len(run_ids) else set()
        )
        if not partial_names <= expected_partial or any(
            not path.is_dir() for path in work_session.iterdir()
        ):
            raise FormalExecutionError("work root contains a non-next or malformed partial run")
    checkpoint_path = root / paths["checkpoint"]
    _verify_sidecar(checkpoint_path, checkpoint_path.with_suffix(".json.sha256"))
    checkpoint = _read_json(checkpoint_path, "formal checkpoint")
    manifest_sha = sha256_file(completed_path)
    chain_head = records[-1]["record_sha256"] if records else None
    next_id = run_ids[len(records)] if len(records) < len(run_ids) else None
    expected_checkpoint = {
        "completed_manifest_sha256": manifest_sha,
        "record_chain_head_sha256": chain_head,
        "completed_count": len(records),
        "completed_run_ids": completed_ids,
        "next_index": len(records),
        "next_run_id": next_id,
        "total_count": REQUIRED_RUN_COUNT,
    }
    if any(checkpoint.get(key) != value for key, value in expected_checkpoint.items()):
        raise FormalExecutionError("checkpoint differs from completed-run hash chain")
    return ResumeState(
        session_id=str(session["session_id"]),
        completed_count=len(records),
        total_count=REQUIRED_RUN_COUNT,
        next_run_id=next_id,
        completed_run_ids=tuple(completed_ids),
        remaining_run_ids=tuple(run_ids[len(records) :]),
        status=str(checkpoint["status"]),
        safe_to_resume=True,
    )


def validate_formal_session(config_path: str | Path) -> dict[str, Any]:
    state = load_resume_state(config_path)
    return {
        "session_id": state.session_id,
        "status": state.status,
        "completed_count": state.completed_count,
        "remaining_count": state.total_count - state.completed_count,
        "next_run_id": state.next_run_id,
        "exact_prefix_verified": True,
        "hash_chain_verified": True,
        "unexpected_conditions": 0,
        "safe_to_resume": state.safe_to_resume,
        "validation_mode": "READ_ONLY",
    }


def _prediction_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=PREDICTION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        sorted(rows, key=lambda row: (str(row["font_id"]), str(row["sample_token"])))
    )
    return buffer.getvalue().encode("utf-8")


def sample_token(sample_id: str) -> str:
    """Pseudonymize a frozen sample identity with an explicit fixed domain."""

    digest = hashlib.sha256()
    digest.update(SAMPLE_TOKEN_DOMAIN)
    digest.update(sample_id.encode("utf-8"))
    return digest.hexdigest()


def _write_staged_predictions(
    staging: Path,
    condition: Mapping[str, Any],
    session_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    prediction_path = staging / "predictions.csv"
    _atomic_bytes(prediction_path, _prediction_csv_bytes(rows))
    _validate_prediction_file(prediction_path, condition)
    manifest = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": "PASS_COMMITTED_PREDICTIONS_ONLY",
        "session_id": session_id,
        "run_id": condition["run_id"],
        "method": condition["method"],
        "budget": condition["budget"],
        "mapping_seed": condition["mapping_seed"],
        "prediction_row_count": 720,
        "predictions_sha256": sha256_file(prediction_path),
        "prediction_executed": True,
    }
    _atomic_json(staging / "run_manifest.json", manifest)


def _append_completed_record(
    protocol: MultiBudgetProtocol,
    plan: Mapping[str, Any],
    state: ResumeState,
    output: Path,
) -> dict[str, Any]:
    root = protocol.project_root
    _, paths = _active_session(protocol)
    condition = _condition_by_index(plan, state.completed_count)
    files = _validate_output_directory(output, condition, state.session_id)
    previous = None
    completed_path = root / paths["completed"]
    records = _read_records(completed_path)
    if records:
        previous = records[-1]["record_sha256"]
    unsigned = {
        "record_index": state.completed_count,
        "session_id": state.session_id,
        "run_id": condition["run_id"],
        "completed_utc": _utc_now(),
        "output_dir": output.relative_to(root).as_posix(),
        "files": files,
        "previous_record_sha256": previous,
    }
    record = {
        **unsigned,
        "record_sha256": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }
    existing = completed_path.read_bytes()
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    _atomic_bytes(completed_path, existing + line)
    completed_sha = _write_sidecar(
        completed_path, completed_path.with_suffix(".jsonl.sha256")
    )
    completed_ids = [*state.completed_run_ids, str(condition["run_id"])]
    next_index = len(completed_ids)
    next_run_id = (
        str(plan["frozen_run_ids"][next_index])
        if next_index < REQUIRED_RUN_COUNT
        else None
    )
    checkpoint_path = root / paths["checkpoint"]
    old = _read_json(checkpoint_path, "formal checkpoint")
    checkpoint = {
        **old,
        "status": "COMPLETE_50_OF_50" if next_run_id is None else "IN_PROGRESS",
        "generation": int(old["generation"]) + 1,
        "updated_utc": record["completed_utc"],
        "completed_manifest_sha256": completed_sha,
        "record_chain_head_sha256": record["record_sha256"],
        "completed_count": next_index,
        "completed_run_ids": completed_ids,
        "next_index": next_index,
        "next_run_id": next_run_id,
    }
    _atomic_json(checkpoint_path, checkpoint)
    _write_sidecar(checkpoint_path, checkpoint_path.with_suffix(".json.sha256"))
    return record


def _repair_checkpoint_after_interruption(
    protocol: MultiBudgetProtocol,
    session: Mapping[str, Any],
    paths: Mapping[str, Path],
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Repair only bookkeeping already proven by chained records and outputs."""

    root = protocol.project_root
    completed_path = root / paths["completed"]
    checkpoint_path = root / paths["checkpoint"]
    checkpoint = _read_json(checkpoint_path, "formal checkpoint")
    session_path = root / paths["session"]
    session_sha = sha256_file(session_path)
    lock_sha = str(session["formal_lock"]["sha256"])
    if (
        checkpoint.get("session_id") != session.get("session_id")
        or checkpoint.get("formal_lock_sha256") != lock_sha
        or checkpoint.get("session_sha256") != session_sha
        or checkpoint.get("total_count") != REQUIRED_RUN_COUNT
    ):
        raise FormalExecutionError("checkpoint immutable bindings changed")
    _write_sidecar(completed_path, completed_path.with_suffix(".jsonl.sha256"))
    run_ids = list(plan["frozen_run_ids"])
    completed_ids = [str(item["run_id"]) for item in records]
    next_index = len(records)
    next_run_id = run_ids[next_index] if next_index < REQUIRED_RUN_COUNT else None
    repaired = {
        **checkpoint,
        "status": "COMPLETE_50_OF_50" if next_run_id is None else (
            "READY_ZERO_OF_50" if next_index == 0 else "IN_PROGRESS"
        ),
        "generation": max(int(checkpoint.get("generation", 0)), next_index),
        "updated_utc": str(records[-1]["completed_utc"]) if records else checkpoint["updated_utc"],
        "completed_manifest_sha256": sha256_file(completed_path),
        "record_chain_head_sha256": records[-1]["record_sha256"] if records else None,
        "completed_count": next_index,
        "completed_run_ids": completed_ids,
        "next_index": next_index,
        "next_run_id": next_run_id,
        "safe_to_resume": True,
    }
    _atomic_json(checkpoint_path, repaired)
    _write_sidecar(checkpoint_path, checkpoint_path.with_suffix(".json.sha256"))


def _recover_interrupted_commit_unlocked(config_path: str | Path) -> str | None:
    """Finish bookkeeping for one already-atomic next output during --resume."""

    protocol = load_protocol(config_path)
    lock = validate_formal_lock(config_path)
    plan, plan_sha = _load_plan(protocol)
    session, paths = _active_session(protocol)
    _validate_session_immutable_bindings(
        protocol,
        session,
        paths,
        lock["lock_sha256"],
        plan_sha,
        plan,
    )
    root = protocol.project_root
    completed_path = root / paths["completed"]
    records = _read_records(completed_path)
    run_ids = list(plan["frozen_run_ids"])
    completed_ids = [str(item.get("run_id")) for item in records]
    if completed_ids != run_ids[: len(completed_ids)]:
        raise FormalExecutionError("completed runs are not an exact frozen-plan prefix")
    for index, record in enumerate(records):
        _verify_record_outputs(
            root,
            record,
            _condition_by_index(plan, index),
            str(session["session_id"]),
        )
    if protocol.formal_runs_root.exists():
        formal_entries = list(protocol.formal_runs_root.iterdir())
        if any(not path.is_dir() for path in formal_entries):
            raise FormalExecutionError("formal_runs_root contains unexpected files")
        actual_dirs = {path.name for path in formal_entries}
    else:
        actual_dirs = set()
    expected_dirs = set(completed_ids)
    next_id = run_ids[len(records)] if len(records) < REQUIRED_RUN_COUNT else None
    if actual_dirs == expected_dirs:
        _repair_checkpoint_after_interruption(protocol, session, paths, plan, records)
        return None
    if next_id is None or actual_dirs != expected_dirs | {next_id}:
        raise FormalExecutionError("formal_runs_root contains irrecoverable extra conditions")
    condition = _condition_by_index(plan, len(records))
    output = protocol.formal_runs_root / next_id
    _validate_output_directory(output, condition, str(session["session_id"]))
    _repair_checkpoint_after_interruption(protocol, session, paths, plan, records)
    state = ResumeState(
        session_id=str(session["session_id"]),
        completed_count=len(records),
        total_count=REQUIRED_RUN_COUNT,
        next_run_id=next_id,
        completed_run_ids=tuple(completed_ids),
        remaining_run_ids=tuple(run_ids[len(records) :]),
        status="IN_PROGRESS",
        safe_to_resume=True,
    )
    _append_completed_record(protocol, plan, state, output)
    return next_id


def _recover_interrupted_commit(config_path: str | Path) -> str | None:
    protocol = load_protocol(config_path)
    with _state_lock(protocol.project_root):
        return _recover_interrupted_commit_unlocked(config_path)


def _commit_prediction_rows_unlocked(
    config_path: str | Path,
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically commit the exact next run's prediction-only rows."""

    protocol = load_protocol(config_path)
    state = load_resume_state(config_path)
    if run_id != state.next_run_id:
        raise FormalExecutionError("only the exact next frozen run may be committed")
    plan, _ = _load_plan(protocol)
    condition = _condition_by_index(plan, state.completed_count)
    staging = protocol.project_root / WORK_ROOT / state.session_id / f"{run_id}.partial"
    staging.mkdir(parents=True, exist_ok=True)
    _write_staged_predictions(staging, condition, state.session_id, rows)
    _validate_output_directory(staging, condition, state.session_id)
    output = protocol.formal_runs_root / run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FormalExecutionError("formal output directory already exists")
    os.replace(staging, output)
    return _append_completed_record(protocol, plan, state, output)


def commit_prediction_rows(
    config_path: str | Path,
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Commit the exact next prediction-only run under the OS state lock."""

    protocol = load_protocol(config_path)
    with _state_lock(protocol.project_root):
        return _commit_prediction_rows_unlocked(config_path, run_id, rows)


def _asset_paths(
    protocol: MultiBudgetProtocol, manifest: Mapping[str, Any]
) -> tuple[Path, tuple[str, ...], dict[int, Path]]:
    base = protocol.asset_manifest_path.parent
    glyph = manifest["glyph_archive"]
    glyph_path = _resolve_path(protocol.project_root, glyph["file"], base=base)
    fonts = tuple(str(item["font_id"]) for item in manifest["fonts"])
    mappings = {
        int(item["seed"]): _resolve_path(protocol.project_root, item["file"], base=base)
        for item in manifest["mappings"]
    }
    return glyph_path, fonts, mappings


def _training_spec(protocol: MultiBudgetProtocol) -> _TrainingSpec:
    path = protocol.project_root / TRAINING_CONFIG
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FormalExecutionError(f"cannot load training template: {path}") from error
    if not isinstance(document, Mapping):
        raise FormalExecutionError("training template root must be a mapping")
    training = document.get("training")
    feature = document.get("feature")
    if not isinstance(training, Mapping) or not isinstance(feature, Mapping):
        raise FormalExecutionError("training template is incomplete")
    ideal = training.get("ideal_mapping")
    moderate = training.get("moderate_mappings")
    fonts = training.get("fonts")
    if (
        not isinstance(ideal, Mapping)
        or not isinstance(moderate, Mapping)
        or not isinstance(fonts, list)
    ):
        raise FormalExecutionError("training template input bindings are incomplete")
    return _TrainingSpec(
        glyph_archive=_resolve_path(protocol.project_root, training.get("glyphs_npz")),
        fonts=tuple(str(item) for item in fonts),
        ideal_mapping=_resolve_path(protocol.project_root, ideal.get("path")),
        moderate_mappings={
            int(seed): _resolve_path(protocol.project_root, entry.get("path"))
            for seed, entry in moderate.items()
            if isinstance(entry, Mapping)
        },
        feature=dict(feature),
    )


def _load_glyph_archive(path: Path, selected_fonts: Sequence[str]) -> _Archive:
    with np.load(path, allow_pickle=False) as archive:
        required = {"images", "class_ids", "characters", "font_ids", "sample_ids"}
        if not required <= set(archive.files):
            raise FormalExecutionError(f"glyph archive is incomplete: {path}")
        raw = {name: archive[name] for name in required}
    mask = np.isin(raw["font_ids"], np.asarray(selected_fonts, dtype=np.str_))
    expected_count = len(selected_fonts) * 180
    if int(mask.sum()) != expected_count:
        raise FormalExecutionError(f"glyph archive does not contain fonts x 180: {path}")
    result = _Archive(
        images=np.ascontiguousarray(raw["images"][mask], dtype=np.float32),
        class_ids=np.ascontiguousarray(raw["class_ids"][mask], dtype=np.int64),
        characters=np.asarray(raw["characters"][mask], dtype=np.str_).copy(),
        font_ids=np.asarray(raw["font_ids"][mask], dtype=np.str_).copy(),
        sample_ids=np.asarray(raw["sample_ids"][mask], dtype=np.str_).copy(),
    )
    if result.images.shape != (expected_count, 96, 96):
        raise FormalExecutionError("glyph image shape differs from protocol")
    if not np.isfinite(result.images).all() or not (
        0.0 <= float(result.images.min()) <= float(result.images.max()) <= 1.0
    ):
        raise FormalExecutionError("glyph images contain invalid numeric values")
    if len(set(map(str, result.sample_ids))) != expected_count:
        raise FormalExecutionError("glyph sample IDs must be unique")
    if set(map(str, np.unique(result.font_ids))) != set(selected_fonts):
        raise FormalExecutionError("glyph font identities differ from manifest")
    for font in selected_fonts:
        labels = np.sort(result.class_ids[result.font_ids == font])
        if not np.array_equal(labels, np.arange(180, dtype=np.int64)):
            raise FormalExecutionError(f"font {font} does not contain classes 0..179")
    return result


def _character_sequence(archive: _Archive) -> tuple[str, ...]:
    sequence: list[str] = []
    for class_id in range(180):
        values = np.unique(archive.characters[archive.class_ids == class_id])
        if values.shape != (1,):
            raise FormalExecutionError(f"ambiguous character for class {class_id}")
        sequence.append(str(values[0]))
    return tuple(sequence)


def _rank_positions(
    images: NDArray[np.float32], ideal_mapping: Any
) -> NDArray[np.int64]:
    grid = make_regular_grid(rows=16, cols=16, image_shape=(96, 96)).electrodes
    if not np.array_equal(ideal_mapping.centers_xy, grid.centers_xy):
        raise FormalExecutionError("ideal mapping differs from the regular 16x16 grid")
    encoder = SkeletonTopKEncoder()
    projector = IdealGaussianUtilityProjector(ideal_mapping)
    ranking = np.empty((images.shape[0], 128), dtype=np.int64)
    for index, image in enumerate(images):
        ranking[index] = encoder.encode(
            image, grid, 128, projector=projector
        ).selected_indices
    return np.ascontiguousarray(ranking)


def _feature_extractor(spec: _TrainingSpec) -> HogFeatureExtractor:
    return HogFeatureExtractor(
        image_shape=(96, 96),
        orientations=int(spec.feature.get("orientations", 9)),
        pixels_per_cell=tuple(map(int, spec.feature.get("pixels_per_cell", [8, 8]))),
        cells_per_block=tuple(map(int, spec.feature.get("cells_per_block", [2, 2]))),
        block_norm=str(spec.feature.get("block_norm", "L2-Hys")),
    )


def _mapping_features(
    mapping: Any,
    selected: NDArray[np.int64],
    extractor: HogFeatureExtractor,
) -> NDArray[np.float32]:
    amplitudes = np.ones(selected.shape, dtype=np.float32)
    phosphene_images = GaussianPhospheneForwardModel(mapping).generate_variable_batch(
        selected, amplitudes
    )
    return extractor.extract_batch(phosphene_images)


def _fit_predictor(
    features_by_mapping: Mapping[str, NDArray[np.float32]],
    archive: _Archive,
    mapping_ids: Sequence[str],
) -> MultiMappingCosinePrototypeClassifier:
    features = np.concatenate([features_by_mapping[name] for name in mapping_ids])
    labels = np.tile(archive.class_ids, len(mapping_ids))
    fonts = np.tile(archive.font_ids, len(mapping_ids))
    mappings = np.concatenate(
        [np.full(archive.class_ids.shape, name) for name in mapping_ids]
    )
    return MultiMappingCosinePrototypeClassifier(tuple(range(180))).fit(
        np.ascontiguousarray(features, dtype=np.float32),
        np.ascontiguousarray(labels, dtype=np.int64),
        np.asarray(fonts, dtype=np.str_),
        np.asarray(mappings, dtype=np.str_),
    )


def _predict_condition(
    protocol: MultiBudgetProtocol, condition: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Produce labels for one frozen cell without computing any score."""

    manifest, _ = _asset_manifest(protocol)
    training = _training_spec(protocol)
    test_path, test_fonts, test_mapping_paths = _asset_paths(protocol, manifest)
    train = _load_glyph_archive(training.glyph_archive, training.fonts)
    test = _load_glyph_archive(test_path, test_fonts)
    if set(map(str, train.sample_ids)) & set(map(str, test.sample_ids)):
        raise FormalExecutionError("training and confirmation sample IDs overlap")
    if _character_sequence(train) != _character_sequence(test):
        raise FormalExecutionError("training and confirmation character sequences differ")
    budget = int(condition["budget"])
    seed = int(condition["mapping_seed"])
    ideal = load_mapping(
        training.ideal_mapping,
        condition=ConditionSpec("skeleton_topk", budget, "ideal", 0),
    )
    train_mappings = {
        item_seed: load_mapping(
            path,
            condition=ConditionSpec("skeleton_topk", budget, "moderate", item_seed),
        )
        for item_seed, path in training.moderate_mappings.items()
    }
    test_mapping = load_mapping(
        test_mapping_paths[seed],
        condition=ConditionSpec("skeleton_topk", budget, "moderate", seed),
    )
    with threadpool_limits(limits=1):
        train_selected = np.ascontiguousarray(_rank_positions(train.images, ideal)[:, :budget])
        test_selected = np.ascontiguousarray(_rank_positions(test.images, ideal)[:, :budget])
        extractor = _feature_extractor(training)
        features = {"ideal": _mapping_features(ideal, train_selected, extractor)}
        for item_seed, mapping in sorted(train_mappings.items()):
            features[f"moderate_seed{item_seed}"] = _mapping_features(
                mapping, train_selected, extractor
            )
        mapping_ids = tuple(str(item) for item in condition["training_mapping_ids"])
        model = _fit_predictor(features, train, mapping_ids)
        test_features = _mapping_features(test_mapping, test_selected, extractor)
        prediction = np.asarray(model.predict(test_features), dtype=np.int64)
    return [
        {
            "method": condition["method"],
            "budget": budget,
            "seed": seed,
            "font_id": str(test.font_ids[index]),
            "sample_token": sample_token(str(test.sample_ids[index])),
            "y_pred": int(prediction[index]),
        }
        for index in range(test.class_ids.size)
    ]


def execute_next_run(config_path: str | Path) -> dict[str, Any]:
    """Execute and commit exactly the next frozen condition, for resumability."""

    protocol = load_protocol(config_path)
    state = load_resume_state(config_path)
    if state.next_run_id is None:
        return {
            "status": "COMPLETE_50_OF_50",
            "completed_count": state.completed_count,
            "next_run_id": None,
        }
    plan, _ = _load_plan(protocol)
    condition = _condition_by_index(plan, state.completed_count)
    rows = _predict_condition(protocol, condition)
    record = commit_prediction_rows(config_path, state.next_run_id, rows)
    next_state = load_resume_state(config_path)
    return {
        "status": next_state.status,
        "committed_run_id": record["run_id"],
        "completed_count": next_state.completed_count,
        "remaining_count": next_state.total_count - next_state.completed_count,
        "next_run_id": next_state.next_run_id,
    }


def resume_formal_session(
    config_path: str | Path, *, max_runs: int = 1
) -> dict[str, Any]:
    if not 1 <= max_runs <= REQUIRED_RUN_COUNT:
        raise FormalExecutionError("max_runs must be between 1 and 50")
    committed: list[str] = []
    recovered = _recover_interrupted_commit(config_path)
    if recovered is not None:
        committed.append(recovered)
    for _ in range(max_runs - len(committed)):
        state = load_resume_state(config_path)
        if state.next_run_id is None:
            break
        result = execute_next_run(config_path)
        committed.append(str(result["committed_run_id"]))
    final = load_resume_state(config_path)
    return {
        "status": final.status,
        "committed_this_invocation": committed,
        "completed_count": final.completed_count,
        "remaining_count": final.total_count - final.completed_count,
        "next_run_id": final.next_run_id,
        "safe_to_resume": final.safe_to_resume,
    }


__all__ = [
    "ACTIVE_SESSION",
    "FORMAL_LOCK",
    "FORMAL_LOCK_SHA",
    "FormalExecutionError",
    "PREDICTION_COLUMNS",
    "ResumeState",
    "commit_prediction_rows",
    "execute_next_run",
    "freeze_formal_lock",
    "load_resume_state",
    "prepare_formal_session",
    "resume_formal_session",
    "sample_token",
    "validate_formal_lock",
    "validate_formal_session",
]
