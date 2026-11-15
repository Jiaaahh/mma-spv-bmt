"""Integrity lock for the one-shot E1 confirmation experiment.

The lock binds the protocol configuration, the frozen asset manifest, every
explicitly declared external input, the pinned requirements file, and the
execution source bundle.  It intentionally contains no performance result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import yaml


LOCK_SCHEMA = "hanzi-spv-e1-confirmation-lock"
LOCK_SCHEMA_VERSION = 1
DEFAULT_LOCK_PATH = Path("locks/e1_confirmation.lock.json")
DEFAULT_REQUIREMENTS_PATH = Path("requirements-windows-dev.txt")


class E1IntegrityError(RuntimeError):
    """Raised when an E1 freeze or integrity validation fails."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""

    source = Path(path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not source.is_file():
        raise E1IntegrityError(f"integrity input is not a file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_bundle_paths(project_root: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    source_root = project_root / "src"
    if source_root.is_dir():
        paths.update(path for path in source_root.rglob("*.py") if path.is_file())
    for name in ("prepare_e1_assets.py", "freeze_e1_confirmation.py"):
        path = project_root / name
        if path.is_file():
            paths.add(path)
    paths.update(path for path in project_root.glob("run_e1_*.py") if path.is_file())
    if not paths:
        raise E1IntegrityError(f"E1 source bundle is empty: {project_root}")
    return tuple(sorted(paths, key=lambda path: path.relative_to(project_root).as_posix()))


def source_bundle_sha256(project_root: str | Path) -> str:
    """Hash the deterministic E1 execution-source bundle.

    Each entry is bound by its UTF-8 project-relative path and raw file digest,
    so both a rename and a content change alter the bundle hash.
    """

    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    for path in _source_bundle_paths(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def atomic_json(path: str | Path, document: Mapping[str, Any]) -> None:
    """Atomically replace *path* with canonical, human-readable JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_config(config_path: Path) -> Mapping[str, Any]:
    if not config_path.is_file():
        raise E1IntegrityError(f"E1 config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise E1IntegrityError("E1 config root must be a mapping")
    return document


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _find_asset_manifest(config: Mapping[str, Any]) -> str | Path | None:
    direct = config.get("asset_manifest")
    if isinstance(direct, (str, Path)):
        return direct
    for section_name in ("assets", "confirmation", "test"):
        section = config.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for key in ("asset_manifest", "manifest"):
            value = section.get(key)
            if isinstance(value, (str, Path)):
                base = section.get("asset_dir")
                if isinstance(base, (str, Path)) and not Path(value).is_absolute():
                    return Path(base) / Path(value)
                return value
    return None


def _flatten_path_values(value: Any, prefix: str = "external") -> dict[str, str | Path]:
    """Flatten a deliberately explicit external-input declaration."""

    if isinstance(value, (str, Path)):
        return {prefix: value}
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), (str, Path)):
            name = value.get("name", prefix)
            return {str(name): value["path"]}
        flattened: dict[str, str | Path] = {}
        for key in sorted(value, key=lambda item: str(item)):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_path_values(value[key], child_prefix))
        return flattened
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten_path_values(item, f"{prefix}.{index:03d}"))
        return flattened
    raise E1IntegrityError(
        f"external input declaration {prefix!r} must contain file paths"
    )


def _external_inputs(
    config: Mapping[str, Any],
    supplied: Mapping[str, str | Path] | Iterable[str | Path] | None,
) -> dict[str, str | Path]:
    if supplied is None:
        declared = config.get("external_inputs")
        if declared is None:
            confirmation = config.get("confirmation")
            if isinstance(confirmation, Mapping):
                declared = confirmation.get("external_inputs")
        if declared is None:
            raise E1IntegrityError(
                "no external inputs were supplied or declared under external_inputs"
            )
        return _flatten_path_values(declared, "")
    if isinstance(supplied, Mapping):
        return _flatten_path_values(supplied, "")
    return _flatten_path_values(tuple(supplied), "external")


def _path_record(path: Path, project_root: Path) -> dict[str, str]:
    try:
        display = path.relative_to(project_root).as_posix()
        scope = "project_relative"
    except ValueError:
        display = str(path)
        scope = "absolute"
    return {"path": display, "path_scope": scope, "sha256": sha256_file(path)}


def _record_path(record: Mapping[str, Any], project_root: Path, label: str) -> Path:
    value = record.get("path")
    scope = record.get("path_scope")
    if not isinstance(value, str) or scope not in {"project_relative", "absolute"}:
        raise E1IntegrityError(f"invalid path record in lock: {label}")
    path = Path(value) if scope == "absolute" else project_root / Path(value)
    return path.resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def freeze(
    config_path: str | Path,
    asset_manifest_path: str | Path | None = None,
    external_input_paths: Mapping[str, str | Path] | Iterable[str | Path] | None = None,
    *,
    lock_path: str | Path | None = None,
    requirements_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze the E1 confirmation protocol before any prediction is executed."""

    config_source = Path(config_path).resolve()
    project_root = config_source.parent.parent.resolve()
    config = _load_config(config_source)
    manifest_value = asset_manifest_path or _find_asset_manifest(config)
    if manifest_value is None:
        raise E1IntegrityError("asset manifest path was not supplied or declared")
    manifest = _resolve(project_root, manifest_value)
    requirements = _resolve(
        project_root, requirements_path or DEFAULT_REQUIREMENTS_PATH
    )
    target = _resolve(project_root, lock_path or DEFAULT_LOCK_PATH)
    if target.exists() and not force:
        raise E1IntegrityError(
            f"E1 confirmation lock already exists; refusing to overwrite: {target}"
        )

    declared = _external_inputs(config, external_input_paths)
    input_records: dict[str, dict[str, str]] = {}
    seen_paths: dict[Path, str] = {}
    for raw_name, raw_path in sorted(declared.items()):
        name = str(raw_name).strip()
        if not name:
            raise E1IntegrityError("external input names must not be empty")
        resolved = _resolve(project_root, raw_path)
        if resolved in seen_paths:
            raise E1IntegrityError(
                f"duplicate external input path: {name} and {seen_paths[resolved]}"
            )
        seen_paths[resolved] = name
        input_records[name] = _path_record(resolved, project_root)

    config_record = _path_record(config_source, project_root)
    manifest_record = _path_record(manifest, project_root)
    requirements_record = _path_record(requirements, project_root)
    document: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "schema_version": LOCK_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "status": "FROZEN_AWAITING_ONE_SHOT_E1_CONFIRMATION",
        "project_root": str(project_root),
        "config": config_record,
        "config_sha256": config_record["sha256"],
        "asset_manifest": manifest_record,
        "asset_manifest_sha256": manifest_record["sha256"],
        "requirements": requirements_record,
        "requirements_sha256": requirements_record["sha256"],
        "source_bundle_sha256": source_bundle_sha256(project_root),
        "input_files": input_records,
        "input_file_sha256": {
            name: record["sha256"] for name, record in input_records.items()
        },
        "prediction_executed": False,
        "accuracy_read": False,
    }
    atomic_json(target, document)
    result = validate_lock(target, project_root=project_root)
    return result


def _validate_record(
    label: str,
    record: Any,
    project_root: Path,
) -> None:
    if not isinstance(record, Mapping):
        raise E1IntegrityError(f"missing or invalid lock record: {label}")
    path = _record_path(record, project_root, label)
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise E1IntegrityError(f"invalid frozen digest for {label}")
    actual = sha256_file(path)
    if actual != expected:
        raise E1IntegrityError(
            f"frozen input changed after E1 freeze: {label} ({path})"
        )


def validate_lock(
    lock_path: str | Path = DEFAULT_LOCK_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute and validate every binding in an E1 confirmation lock."""

    source = Path(lock_path).resolve()
    if not source.is_file():
        raise E1IntegrityError(f"E1 confirmation lock does not exist: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise E1IntegrityError(f"cannot read E1 confirmation lock: {source}") from error
    if not isinstance(document, Mapping):
        raise E1IntegrityError("E1 confirmation lock root must be a mapping")
    if (
        document.get("schema") != LOCK_SCHEMA
        or document.get("schema_version") != LOCK_SCHEMA_VERSION
    ):
        raise E1IntegrityError("unsupported E1 confirmation lock schema")
    if document.get("status") != "FROZEN_AWAITING_ONE_SHOT_E1_CONFIRMATION":
        raise E1IntegrityError("E1 confirmation lock status is invalid")
    if document.get("prediction_executed") is not False:
        raise E1IntegrityError("lock was not frozen before prediction execution")
    if document.get("accuracy_read") is not False:
        raise E1IntegrityError("lock was not frozen before accuracy access")

    if project_root is None:
        root = source.parent.parent.resolve()
    else:
        root = Path(project_root).resolve()
    frozen_root = document.get("project_root")
    if not isinstance(frozen_root, str) or Path(frozen_root).resolve() != root:
        raise E1IntegrityError("E1 lock project root differs from the validation root")

    _validate_record("config", document.get("config"), root)
    _validate_record("asset_manifest", document.get("asset_manifest"), root)
    _validate_record("requirements", document.get("requirements"), root)
    for record_name, alias in (
        ("config", "config_sha256"),
        ("asset_manifest", "asset_manifest_sha256"),
        ("requirements", "requirements_sha256"),
    ):
        record = document[record_name]
        if document.get(alias) != record.get("sha256"):
            raise E1IntegrityError(f"inconsistent digest alias in lock: {alias}")

    inputs = document.get("input_files")
    aliases = document.get("input_file_sha256")
    if not isinstance(inputs, Mapping) or not inputs:
        raise E1IntegrityError("E1 lock contains no external input files")
    if not isinstance(aliases, Mapping):
        raise E1IntegrityError("E1 lock external-input digest index is invalid")
    if list(inputs) != sorted(inputs):
        raise E1IntegrityError("E1 lock external inputs are not deterministically ordered")
    for name, record in inputs.items():
        _validate_record(f"input_files.{name}", record, root)
        if aliases.get(name) != record.get("sha256"):
            raise E1IntegrityError(
                f"inconsistent external-input digest alias: {name}"
            )
    if set(aliases) != set(inputs):
        raise E1IntegrityError("E1 lock external-input digest index differs from records")

    expected_source = document.get("source_bundle_sha256")
    if expected_source != source_bundle_sha256(root):
        raise E1IntegrityError("E1 execution source changed after protocol freeze")
    return {
        "status": "PASS_E1_CONFIRMATION_LOCK_VALID",
        "lock_path": str(source),
        "lock_sha256": sha256_file(source),
        "config_sha256": document["config_sha256"],
        "asset_manifest_sha256": document["asset_manifest_sha256"],
        "requirements_sha256": document["requirements_sha256"],
        "source_bundle_sha256": expected_source,
        "input_file_count": len(inputs),
        "prediction_executed": False,
        "accuracy_read": False,
    }


__all__ = [
    "DEFAULT_LOCK_PATH",
    "E1IntegrityError",
    "LOCK_SCHEMA",
    "atomic_json",
    "freeze",
    "sha256_file",
    "source_bundle_sha256",
    "validate_lock",
]
