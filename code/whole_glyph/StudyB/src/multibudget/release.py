"""One-way post-50/50 truth release and Accuracy--Budget analysis.

The administrative inspection path deliberately does not open a prediction
CSV or the confirmation glyph archive.  The explicit release path first
proves that the formal session is the exact completed 50-run prefix, then asks
the formal validator to verify every committed file hash.  Only after those
gates pass does it load sample identities and labels, join pseudonymous
prediction tokens, and invoke the preregistered paired analysis.

This is a cooperative workflow-governance boundary, not a cryptographic
security boundary.  The deterministic SHA-256 sample token is de-labeling only
and can be enumerated by a party that already has the frozen glyph archive.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from src.multibudget.analysis import (
    ANALYSIS_RELEASE_STATUS,
    AnalysisResult,
    AnalysisSpec,
    REQUIRED_COLUMNS,
    analyze_predictions,
    require_analysis_release,
)
from src.multibudget.formal import (
    PREDICTION_COLUMNS,
    sample_token,
    validate_formal_lock,
    validate_formal_session,
)
from src.multibudget.protocol import (
    REQUIRED_RUN_COUNT,
    MultiBudgetProtocol,
    canonical_json_bytes,
    load_protocol,
    sha256_file,
    validate_run_plan,
)


RELEASE_SCHEMA_VERSION = 1
RELEASE_ROOT = Path("outputs/multibudget_analysis_release")
ACTIVE_SESSION = Path("outputs/multibudget_formal/ACTIVE_SESSION.json")
FORMAL_LOCK = Path("locks/multibudget_formal.lock.json")
EXPECTED_FONT_IDS = ("MB_F01", "MB_F02", "MB_F03", "MB_F04")
EXPECTED_ROWS_PER_RUN = 720
EXPECTED_MERGED_ROWS = 36_000
COMPLETED_STATUS = "COMPLETE_50_OF_50"
PREDICTION_COMMIT_STATUS = "PASS_COMMITTED_PREDICTIONS_ONLY"
ADMIN_INSPECTION_STATUS = "ADMINISTRATIVE_RELEASE_GATE_INSPECTED"


class AnalysisReleaseError(RuntimeError):
    """Raised before an unsafe or malformed post-formal release."""


@dataclass(frozen=True, slots=True)
class ReleaseInspection:
    """Prediction- and truth-blind view of the formal completion state."""

    project_root: Path
    session_id: str
    status: str
    completed_count: int
    total_count: int
    unexpected_conditions: int
    exact_prefix_verified: bool
    record_hash_chain_verified: bool
    frozen_run_ids: tuple[str, ...]
    completed_run_ids: tuple[str, ...]
    conditions: tuple[Mapping[str, Any], ...]
    completed_manifest_sha256: str
    record_chain_head_sha256: str | None

    @property
    def ready_for_release(self) -> bool:
        return (
            self.status == COMPLETED_STATUS
            and self.completed_count == REQUIRED_RUN_COUNT
            and self.total_count == REQUIRED_RUN_COUNT
            and self.unexpected_conditions == 0
            and self.exact_prefix_verified
            and self.record_hash_chain_verified
            and self.completed_run_ids == self.frozen_run_ids
        )

    def report(self) -> dict[str, Any]:
        return {
            "status": ADMIN_INSPECTION_STATUS,
            "formal_session_status": self.status,
            "session_id": self.session_id,
            "completed_count": self.completed_count,
            "total_count": self.total_count,
            "unexpected_conditions": self.unexpected_conditions,
            "exact_prefix_verified": self.exact_prefix_verified,
            "record_hash_chain_verified": self.record_hash_chain_verified,
            "ready_for_release": self.ready_for_release,
            "prediction_files_read": False,
            "truth_archive_read": False,
            "analysis_release": False,
            "threat_model": {
                "sample_token_is_de_labeling_only": True,
                "cryptographic_confidentiality": False,
                "release_gate_is_cooperative_workflow_governance": True,
                "malicious_workspace_reader_is_out_of_scope": True,
            },
            "hint": (
                "use --release only after COMPLETE_50_OF_50"
                if not self.ready_for_release
                else "explicit --release is required to read predictions and truth"
            ),
        }


@dataclass(frozen=True, slots=True)
class TruthRow:
    sample_token: str
    font_id: str
    char_id: str
    y_true: str


def _read_json(path: Path, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AnalysisReleaseError(f"non-finite JSON constant {value!r}: {path}")

    if not path.is_file():
        raise AnalysisReleaseError(f"{name} does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisReleaseError(f"{name} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise AnalysisReleaseError(f"{name} root must be an object: {path}")
    return value


def _safe_project_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisReleaseError(f"{name} must be a non-empty project-relative path")
    raw = Path(value)
    if raw.is_absolute():
        raise AnalysisReleaseError(f"{name} must be project-relative")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AnalysisReleaseError(f"{name} escapes the project root") from error
    return path


def _verify_sidecar(path: Path, sidecar: Path | None = None) -> str:
    sidecar_path = sidecar or path.with_suffix(path.suffix + ".sha256")
    try:
        fields = sidecar_path.read_text(encoding="ascii").split()
    except OSError as error:
        raise AnalysisReleaseError(
            f"missing SHA-256 sidecar: {sidecar_path}"
        ) from error
    if not fields:
        raise AnalysisReleaseError(f"empty SHA-256 sidecar: {sidecar_path}")
    digest = sha256_file(path)
    if fields[0].lower() != digest:
        raise AnalysisReleaseError(f"SHA-256 sidecar mismatch: {path}")
    return digest


def _read_completed_records(path: Path) -> tuple[dict[str, Any], ...]:
    """Verify only the administrative record chain, never its output files."""

    records: list[dict[str, Any]] = []
    previous: str | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AnalysisReleaseError(
            f"completed-run manifest is unreadable: {path}"
        ) from error
    for index, line in enumerate(lines):
        if not line:
            raise AnalysisReleaseError("completed-run manifest contains a blank line")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnalysisReleaseError(
                "completed-run manifest contains invalid JSON"
            ) from error
        if not isinstance(record, dict):
            raise AnalysisReleaseError("completed-run record must be an object")
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        digest = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        if (
            record.get("record_index") != index
            or record.get("previous_record_sha256") != previous
            or record.get("record_sha256") != digest
        ):
            raise AnalysisReleaseError("completed-run record hash chain is invalid")
        previous = digest
        records.append(record)
    return tuple(records)


def inspect_release_gate(config_path: str | Path) -> ReleaseInspection:
    """Inspect completion metadata without opening predictions or glyph truth."""

    protocol = load_protocol(config_path)
    root = protocol.project_root
    plan = _read_json(protocol.run_plan_path, "frozen run plan")
    validate_run_plan(protocol, plan)
    _verify_sidecar(protocol.run_plan_path, protocol.run_plan_sha256_path)
    raw_run_ids = plan.get("frozen_run_ids")
    conditions = plan.get("conditions")
    if not isinstance(raw_run_ids, list) or not isinstance(conditions, list):
        raise AnalysisReleaseError("frozen run plan lacks run IDs or conditions")
    frozen_run_ids = tuple(str(value) for value in raw_run_ids)
    if len(frozen_run_ids) != REQUIRED_RUN_COUNT or len(conditions) != REQUIRED_RUN_COUNT:
        raise AnalysisReleaseError("frozen run plan is not exactly 50 conditions")
    if any(not isinstance(item, Mapping) for item in conditions):
        raise AnalysisReleaseError("frozen run plan contains a malformed condition")

    pointer_path = root / ACTIVE_SESSION
    _verify_sidecar(pointer_path)
    pointer = _read_json(pointer_path, "active formal-session pointer")
    session_id = pointer.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise AnalysisReleaseError("active formal-session ID is invalid")
    expected_base = Path("outputs/multibudget_formal/sessions") / session_id
    expected_session = expected_base / "session.json"
    expected_checkpoint = expected_base / "checkpoint.json"
    expected_completed = expected_base / "completed_runs.jsonl"
    if pointer.get("session_path") != expected_session.as_posix():
        raise AnalysisReleaseError("active pointer does not bind the expected session path")

    session_path = root / expected_session
    session_sha = _verify_sidecar(session_path)
    if pointer.get("session_sha256") != session_sha:
        raise AnalysisReleaseError("active pointer session digest changed")
    session = _read_json(session_path, "formal session")
    if (
        session.get("session_id") != session_id
        or session.get("planned_run_ids") != list(frozen_run_ids)
    ):
        raise AnalysisReleaseError("formal session run-plan binding changed")
    lock_path = root / FORMAL_LOCK
    lock_sha = _verify_sidecar(
        lock_path, lock_path.with_name("multibudget_formal.lock.sha256")
    )
    if session.get("formal_lock") != {
        "path": FORMAL_LOCK.as_posix(),
        "sha256": lock_sha,
    }:
        raise AnalysisReleaseError("formal session lock binding changed")
    paths = session.get("paths")
    if not isinstance(paths, Mapping):
        raise AnalysisReleaseError("formal session paths are missing")
    if (
        paths.get("completed_runs") != expected_completed.as_posix()
        or paths.get("checkpoint") != expected_checkpoint.as_posix()
        or paths.get("formal_runs_root")
        != protocol.formal_runs_root.relative_to(root).as_posix()
    ):
        raise AnalysisReleaseError("formal session administrative paths changed")

    completed_path = root / expected_completed
    completed_sha = _verify_sidecar(completed_path)
    records = _read_completed_records(completed_path)
    completed_run_ids = tuple(str(record.get("run_id")) for record in records)
    exact_prefix = completed_run_ids == frozen_run_ids[: len(completed_run_ids)]
    if not exact_prefix:
        raise AnalysisReleaseError("completed runs are not an exact frozen-plan prefix")
    if len(records) > REQUIRED_RUN_COUNT:
        raise AnalysisReleaseError("completed-run manifest contains extra conditions")

    formal_root = protocol.formal_runs_root
    actual_dirs: set[str] = set()
    if formal_root.exists():
        if not formal_root.is_dir():
            raise AnalysisReleaseError("formal run root is not a directory")
        entries = list(formal_root.iterdir())
        if any(not entry.is_dir() for entry in entries):
            raise AnalysisReleaseError("formal run root contains unexpected files")
        actual_dirs = {entry.name for entry in entries}
    expected_dirs = set(completed_run_ids)
    unexpected = len(actual_dirs - expected_dirs) + len(expected_dirs - actual_dirs)
    if unexpected:
        raise AnalysisReleaseError(
            "formal run root contains uncommitted, missing, or extra conditions"
        )

    checkpoint_path = root / expected_checkpoint
    _verify_sidecar(checkpoint_path)
    checkpoint = _read_json(checkpoint_path, "formal checkpoint")
    chain_head = records[-1]["record_sha256"] if records else None
    next_run_id = (
        frozen_run_ids[len(records)] if len(records) < REQUIRED_RUN_COUNT else None
    )
    expected_checkpoint_fields = {
        "session_id": session_id,
        "completed_manifest_sha256": completed_sha,
        "record_chain_head_sha256": chain_head,
        "completed_count": len(records),
        "completed_run_ids": list(completed_run_ids),
        "next_index": len(records),
        "next_run_id": next_run_id,
        "total_count": REQUIRED_RUN_COUNT,
    }
    if any(
        checkpoint.get(key) != value
        for key, value in expected_checkpoint_fields.items()
    ):
        raise AnalysisReleaseError(
            "checkpoint differs from the completed-run administrative chain"
        )
    return ReleaseInspection(
        project_root=root,
        session_id=session_id,
        status=str(checkpoint.get("status", "")),
        completed_count=len(records),
        total_count=REQUIRED_RUN_COUNT,
        unexpected_conditions=0,
        exact_prefix_verified=True,
        record_hash_chain_verified=True,
        frozen_run_ids=frozen_run_ids,
        completed_run_ids=completed_run_ids,
        conditions=tuple(conditions),
        completed_manifest_sha256=completed_sha,
        record_chain_head_sha256=chain_head,
    )


def _require_complete(inspection: ReleaseInspection) -> None:
    if not inspection.ready_for_release:
        raise AnalysisReleaseError(
            "analysis release requires COMPLETE_50_OF_50 with exact 50/50 prefix "
            f"and zero extras; observed status={inspection.status!r}, "
            f"completed={inspection.completed_count}/{inspection.total_count}"
        )


def _load_truth_index(
    protocol: MultiBudgetProtocol,
) -> tuple[dict[str, TruthRow], Path, str]:
    """Open only the identity/label arrays after the 50/50 release gate."""

    manifest = _read_json(protocol.asset_manifest_path, "confirmation asset manifest")
    glyph = manifest.get("glyph_archive")
    if not isinstance(glyph, Mapping):
        raise AnalysisReleaseError("asset manifest lacks confirmation glyph archive")
    if (
        glyph.get("sample_count") != EXPECTED_ROWS_PER_RUN
        or glyph.get("class_count") != 180
        or glyph.get("font_count") != len(EXPECTED_FONT_IDS)
    ):
        raise AnalysisReleaseError("confirmation glyph dimensions changed")
    glyph_path = _safe_project_path(
        protocol.asset_manifest_path.parent,
        glyph.get("file"),
        "glyph_archive.file",
    )
    expected_sha = str(glyph.get("file_sha256", "")).lower()
    glyph_sha = sha256_file(glyph_path)
    if glyph_sha != expected_sha:
        raise AnalysisReleaseError("confirmation glyph archive SHA-256 changed")
    try:
        with np.load(glyph_path, allow_pickle=False) as archive:
            required = {"sample_ids", "class_ids", "font_ids"}
            if not required <= set(archive.files):
                raise AnalysisReleaseError(
                    "confirmation glyph archive lacks sample identity or class arrays"
                )
            sample_ids = np.asarray(archive["sample_ids"])
            class_ids = np.asarray(archive["class_ids"])
            font_ids = np.asarray(archive["font_ids"])
    except (OSError, ValueError) as error:
        raise AnalysisReleaseError(
            f"confirmation glyph archive is invalid: {glyph_path}"
        ) from error
    if (
        sample_ids.shape != (EXPECTED_ROWS_PER_RUN,)
        or class_ids.shape != (EXPECTED_ROWS_PER_RUN,)
        or font_ids.shape != (EXPECTED_ROWS_PER_RUN,)
        or class_ids.dtype != np.dtype("int64")
        or sample_ids.dtype.kind != "U"
        or font_ids.dtype.kind != "U"
    ):
        raise AnalysisReleaseError("confirmation truth array shape or dtype changed")

    index: dict[str, TruthRow] = {}
    sample_id_values = tuple(str(value) for value in sample_ids)
    if len(set(sample_id_values)) != EXPECTED_ROWS_PER_RUN:
        raise AnalysisReleaseError("confirmation sample IDs are not unique")
    for sample_id, raw_class, raw_font in zip(
        sample_id_values, class_ids.tolist(), font_ids.tolist(), strict=True
    ):
        class_id = int(raw_class)
        font_id = str(raw_font)
        if font_id not in EXPECTED_FONT_IDS or not 0 <= class_id < 180:
            raise AnalysisReleaseError("confirmation truth contains an invalid class or font")
        token = sample_token(sample_id)
        if token in index:
            raise AnalysisReleaseError("sample-token collision in confirmation truth")
        index[token] = TruthRow(
            sample_token=token,
            font_id=font_id,
            char_id=f"c{class_id:03d}",
            y_true=str(class_id),
        )
    observed_pairs = {(row.font_id, row.char_id) for row in index.values()}
    expected_pairs = {
        (font_id, f"c{class_id:03d}")
        for font_id in EXPECTED_FONT_IDS
        for class_id in range(180)
    }
    if observed_pairs != expected_pairs:
        raise AnalysisReleaseError(
            "confirmation truth is not the exact four-font by 180-class product"
        )
    return index, glyph_path, glyph_sha


def _read_prediction_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
                raise AnalysisReleaseError("formal prediction columns changed")
            return tuple(dict(row) for row in reader)
    except OSError as error:
        raise AnalysisReleaseError(f"formal prediction file is unreadable: {path}") from error


def merge_predictions_with_truth(
    prediction_batches: Iterable[
        tuple[Mapping[str, Any], Iterable[Mapping[str, Any]]]
    ],
    truth_by_token: Mapping[str, TruthRow],
    *,
    expected_condition_count: int,
) -> tuple[dict[str, Any], ...]:
    """Strictly join prediction tokens to truth and reject every imbalance."""

    rows: list[dict[str, Any]] = []
    condition_keys: set[tuple[str, int, int]] = set()
    all_tokens = set(truth_by_token)
    for condition_number, (condition, raw_rows) in enumerate(
        prediction_batches, start=1
    ):
        try:
            method = str(condition["method"])
            budget = int(condition["budget"])
            seed = int(condition["mapping_seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisReleaseError("formal condition metadata is invalid") from error
        condition_key = (method, budget, seed)
        if condition_key in condition_keys:
            raise AnalysisReleaseError(f"duplicate formal condition: {condition_key!r}")
        condition_keys.add(condition_key)
        observed_tokens: set[str] = set()
        condition_rows = list(raw_rows)
        if len(condition_rows) != len(truth_by_token):
            raise AnalysisReleaseError(
                f"condition {condition_number} prediction row count mismatch"
            )
        for row_number, raw in enumerate(condition_rows, start=2):
            if tuple(raw) != PREDICTION_COLUMNS:
                raise AnalysisReleaseError(
                    f"condition {condition_number} row {row_number} columns changed"
                )
            if (
                raw["method"] != method
                or str(raw["budget"]) != str(budget)
                or str(raw["seed"]) != str(seed)
            ):
                raise AnalysisReleaseError(
                    f"condition {condition_number} prediction binding changed"
                )
            token = str(raw["sample_token"])
            if token in observed_tokens:
                raise AnalysisReleaseError(
                    f"condition {condition_number} contains a duplicate sample token"
                )
            observed_tokens.add(token)
            truth = truth_by_token.get(token)
            if truth is None:
                raise AnalysisReleaseError(
                    f"condition {condition_number} contains an unknown sample token"
                )
            if str(raw["font_id"]) != truth.font_id:
                raise AnalysisReleaseError(
                    f"condition {condition_number} sample token has the wrong font"
                )
            try:
                y_pred = int(raw["y_pred"])
            except (TypeError, ValueError) as error:
                raise AnalysisReleaseError("predicted class is not an integer") from error
            if str(y_pred) != str(raw["y_pred"]) or not 0 <= y_pred < 180:
                raise AnalysisReleaseError("predicted class is outside the frozen domain")
            rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "seed": seed,
                    "font_id": truth.font_id,
                    "char_id": truth.char_id,
                    "y_true": truth.y_true,
                    "y_pred": str(y_pred),
                }
            )
        if observed_tokens != all_tokens:
            raise AnalysisReleaseError(
                f"condition {condition_number} has missing or extra sample tokens"
            )
    if len(condition_keys) != expected_condition_count:
        raise AnalysisReleaseError(
            "formal prediction batch count mismatch: "
            f"expected {expected_condition_count}, observed {len(condition_keys)}"
        )
    if len(rows) != expected_condition_count * len(truth_by_token):
        raise AnalysisReleaseError("merged formal prediction row count mismatch")
    rows.sort(
        key=lambda row: (
            str(row["method"]),
            int(row["budget"]),
            int(row["seed"]),
            str(row["font_id"]),
            str(row["char_id"]),
        )
    )
    return tuple(rows)


def _prediction_batches(
    protocol: MultiBudgetProtocol, conditions: Sequence[Mapping[str, Any]]
) -> Iterable[tuple[Mapping[str, Any], tuple[dict[str, str], ...]]]:
    for condition in conditions:
        run_id = condition.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise AnalysisReleaseError("formal condition lacks a run ID")
        output = protocol.formal_runs_root / run_id
        manifest = _read_json(output / "run_manifest.json", "formal run manifest")
        if (
            manifest.get("status") != PREDICTION_COMMIT_STATUS
            or manifest.get("run_id") != run_id
            or manifest.get("prediction_row_count") != EXPECTED_ROWS_PER_RUN
        ):
            raise AnalysisReleaseError("formal run manifest release binding changed")
        prediction_path = output / "predictions.csv"
        if manifest.get("predictions_sha256") != sha256_file(prediction_path):
            raise AnalysisReleaseError("formal prediction digest changed during release")
        yield condition, _read_prediction_csv(prediction_path)


def _analysis_spec(protocol: MultiBudgetProtocol) -> AnalysisSpec:
    analysis = protocol.analysis
    return AnalysisSpec(
        reference_method=protocol.methods[0],
        treatment_method=protocol.methods[1],
        budgets=protocol.budgets,
        seeds=protocol.mapping_seeds,
        font_ids=EXPECTED_FONT_IDS,
        char_ids=tuple(f"c{value:03d}" for value in range(protocol.class_count)),
        reference_budget=64,
        bootstrap_replicates=int(analysis["bootstrap_iterations"]),
        bootstrap_seed=int(analysis["bootstrap_seed"]),
        confidence_level=0.95,
    )


def _interval(value: Any) -> dict[str, float]:
    return {
        "point": float(value.point),
        "lower": float(value.lower),
        "upper": float(value.upper),
    }


def _serialize_result(result: AnalysisResult) -> dict[str, Any]:
    bootstrap: dict[str, Any] | None = None
    if result.bootstrap is not None:
        bootstrap = {
            "cluster": result.bootstrap.cluster,
            "replicates": result.bootstrap.replicates,
            "seed": result.bootstrap.seed,
            "confidence_level": result.bootstrap.confidence_level,
            "accuracy": [
                {
                    "method": method,
                    "budget": budget,
                    **_interval(interval),
                }
                for (method, budget), interval in sorted(
                    result.bootstrap.accuracy.items()
                )
            ],
            "delta": [
                {"budget": budget, **_interval(interval)}
                for budget, interval in sorted(result.bootstrap.delta.items())
            ],
            "naubc": [
                {"method": method, **_interval(interval)}
                for method, interval in sorted(result.bootstrap.naubc.items())
            ],
            "delta_naubc": _interval(result.bootstrap.delta_naubc),
        }
    return {
        "schema": "mma-multibudget-v1-analysis-results",
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": ANALYSIS_RELEASE_STATUS,
        "analysis_release": True,
        "primary_endpoint": "delta_normalized_area_under_accuracy_budget_curve",
        "primary_success_rule": "ci95_lower_bound_strictly_greater_than_zero",
        "primary_success": (
            result.bootstrap is not None
            and result.bootstrap.delta_naubc.lower > 0.0
        ),
        "accuracy_budget": [asdict(value) for value in result.accuracy],
        "delta_budget": [asdict(value) for value in result.delta],
        "naubc": [asdict(value) for value in result.naubc],
        "delta_naubc": result.delta_naubc,
        "b_eq64": asdict(result.b_eq64),
        "bootstrap": bootstrap,
        "threat_model": {
            "sample_token_is_de_labeling_only": True,
            "cryptographic_confidentiality": False,
            "release_gate_is_cooperative_workflow_governance": True,
            "malicious_workspace_reader_is_out_of_scope": True,
        },
    }


def _csv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_release_root(
    protocol: MultiBudgetProtocol, output_dir: str | Path | None
) -> Path:
    value = RELEASE_ROOT if output_dir is None else Path(output_dir)
    path = value.resolve() if value.is_absolute() else (protocol.project_root / value).resolve()
    try:
        path.relative_to(protocol.project_root)
    except ValueError as error:
        raise AnalysisReleaseError("analysis release output must stay in the project") from error
    return path


def _write_release(
    final_root: Path,
    inspection: ReleaseInspection,
    protocol: MultiBudgetProtocol,
    glyph_path: Path,
    glyph_sha: str,
    lock_report: Mapping[str, Any],
    merged: Sequence[Mapping[str, Any]],
    result: AnalysisResult,
) -> dict[str, Any]:
    if final_root.exists():
        raise AnalysisReleaseError(
            f"analysis release output exists; refusing overwrite: {final_root}"
        )
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = final_root.parent / f".{final_root.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    try:
        merged_path = staging / "merged_predictions_with_truth.csv"
        _write_new(merged_path, _csv_bytes(REQUIRED_COLUMNS, merged))
        result_document = _serialize_result(result)
        results_path = staging / "main_results.json"
        _write_new(results_path, canonical_json_bytes(result_document))

        accuracy_rows = []
        if result.bootstrap is None:
            raise AnalysisReleaseError("frozen release requires bootstrap intervals")
        for point in result.accuracy:
            interval = result.bootstrap.accuracy[(point.method, point.budget)]
            accuracy_rows.append({**asdict(point), "ci_lower": interval.lower, "ci_upper": interval.upper})
        accuracy_path = staging / "accuracy_budget.csv"
        _write_new(
            accuracy_path,
            _csv_bytes(
                (
                    "method",
                    "budget",
                    "n_samples",
                    "n_correct",
                    "accuracy",
                    "ci_lower",
                    "ci_upper",
                ),
                accuracy_rows,
            ),
        )
        delta_rows = []
        for point in result.delta:
            interval = result.bootstrap.delta[point.budget]
            delta_rows.append({**asdict(point), "ci_lower": interval.lower, "ci_upper": interval.upper})
        delta_path = staging / "delta_budget.csv"
        _write_new(
            delta_path,
            _csv_bytes(
                (
                    "budget",
                    "n_pairs",
                    "reference_accuracy",
                    "treatment_accuracy",
                    "delta_accuracy",
                    "ci_lower",
                    "ci_upper",
                ),
                delta_rows,
            ),
        )
        naubc_rows = []
        for point in result.naubc:
            interval = result.bootstrap.naubc[point.method]
            naubc_rows.append(
                {
                    **asdict(point),
                    "ci_lower": interval.lower,
                    "ci_upper": interval.upper,
                }
            )
        _write_new(
            staging / "naubc.csv",
            _csv_bytes(
                ("method", "naubc", "ci_lower", "ci_upper"), naubc_rows
            ),
        )
        artifacts = []
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            artifacts.append(
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        release_manifest = {
            "schema": "mma-multibudget-v1-analysis-release",
            "schema_version": RELEASE_SCHEMA_VERSION,
            "status": ANALYSIS_RELEASE_STATUS,
            "analysis_release": True,
            "threat_model": {
                "sample_token_purpose": "de_labeling_only",
                "deterministic_sha256_token_is_enumeration_resistant": False,
                "cryptographic_confidentiality": False,
                "release_gate_security_boundary": False,
                "governance_model": (
                    "cooperative_workflow_gate_for_the_official_analysis_entrypoint"
                ),
                "malicious_workspace_file_access": "out_of_scope",
            },
            "expected_conditions": REQUIRED_RUN_COUNT,
            "completed_conditions": REQUIRED_RUN_COUNT,
            "unexpected_conditions": 0,
            "hashes_verified": True,
            "formal_run_hashes_verified": True,
            "exact_prefix_verified": True,
            "condition_statuses": ["PASS_COMMITTED"] * REQUIRED_RUN_COUNT,
            "run_ids": list(inspection.completed_run_ids),
            "frozen_run_ids": list(inspection.frozen_run_ids),
            "formal_session_id": inspection.session_id,
            "formal_lock_sha256": lock_report["lock_sha256"],
            "completed_manifest_sha256": inspection.completed_manifest_sha256,
            "record_chain_head_sha256": inspection.record_chain_head_sha256,
            "glyph_archive": {
                "path": glyph_path.relative_to(protocol.project_root).as_posix(),
                "sha256": glyph_sha,
                "truth_arrays_read_after_50_of_50_gate": [
                    "sample_ids",
                    "class_ids",
                    "font_ids",
                ],
            },
            "merged_prediction_rows": len(merged),
            "artifacts": artifacts,
        }
        require_analysis_release(release_manifest)
        manifest_path = staging / "release_manifest.json"
        _write_new(manifest_path, canonical_json_bytes(release_manifest))
        manifest_sha = sha256_file(manifest_path)
        _write_new(
            staging / "release_manifest.sha256",
            f"{manifest_sha}  release_manifest.json\n".encode("ascii"),
        )
        if final_root.exists():
            raise AnalysisReleaseError(
                f"analysis release output appeared; refusing overwrite: {final_root}"
            )
        os.rename(staging, final_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": ANALYSIS_RELEASE_STATUS,
        "analysis_release": True,
        "output_root": final_root.relative_to(protocol.project_root).as_posix(),
        "release_manifest": (
            final_root / "release_manifest.json"
        ).relative_to(protocol.project_root).as_posix(),
        "release_manifest_sha256": sha256_file(final_root / "release_manifest.json"),
        "merged_prediction_rows": len(merged),
        "completed_conditions": REQUIRED_RUN_COUNT,
        "unexpected_conditions": 0,
    }


def release_multibudget_analysis(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Explicitly release truth and atomically compute the frozen analysis."""

    inspection = inspect_release_gate(config_path)
    _require_complete(inspection)
    protocol = load_protocol(config_path)
    final_root = _resolve_release_root(protocol, output_dir)
    if final_root.exists():
        raise AnalysisReleaseError(
            f"analysis release output exists; refusing overwrite: {final_root}"
        )

    # This public formal validation re-hashes and structurally verifies all 50
    # committed prediction files.  It runs only after the administrative 50/50
    # gate and before the truth archive is opened.
    lock_report = validate_formal_lock(config_path)
    formal_report = validate_formal_session(config_path)
    if (
        formal_report.get("status") != COMPLETED_STATUS
        or formal_report.get("completed_count") != REQUIRED_RUN_COUNT
        or formal_report.get("remaining_count") != 0
        or formal_report.get("unexpected_conditions") != 0
        or formal_report.get("exact_prefix_verified") is not True
        or formal_report.get("hash_chain_verified") is not True
    ):
        raise AnalysisReleaseError("formal validator did not prove exact COMPLETE_50_OF_50")

    truth, glyph_path, glyph_sha = _load_truth_index(protocol)
    merged = merge_predictions_with_truth(
        _prediction_batches(protocol, inspection.conditions),
        truth,
        expected_condition_count=REQUIRED_RUN_COUNT,
    )
    if len(merged) != EXPECTED_MERGED_ROWS:
        raise AnalysisReleaseError("formal merged prediction count is not exactly 36000")
    spec = _analysis_spec(protocol)
    result = analyze_predictions(merged, spec, include_bootstrap=True)
    return _write_release(
        final_root,
        inspection,
        protocol,
        glyph_path,
        glyph_sha,
        lock_report,
        merged,
        result,
    )


__all__ = [
    "ADMIN_INSPECTION_STATUS",
    "AnalysisReleaseError",
    "ReleaseInspection",
    "TruthRow",
    "inspect_release_gate",
    "merge_predictions_with_truth",
    "release_multibudget_analysis",
]
