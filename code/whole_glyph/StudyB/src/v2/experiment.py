"""Auditable CPU-only experiment runner for calibration-aware Hanzi SPV."""

from __future__ import annotations

import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from src.classifier import LinearSvcClassifier, classification_accuracy
from src.contracts import ElectrodeGrid, ElectrodeMapping
from src.deterministic_io import array_sha256, sha256_file, write_deterministic_npz
from src.encoders.skeleton_topk import SkeletonTopKEncoder
from src.features import HogFeatureExtractor
from src.grid import make_regular_grid
from src.spv_forward import (
    GaussianPhospheneForwardModel,
    IdealGaussianUtilityProjector,
    expected_output_energy,
)
from src.v2.calibrated_skeleton_coverage import (
    CalibrationAwareSkeletonCoverageEncoder,
)
from src.v2.calibrated_skeleton_topk import MappingAwareSkeletonTopKEncoder
from src.v2.mapped_projector import MappingAwareGaussianUtilityProjector


ALLOWED_METHODS = (
    "skeleton_topk",
    "calibrated_skeleton_topk",
    "nominal_skeleton_coverage",
    "ca_skelcov",
)
ALLOWED_BUDGETS = (32, 48, 64, 96, 128)
EXPECTED_IMAGE_SHAPE = (96, 96)
EXPECTED_CANDIDATES = 256


class ExperimentError(ValueError):
    """Raised when an experiment input or audit condition is invalid."""


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    method: str
    budget: int
    mapping_type: str
    mapping_seed: int

    def __post_init__(self) -> None:
        if self.method not in ALLOWED_METHODS:
            raise ExperimentError(f"unsupported method: {self.method}")
        if self.budget not in ALLOWED_BUDGETS:
            raise ExperimentError(f"budget must be one of {ALLOWED_BUDGETS}")
        if self.mapping_type not in {"ideal", "moderate"}:
            raise ExperimentError("mapping_type must be ideal or moderate")
        if self.mapping_type == "ideal" and self.mapping_seed != 0:
            raise ExperimentError("ideal mapping requires seed 0")
        if self.mapping_type == "moderate" and self.mapping_seed <= 0:
            raise ExperimentError("moderate mapping requires a positive seed")

    def run_id(self, experiment_id: str) -> str:
        return (
            f"{experiment_id}_{self.mapping_type}_s{self.mapping_seed}_"
            f"{self.method}_b{self.budget:03d}"
        )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    source_path: Path
    project_root: Path
    experiment_id: str
    legacy_root: Path
    dataset_path: Path
    expected_dataset_sha256: str | None
    mapping_dir: Path
    expected_mapping_sha256: Mapping[str, str]
    output_root: Path
    train_fonts: tuple[str, ...]
    test_fonts: tuple[str, ...]
    feature: Mapping[str, Any]
    classifier: Mapping[str, Any]
    save_spv: bool
    max_blas_threads: int
    report_metrics: bool
    enforce_environment: bool
    expected_environment: Mapping[str, str]
    conditions: tuple[ConditionSpec, ...]


@dataclass(frozen=True, slots=True)
class GlyphDataset:
    images: NDArray[np.float32]
    class_ids: NDArray[np.int64]
    sample_ids: NDArray[np.str_]
    characters: NDArray[np.str_]
    font_ids: NDArray[np.str_]
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentError(f"{name} must be a mapping")
    return value


def _require_string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ExperimentError(f"{name} must be a non-empty list")
    result = tuple(map(str, value))
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ExperimentError(f"{name} contains empty or duplicate values")
    return result


def _resolve_under(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{name} must be a non-empty path string")
    raw = value.strip()
    if (
        os.name != "nt"
        and len(raw) >= 3
        and raw[1] == ":"
        and raw[2] in {"\\", "/"}
    ):
        drive = raw[0].lower()
        remainder = raw[3:].replace("\\", "/")
        candidate = Path("/mnt") / drive / remainder
    else:
        candidate = Path(raw)
    return candidate if candidate.is_absolute() else root / candidate


def _optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ExperimentError(f"{name} must be a lowercase SHA-256 hex digest")
    return digest


def load_experiment_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    """Load and validate one E0/E1 YAML plan without reading image pixels."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ExperimentError(f"config file does not exist: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    document = _require_mapping(raw, "config")
    if int(document.get("schema_version", -1)) != 1:
        raise ExperimentError("schema_version must equal 1")
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    experiment_id = str(document.get("experiment_id", source.stem)).strip()
    if not experiment_id or not all(c.isalnum() or c in "-_" for c in experiment_id):
        raise ExperimentError("experiment_id must be filesystem-safe")
    legacy_root = _resolve_under(root, document.get("legacy_root"), "legacy_root")
    dataset_cfg = _require_mapping(document.get("dataset"), "dataset")
    mapping_cfg = _require_mapping(document.get("mapping"), "mapping")
    feature_cfg = _require_mapping(document.get("feature"), "feature")
    classifier_cfg = _require_mapping(document.get("classifier"), "classifier")
    runtime_cfg = _require_mapping(document.get("runtime", {}), "runtime")
    environment_cfg = _require_mapping(document.get("environment", {}), "environment")
    dataset_path = _resolve_under(
        legacy_root, dataset_cfg.get("glyphs_npz"), "dataset.glyphs_npz"
    )
    mapping_dir = _resolve_under(
        legacy_root, mapping_cfg.get("directory"), "mapping.directory"
    )
    expected_dataset_sha256 = _optional_sha256(
        dataset_cfg.get("sha256"), "dataset.sha256"
    )
    raw_mapping_hashes = _require_mapping(
        mapping_cfg.get("sha256", {}), "mapping.sha256"
    )
    expected_mapping_sha256 = {
        str(filename): _optional_sha256(value, f"mapping.sha256.{filename}") or ""
        for filename, value in raw_mapping_hashes.items()
    }
    output_root = _resolve_under(root, document.get("output_root"), "output_root")
    train_fonts = _require_string_list(dataset_cfg.get("train_fonts"), "train_fonts")
    test_fonts = _require_string_list(dataset_cfg.get("test_fonts"), "test_fonts")
    if set(train_fonts) & set(test_fonts):
        raise ExperimentError("train and test font lists overlap")
    raw_conditions = document.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ExperimentError("conditions must be a non-empty list")
    conditions: list[ConditionSpec] = []
    for index, item in enumerate(raw_conditions):
        condition = _require_mapping(item, f"conditions[{index}]")
        conditions.append(
            ConditionSpec(
                method=str(condition.get("method", "")),
                budget=int(condition.get("budget", -1)),
                mapping_type=str(condition.get("mapping_type", "")),
                mapping_seed=int(condition.get("mapping_seed", -1)),
            )
        )
    run_ids = [item.run_id(experiment_id) for item in conditions]
    if len(set(run_ids)) != len(run_ids):
        raise ExperimentError("conditions contain duplicate run IDs")
    save_spv = runtime_cfg.get("save_spv", False)
    if not isinstance(save_spv, bool):
        raise ExperimentError("runtime.save_spv must be true or false")
    report_metrics = runtime_cfg.get("report_metrics", True)
    if not isinstance(report_metrics, bool):
        raise ExperimentError("runtime.report_metrics must be true or false")
    max_blas_threads = int(runtime_cfg.get("max_blas_threads", 1))
    if max_blas_threads <= 0:
        raise ExperimentError("runtime.max_blas_threads must be positive")
    enforce_environment = environment_cfg.get("enforce", False)
    if not isinstance(enforce_environment, bool):
        raise ExperimentError("environment.enforce must be true or false")
    expected_environment: dict[str, str] = {}
    if "python" in environment_cfg:
        expected_environment["python"] = str(environment_cfg["python"])
    raw_packages = _require_mapping(environment_cfg.get("packages", {}), "environment.packages")
    expected_environment.update(
        {f"package:{name}": str(version) for name, version in raw_packages.items()}
    )
    if enforce_environment and not expected_environment:
        raise ExperimentError("enforced environment must specify Python or package versions")
    return ExperimentConfig(
        source_path=source,
        project_root=root,
        experiment_id=experiment_id,
        legacy_root=legacy_root,
        dataset_path=dataset_path,
        expected_dataset_sha256=expected_dataset_sha256,
        mapping_dir=mapping_dir,
        expected_mapping_sha256=expected_mapping_sha256,
        output_root=output_root,
        train_fonts=train_fonts,
        test_fonts=test_fonts,
        feature=dict(feature_cfg),
        classifier=dict(classifier_cfg),
        save_spv=save_spv,
        max_blas_threads=max_blas_threads,
        report_metrics=report_metrics,
        enforce_environment=enforce_environment,
        expected_environment=expected_environment,
        conditions=tuple(conditions),
    )


def validate_plan_files(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Validate all external files needed by the selected condition plan."""

    validate_runtime_environment(config)
    if not config.legacy_root.is_dir():
        raise ExperimentError(f"legacy_root does not exist: {config.legacy_root}")
    if not config.dataset_path.is_file():
        raise ExperimentError(f"glyph archive does not exist: {config.dataset_path}")
    ideal = config.mapping_dir / "ideal_mapping.npz"
    if not ideal.is_file():
        raise ExperimentError(f"ideal mapping does not exist: {ideal}")
    if (
        config.expected_dataset_sha256 is not None
        and sha256_file(config.dataset_path) != config.expected_dataset_sha256
    ):
        raise ExperimentError("glyph archive SHA-256 differs from the frozen config")
    checked_mapping_paths: set[Path] = {ideal}
    plan: list[dict[str, Any]] = []
    for condition in config.conditions:
        mapping_path = mapping_path_for_condition(config, condition)
        if not mapping_path.is_file():
            raise ExperimentError(f"mapping file does not exist: {mapping_path}")
        checked_mapping_paths.add(mapping_path)
        plan.append(
            {
                "run_id": condition.run_id(config.experiment_id),
                "method": condition.method,
                "budget": condition.budget,
                "mapping_type": condition.mapping_type,
                "mapping_seed": condition.mapping_seed,
                "mapping_path": str(mapping_path),
                "output_path": str(
                    config.output_root / condition.run_id(config.experiment_id)
                ),
            }
        )
    if config.expected_mapping_sha256:
        for mapping_path in sorted(checked_mapping_paths):
            expected = config.expected_mapping_sha256.get(mapping_path.name)
            if expected is None:
                raise ExperimentError(
                    f"frozen mapping SHA-256 is missing for {mapping_path.name}"
                )
            if sha256_file(mapping_path) != expected:
                raise ExperimentError(
                    f"mapping SHA-256 differs from config: {mapping_path.name}"
                )
    return plan


def mapping_path_for_condition(
    config: ExperimentConfig, condition: ConditionSpec
) -> Path:
    if condition.mapping_type == "ideal":
        return config.mapping_dir / "ideal_mapping.npz"
    return config.mapping_dir / f"moderate_seed_{condition.mapping_seed}.npz"


def _runtime_identity() -> dict[str, Any]:
    packages = {
        name: importlib_metadata.version(name)
        for name in (
            "numpy",
            "scipy",
            "scikit-image",
            "scikit-learn",
            "PyYAML",
            "threadpoolctl",
        )
    }
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _runtime_fingerprint(identity: Mapping[str, Any] | None = None) -> str:
    document = dict(identity) if identity is not None else _runtime_identity()
    payload = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_runtime_environment(config: ExperimentConfig) -> dict[str, Any]:
    """Validate configured version pins and return the current identity."""

    identity = _runtime_identity()
    if not config.enforce_environment:
        return identity
    mismatches: list[str] = []
    for key, expected in config.expected_environment.items():
        if key == "python":
            actual = str(identity["python"])
        elif key.startswith("package:"):
            package = key.split(":", 1)[1]
            actual = str(identity["packages"].get(package, "MISSING"))
        else:
            raise ExperimentError(f"unsupported environment binding: {key}")
        if actual != expected:
            mismatches.append(f"{key} expected {expected}, found {actual}")
    if mismatches:
        raise ExperimentError("environment mismatch: " + "; ".join(mismatches))
    return identity


def load_glyph_dataset(config: ExperimentConfig) -> GlyphDataset:
    with np.load(config.dataset_path, allow_pickle=False) as archive:
        required = {"images", "class_ids", "sample_ids", "characters", "font_ids"}
        if not required <= set(archive.files):
            raise ExperimentError("glyph archive is missing required arrays")
        images = np.ascontiguousarray(archive["images"], dtype=np.float32)
        class_ids = np.ascontiguousarray(archive["class_ids"], dtype=np.int64)
        sample_ids = np.asarray(archive["sample_ids"], dtype=np.str_).copy()
        characters = np.asarray(archive["characters"], dtype=np.str_).copy()
        font_ids = np.asarray(archive["font_ids"], dtype=np.str_).copy()
    count = images.shape[0]
    if images.ndim != 3 or images.shape[1:] != EXPECTED_IMAGE_SHAPE:
        raise ExperimentError("glyph images must have shape (N, 96, 96)")
    if any(array.shape != (count,) for array in (class_ids, sample_ids, characters, font_ids)):
        raise ExperimentError("glyph metadata lengths differ from the image count")
    if count == 0 or not np.isfinite(images).all():
        raise ExperimentError("glyph image archive is empty or non-finite")
    if float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise ExperimentError("glyph image values must lie in [0, 1]")
    if np.any(images.max(axis=(1, 2)) <= 0.0):
        raise ExperimentError("glyph archive contains an empty image")
    if len(set(map(str, sample_ids))) != count:
        raise ExperimentError("sample IDs must be unique")
    train_mask = np.isin(font_ids, config.train_fonts)
    test_mask = np.isin(font_ids, config.test_fonts)
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise ExperimentError("font split overlaps or omits samples")
    train_indices = np.flatnonzero(train_mask).astype(np.int64)
    test_indices = np.flatnonzero(test_mask).astype(np.int64)
    if train_indices.size == 0 or test_indices.size == 0:
        raise ExperimentError("font split must contain train and test samples")
    train_classes = set(map(int, class_ids[train_indices]))
    test_classes = set(map(int, class_ids[test_indices]))
    if train_classes != test_classes or len(train_classes) < 2:
        raise ExperimentError("train/test splits must contain the same classes")
    return GlyphDataset(
        images=images,
        class_ids=class_ids,
        sample_ids=sample_ids,
        characters=characters,
        font_ids=font_ids,
        train_indices=np.ascontiguousarray(train_indices),
        test_indices=np.ascontiguousarray(test_indices),
    )


def load_mapping(path: Path, *, condition: ConditionSpec) -> ElectrodeMapping:
    with np.load(path, allow_pickle=False) as archive:
        required = {"mapping_id", "seed", "image_shape", "centers_xy", "sigma_px"}
        if not required <= set(archive.files):
            raise ExperimentError(f"mapping archive is incomplete: {path}")
        mapping_id = str(np.asarray(archive["mapping_id"])[0])
        seed = int(np.asarray(archive["seed"])[0])
        image_shape = tuple(map(int, np.asarray(archive["image_shape"]).tolist()))
        centers = np.ascontiguousarray(archive["centers_xy"], dtype=np.float32)
        sigma = np.ascontiguousarray(archive["sigma_px"], dtype=np.float32)
    expected_id = "ideal" if condition.mapping_type == "ideal" else f"moderate_seed{condition.mapping_seed}"
    if seed != condition.mapping_seed or mapping_id != expected_id:
        raise ExperimentError("mapping identity differs from the condition")
    if image_shape != EXPECTED_IMAGE_SHAPE or centers.shape != (EXPECTED_CANDIDATES, 2):
        raise ExperimentError("mapping geometry differs from the 96x96/256 protocol")
    return ElectrodeMapping(
        mapping_id=mapping_id,
        seed=seed,
        centers_xy=centers,
        sigma_px=sigma,
        image_shape=EXPECTED_IMAGE_SHAPE,
    )


def load_ideal_mapping(path: Path) -> ElectrodeMapping:
    return load_mapping(
        path,
        condition=ConditionSpec(
            method="skeleton_topk",
            budget=32,
            mapping_type="ideal",
            mapping_seed=0,
        ),
    )


def _build_feature_and_classifier(
    config: ExperimentConfig,
) -> tuple[HogFeatureExtractor, LinearSvcClassifier]:
    pixels = tuple(map(int, config.feature.get("pixels_per_cell", (8, 8))))
    cells = tuple(map(int, config.feature.get("cells_per_block", (2, 2))))
    extractor = HogFeatureExtractor(
        image_shape=EXPECTED_IMAGE_SHAPE,
        orientations=int(config.feature.get("orientations", 9)),
        pixels_per_cell=pixels,
        cells_per_block=cells,
        block_norm=str(config.feature.get("block_norm", "L2-Hys")),
    )
    decoder = LinearSvcClassifier(
        c=float(config.classifier.get("c", 1.0)),
        dual=str(config.classifier.get("dual", "auto")),
        max_iter=int(config.classifier.get("max_iter", 5000)),
        random_state=int(config.classifier.get("random_state", 2026)),
    )
    return extractor, decoder


def _encode_all(
    dataset: GlyphDataset,
    grid: ElectrodeGrid,
    *,
    condition: ConditionSpec,
    ideal_mapping: ElectrodeMapping,
    target_mapping: ElectrodeMapping,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    dict[str, NDArray[Any]],
    dict[str, Any],
]:
    ideal_forward = GaussianPhospheneForwardModel(ideal_mapping)
    target_forward = GaussianPhospheneForwardModel(target_mapping)
    ideal_projector = IdealGaussianUtilityProjector(ideal_mapping)
    mapped_ideal_projector = MappingAwareGaussianUtilityProjector(ideal_forward)
    mapped_target_projector = MappingAwareGaussianUtilityProjector(target_forward)
    if condition.method == "skeleton_topk":
        encoder: Any = SkeletonTopKEncoder()
        projector: Any = ideal_projector
    elif condition.method == "calibrated_skeleton_topk":
        encoder = MappingAwareSkeletonTopKEncoder()
        projector = mapped_target_projector
    elif condition.method == "nominal_skeleton_coverage":
        encoder = CalibrationAwareSkeletonCoverageEncoder(ideal_forward)
        projector = mapped_ideal_projector
    else:
        encoder = CalibrationAwareSkeletonCoverageEncoder(target_forward)
        projector = mapped_target_projector

    count = dataset.images.shape[0]
    selected = np.empty((count, condition.budget), dtype=np.int64)
    scores = np.empty((count, EXPECTED_CANDIDATES), dtype=np.float32)
    is_coverage = isinstance(encoder, CalibrationAwareSkeletonCoverageEncoder)
    coverage_order = (
        np.empty((count, 128), dtype=np.int64) if is_coverage else None
    )
    coverage_objective = (
        np.empty((count, 129), dtype=np.float64) if is_coverage else None
    )
    coverage_marginal = (
        np.empty((count, 128), dtype=np.float64) if is_coverage else None
    )
    metadata_keys: set[str] | None = None
    for index, image in enumerate(dataset.images):
        if is_coverage:
            full = encoder.encode_full_order(image, grid, projector=projector)
            selected[index] = full.order[: condition.budget]
            scores[index] = full.candidate_scores
            assert coverage_order is not None
            assert coverage_objective is not None
            assert coverage_marginal is not None
            coverage_order[index] = full.order
            coverage_objective[index] = full.trace.objective_values
            coverage_marginal[index] = full.trace.marginal_gains
            keys = {
                "mapping_id",
                "binarization_threshold",
                "skeletonization",
                "drive_map",
                "utility_normalization",
                "candidate_projection",
                "overlap",
                "selection",
                "tie_break",
                "greedy_order_length",
                "budget_prefix",
                "objective_at_budget",
                "theoretical_optimality_claim",
                "topology_enhancement",
            }
        else:
            result = encoder.encode(
                image,
                grid,
                condition.budget,
                projector=projector,
            )
            selected[index] = result.selected_indices
            scores[index] = result.candidate_scores
            keys = set(map(str, result.metadata.keys()))
        if metadata_keys is None:
            metadata_keys = keys
        elif keys != metadata_keys:
            raise ExperimentError("encoder metadata schema changed between samples")
    if any(np.unique(row).size != condition.budget for row in selected):
        raise ExperimentError("a sample contains duplicate selected electrodes")
    if np.any(selected < 0) or np.any(selected >= EXPECTED_CANDIDATES):
        raise ExperimentError("a selection contains an out-of-range electrode")
    trace_arrays: dict[str, NDArray[Any]] = {}
    trace_hashes: dict[str, str] = {}
    if is_coverage:
        assert coverage_order is not None
        assert coverage_objective is not None
        assert coverage_marginal is not None
        trace_arrays = {
            "coverage_order_128": np.ascontiguousarray(coverage_order),
            "coverage_objective_values": np.ascontiguousarray(coverage_objective),
            "coverage_marginal_gains": np.ascontiguousarray(coverage_marginal),
        }
        trace_hashes = {
            name: array_sha256(array) for name, array in trace_arrays.items()
        }
    return (
        np.ascontiguousarray(selected),
        np.ascontiguousarray(scores),
        trace_arrays,
        {
            "method": condition.method,
            "utility_mapping_id": projector.mapping_id,
            "coverage_mapping_id": (
                encoder.mapping_id
                if isinstance(encoder, CalibrationAwareSkeletonCoverageEncoder)
                else None
            ),
            "encoder_metadata_keys": sorted(metadata_keys or ()),
            "unique_and_in_range": True,
            "selections_sha256": array_sha256(selected),
            "candidate_scores_sha256": array_sha256(scores),
            "coverage_trace_sha256": trace_hashes,
        },
    )


def _source_tree_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((project_root / "src").rglob("*.py"))
    entrypoint = project_root / "run_calibrated_experiment.py"
    if entrypoint.is_file():
        paths.append(entrypoint)
    if not paths:
        raise ExperimentError(f"no Python source files found under {project_root}")
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _json_write(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_predictions(
    path: Path,
    dataset: GlyphDataset,
    predictions: NDArray[np.int64],
    *,
    condition: ConditionSpec,
    run_id: str,
) -> None:
    fields = (
        "run_id",
        "method",
        "budget",
        "mapping_type",
        "mapping_seed",
        "sample_id",
        "character",
        "font_id",
        "true_label",
        "predicted_label",
        "correct",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source_index, prediction in zip(
            dataset.test_indices, predictions, strict=True
        ):
            truth = int(dataset.class_ids[source_index])
            predicted = int(prediction)
            writer.writerow(
                {
                    "run_id": run_id,
                    "method": condition.method,
                    "budget": condition.budget,
                    "mapping_type": condition.mapping_type,
                    "mapping_seed": condition.mapping_seed,
                    "sample_id": str(dataset.sample_ids[source_index]),
                    "character": str(dataset.characters[source_index]),
                    "font_id": str(dataset.font_ids[source_index]),
                    "true_label": truth,
                    "predicted_label": predicted,
                    "correct": int(truth == predicted),
                }
            )


def _environment_document(
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = dict(identity) if identity is not None else _runtime_identity()
    return runtime | {
        "python_build": sys.version,
        "fingerprint_sha256": _runtime_fingerprint(runtime),
        "pid": os.getpid(),
    }


def _safe_replace_existing(run_dir: Path, output_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = output_root.resolve()
    if resolved_run.parent != resolved_root or not resolved_run.name:
        raise ExperimentError("refusing to remove a path outside output_root")
    shutil.rmtree(resolved_run)


def validate_completed_result(
    config: ExperimentConfig,
    condition: ConditionSpec,
    *,
    require_current_source: bool = True,
) -> dict[str, Any]:
    """Validate a committed result before treating it as resumable work."""

    run_id = condition.run_id(config.experiment_id)
    run_dir = config.output_root / run_id
    required = {
        "selections.npz",
        "predictions.csv",
        "metrics.json",
        "manifest.json",
    }
    if not run_dir.is_dir():
        raise ExperimentError(f"completed result directory is missing: {run_dir}")
    names = {path.name for path in run_dir.iterdir() if path.is_file()}
    if not required <= names:
        raise ExperimentError(f"completed result is missing required files: {run_dir}")
    try:
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ExperimentError(f"completed result metadata is unreadable: {run_dir}") from error
    if manifest.get("status") != "PASS_COMMITTED" or metrics.get("status") != "PASS_COMMITTED":
        raise ExperimentError(f"completed result does not have PASS_COMMITTED: {run_dir}")
    if manifest.get("run_id") != run_id or metrics.get("run_id") != run_id:
        raise ExperimentError(f"completed result run_id differs from the plan: {run_dir}")
    expected_condition = {
        "method": condition.method,
        "budget": condition.budget,
        "mapping_type": condition.mapping_type,
        "mapping_seed": condition.mapping_seed,
    }
    if manifest.get("condition") != expected_condition:
        raise ExperimentError(f"completed result condition differs from the plan: {run_dir}")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ExperimentError(f"completed result has no hash document: {run_dir}")
    target_path = mapping_path_for_condition(config, condition)
    current_inputs = {
        "config_sha256": sha256_file(config.source_path),
        "dataset_sha256": sha256_file(config.dataset_path),
        "ideal_mapping_sha256": sha256_file(config.mapping_dir / "ideal_mapping.npz"),
        "target_mapping_sha256": sha256_file(target_path),
    }
    for key, value in current_inputs.items():
        if hashes.get(key) != value:
            raise ExperimentError(f"completed result input hash changed ({key}): {run_dir}")
    current_source = _source_tree_sha256(Path(__file__).resolve().parents[2])
    source_matches = hashes.get("source_tree_sha256") == current_source
    if require_current_source and not source_matches:
        raise ExperimentError(f"completed result was produced by different source: {run_dir}")
    recorded_outputs = hashes.get("outputs_before_manifest")
    if not isinstance(recorded_outputs, Mapping):
        raise ExperimentError(f"completed result lacks output hashes: {run_dir}")
    expected_output_names = {"selections.npz", "predictions.csv", "metrics.json"}
    forward_audit = manifest.get("forward_audit")
    if isinstance(forward_audit, Mapping) and forward_audit.get("spv_saved") is True:
        expected_output_names.add("spv_images.npz")
    if set(map(str, recorded_outputs.keys())) != expected_output_names:
        raise ExperimentError(f"completed result output hash set is incomplete: {run_dir}")
    actual_output_names = names - {"manifest.json"}
    if actual_output_names != expected_output_names:
        raise ExperimentError(f"completed result contains missing or extra files: {run_dir}")
    for filename, expected_hash in recorded_outputs.items():
        output = run_dir / str(filename)
        if not output.is_file() or sha256_file(output) != expected_hash:
            raise ExperimentError(f"completed output hash mismatch ({filename}): {run_dir}")
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        raise ExperimentError(f"completed result lacks environment identity: {run_dir}")
    recorded_environment = environment.get("fingerprint_sha256")
    current_environment = _runtime_fingerprint()
    if recorded_environment != current_environment:
        raise ExperimentError(f"completed result was produced in a different environment: {run_dir}")
    return dict(metrics) | {
        "output_path": str(run_dir),
        "source_matches_current": source_matches,
        "recorded_source_tree_sha256": str(hashes.get("source_tree_sha256", "")),
        "recorded_environment_sha256": str(recorded_environment),
        "resumed": True,
    }


def execute_condition(
    config: ExperimentConfig,
    condition: ConditionSpec,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute and atomically commit one condition to its own result directory."""

    validate_plan_files(config)
    run_id = condition.run_id(config.experiment_id)
    ideal_path = config.mapping_dir / "ideal_mapping.npz"
    target_path = mapping_path_for_condition(config, condition)
    runtime_before = validate_runtime_environment(config)
    runtime_fingerprint_before = _runtime_fingerprint(runtime_before)
    source_root = Path(__file__).resolve().parents[2]
    source_hash_before = _source_tree_sha256(source_root)
    input_hashes_before = {
        "config_sha256": sha256_file(config.source_path),
        "dataset_sha256": sha256_file(config.dataset_path),
        "ideal_mapping_sha256": sha256_file(ideal_path),
        "target_mapping_sha256": sha256_file(target_path),
    }
    config.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = config.output_root / run_id
    if run_dir.exists():
        if not overwrite:
            raise ExperimentError(
                f"result already exists (use --overwrite deliberately): {run_dir}"
            )
        _safe_replace_existing(run_dir, config.output_root)
    staging = config.output_root / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        dataset = load_glyph_dataset(config)
        regular_grid = make_regular_grid(
            rows=16, cols=16, image_shape=EXPECTED_IMAGE_SHAPE
        )
        grid = regular_grid.electrodes
        ideal_mapping = load_ideal_mapping(ideal_path)
        if not np.array_equal(ideal_mapping.centers_xy, grid.centers_xy):
            raise ExperimentError("ideal mapping differs from the regular grid")
        target_mapping = load_mapping(target_path, condition=condition)
        with threadpool_limits(limits=config.max_blas_threads):
            selections, scores, trace_arrays, selection_audit = _encode_all(
                dataset,
                grid,
                condition=condition,
                ideal_mapping=ideal_mapping,
                target_mapping=target_mapping,
            )
            target_forward = GaussianPhospheneForwardModel(target_mapping)
            amplitudes = np.ones(selections.shape, dtype=np.float32)
            spv_images = target_forward.generate_variable_batch(selections, amplitudes)
            energy = spv_images.sum(axis=(1, 2), dtype=np.float64)
            expected_energy = expected_output_energy(
                amplitudes,
                amplitude_scale=target_forward.amplitude_scale,
            )
            if not np.allclose(energy, expected_energy, rtol=0.0, atol=2e-5):
                raise ExperimentError("SPV output energy audit failed")
            extractor, decoder = _build_feature_and_classifier(config)
            features = extractor.extract_batch(spv_images)
            decoder.fit(
                features[dataset.train_indices],
                dataset.class_ids[dataset.train_indices],
            )
            predictions = decoder.predict(features[dataset.test_indices])
        true_labels = dataset.class_ids[dataset.test_indices]
        accuracy = classification_accuracy(true_labels, predictions)
        macro_f1 = float(
            f1_score(true_labels, predictions, average="macro", zero_division=0)
        )
        selection_arrays: dict[str, NDArray[Any]] = {
            "sample_ids": np.asarray(dataset.sample_ids),
            "selected_indices": selections,
            "candidate_scores": scores,
            **trace_arrays,
        }
        write_deterministic_npz(
            staging / "selections.npz",
            selection_arrays,
        )
        if config.save_spv:
            write_deterministic_npz(
                staging / "spv_images.npz",
                {
                    "sample_ids": np.asarray(dataset.sample_ids),
                    "spv_images": spv_images,
                },
            )
        _write_predictions(
            staging / "predictions.csv",
            dataset,
            predictions,
            condition=condition,
            run_id=run_id,
        )
        elapsed = time.perf_counter() - started
        metrics = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "PASS_COMMITTED",
            "method": condition.method,
            "budget": condition.budget,
            "mapping_type": condition.mapping_type,
            "mapping_seed": condition.mapping_seed,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "train_sample_count": int(dataset.train_indices.size),
            "test_sample_count": int(dataset.test_indices.size),
            "class_count": int(np.unique(true_labels).size),
            "classifier_n_iter": decoder.n_iter,
            "elapsed_seconds": elapsed,
        }
        _json_write(staging / "metrics.json", metrics)
        input_hashes_after = {
            "config_sha256": sha256_file(config.source_path),
            "dataset_sha256": sha256_file(config.dataset_path),
            "ideal_mapping_sha256": sha256_file(ideal_path),
            "target_mapping_sha256": sha256_file(target_path),
        }
        if input_hashes_after != input_hashes_before:
            raise ExperimentError("an input binding changed while the condition was running")
        if _source_tree_sha256(source_root) != source_hash_before:
            raise ExperimentError("Python source changed while the condition was running")
        runtime_after = validate_runtime_environment(config)
        if _runtime_fingerprint(runtime_after) != runtime_fingerprint_before:
            raise ExperimentError("runtime environment changed while the condition was running")
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "PASS_COMMITTED",
            "started_utc": started_utc,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "condition": {
                "method": condition.method,
                "budget": condition.budget,
                "mapping_type": condition.mapping_type,
                "mapping_seed": condition.mapping_seed,
            },
            "paths": {
                "config": str(config.source_path),
                "dataset": str(config.dataset_path),
                "ideal_mapping": str(ideal_path),
                "target_mapping": str(target_path),
            },
            "font_split": {
                "train_fonts": list(config.train_fonts),
                "test_fonts": list(config.test_fonts),
            },
            "hashes": {
                **input_hashes_before,
                "source_tree_sha256": source_hash_before,
                "outputs_before_manifest": output_hashes,
            },
            "selection_audit": selection_audit,
            "forward_audit": {
                "target_mapping_id": target_mapping.mapping_id,
                "unit_amplitudes": True,
                "expected_energy": float(condition.budget),
                "observed_energy_min": float(energy.min()),
                "observed_energy_max": float(energy.max()),
                "energy_audit_passed": True,
                "spv_images_sha256": array_sha256(spv_images),
                "hog_features_sha256": array_sha256(features),
                "test_predictions_sha256": array_sha256(predictions),
                "spv_saved": config.save_spv,
            },
            "environment": _environment_document(runtime_before),
        }
        _json_write(staging / "manifest.json", manifest)
        staging.replace(run_dir)
        return metrics | {"output_path": str(run_dir)}
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def select_conditions(
    config: ExperimentConfig,
    *,
    run_id: str | None = None,
    method: str | None = None,
    budget: int | None = None,
    mapping_seed: int | None = None,
) -> tuple[ConditionSpec, ...]:
    selected = []
    for condition in config.conditions:
        if run_id is not None and condition.run_id(config.experiment_id) != run_id:
            continue
        if method is not None and condition.method != method:
            continue
        if budget is not None and condition.budget != budget:
            continue
        if mapping_seed is not None and condition.mapping_seed != mapping_seed:
            continue
        selected.append(condition)
    if not selected:
        raise ExperimentError("condition filters selected no runs")
    return tuple(selected)


def execute_plan(
    config: ExperimentConfig,
    conditions: Sequence[ConditionSpec],
    *,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    def reportable(result: Mapping[str, Any]) -> dict[str, Any]:
        if config.report_metrics:
            return dict(result)
        allowed = {
            "schema_version",
            "run_id",
            "status",
            "method",
            "budget",
            "mapping_type",
            "mapping_seed",
            "elapsed_seconds",
            "output_path",
            "resumed",
        }
        return {key: value for key, value in result.items() if key in allowed}

    results = []
    for index, condition in enumerate(conditions, start=1):
        run_dir = config.output_root / condition.run_id(config.experiment_id)
        if run_dir.exists() and not overwrite:
            result = validate_completed_result(config, condition)
            print(
                f"[{index}/{len(conditions)}] {condition.run_id(config.experiment_id)} "
                "SKIP_VALID_COMMITTED",
                flush=True,
            )
            results.append(reportable(result))
            continue
        print(
            f"[{index}/{len(conditions)}] {condition.run_id(config.experiment_id)}",
            flush=True,
        )
        result = execute_condition(config, condition, overwrite=overwrite)
        if config.report_metrics:
            print(
                f"  PASS accuracy={result['accuracy']:.6f} "
                f"macro_f1={result['macro_f1']:.6f} "
                f"seconds={result['elapsed_seconds']:.1f}",
                flush=True,
            )
        else:
            print(
                f"  PASS metrics_blinded seconds={result['elapsed_seconds']:.1f}",
                flush=True,
            )
        results.append(reportable(result))
    return results
