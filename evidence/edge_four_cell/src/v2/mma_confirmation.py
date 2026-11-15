"""One-shot independent confirmation for multi-mapping Hanzi prototypes."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import uuid
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from src.deterministic_io import array_sha256, sha256_file
from src.encoders.skeleton_topk import SkeletonTopKEncoder
from src.features import HogFeatureExtractor
from src.grid import make_regular_grid
from src.spv_forward import GaussianPhospheneForwardModel, IdealGaussianUtilityProjector
from src.v2.experiment import ConditionSpec, load_mapping
from src.v2.multimapping_prototypes import MultiMappingCosinePrototypeClassifier


class MmaConfirmationError(ValueError):
    """Raised when a confirmation input or frozen protocol is invalid."""


METHODS = (
    "ideal_only",
    "single_moderate",
    "multi4",
    "ideal_plus_multi4",
    "oracle_matched",
)
PRIMARY_METHOD = "ideal_plus_multi4"
PRIMARY_BASELINE = "ideal_only"


@dataclass(frozen=True, slots=True)
class ConfirmationConfig:
    source_path: Path
    project_root: Path
    experiment_id: str
    train_dataset: Path
    train_dataset_sha256: str
    train_fonts: tuple[str, ...]
    ideal_mapping: Path
    ideal_mapping_sha256: str
    train_mapping_paths: Mapping[int, Path]
    train_mapping_sha256: Mapping[int, str]
    test_asset_dir: Path
    test_asset_manifest: Path
    test_asset_manifest_sha256: str
    test_dataset: Path
    test_dataset_sha256: str
    test_fonts: tuple[str, ...]
    test_mapping_paths: Mapping[int, Path]
    test_mapping_sha256: Mapping[int, str]
    lock_path: Path
    output_root: Path
    budget: int
    feature: Mapping[str, Any]
    bootstrap_iterations: int
    bootstrap_seed: int
    expected_environment: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Archive:
    images: NDArray[np.float32]
    class_ids: NDArray[np.int64]
    characters: NDArray[np.str_]
    font_ids: NDArray[np.str_]
    sample_ids: NDArray[np.str_]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MmaConfirmationError(f"{name} must be a mapping")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MmaConfirmationError(f"{name} must be a non-empty list")
    result = tuple(map(str, value))
    if len(set(result)) != len(result) or any(not item for item in result):
        raise MmaConfirmationError(f"{name} contains empty or duplicate values")
    return result


def _digest(value: Any, name: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(item not in "0123456789abcdef" for item in result):
        raise MmaConfirmationError(f"{name} must be a SHA-256 hex digest")
    return result


def _path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MmaConfirmationError(f"{name} must be a non-empty path")
    raw = value.strip()
    if os.name != "nt" and len(raw) > 2 and raw[1:3] in {":\\", ":/"}:
        candidate = Path("/mnt") / raw[0].lower() / raw[3:].replace("\\", "/")
    else:
        candidate = Path(raw)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def load_confirmation_config(path: str | Path) -> ConfirmationConfig:
    source = Path(path).resolve()
    if not source.is_file():
        raise MmaConfirmationError(f"config does not exist: {source}")
    document = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "config")
    if int(document.get("schema_version", -1)) != 1:
        raise MmaConfirmationError("schema_version must equal 1")
    root = Path(__file__).resolve().parents[2]
    train = _mapping(document.get("training"), "training")
    test = _mapping(document.get("confirmation"), "confirmation")
    ideal = _mapping(train.get("ideal_mapping"), "training.ideal_mapping")
    raw_train_mappings = _mapping(train.get("moderate_mappings"), "training.moderate_mappings")
    asset_dir = _path(root, test.get("asset_dir"), "confirmation.asset_dir")
    raw_test_mappings = _mapping(test.get("moderate_mappings"), "confirmation.moderate_mappings")
    train_paths: dict[int, Path] = {}
    train_hashes: dict[int, str] = {}
    for raw_seed, raw_spec in raw_train_mappings.items():
        seed = int(raw_seed)
        spec = _mapping(raw_spec, f"training.moderate_mappings.{seed}")
        train_paths[seed] = _path(root, spec.get("path"), f"train mapping {seed}")
        train_hashes[seed] = _digest(spec.get("sha256"), f"train mapping {seed}")
    test_paths: dict[int, Path] = {}
    test_hashes: dict[int, str] = {}
    for raw_seed, raw_spec in raw_test_mappings.items():
        seed = int(raw_seed)
        spec = _mapping(raw_spec, f"confirmation.moderate_mappings.{seed}")
        test_paths[seed] = _path(asset_dir, spec.get("path"), f"test mapping {seed}")
        test_hashes[seed] = _digest(spec.get("sha256"), f"test mapping {seed}")
    if tuple(sorted(train_paths)) != (2026, 2027, 2028, 2029):
        raise MmaConfirmationError("training mapping seeds are frozen at 2026..2029")
    if tuple(sorted(test_paths)) != (2031, 2032, 2033, 2034, 2035):
        raise MmaConfirmationError("confirmation mapping seeds are frozen at 2031..2035")
    if set(train_paths) & set(test_paths):
        raise MmaConfirmationError("training and confirmation mapping seeds overlap")
    feature = _mapping(document.get("feature"), "feature")
    bootstrap = _mapping(document.get("bootstrap"), "bootstrap")
    environment = _mapping(document.get("environment"), "environment")
    packages = _mapping(environment.get("packages", {}), "environment.packages")
    expected_environment = {"python": str(environment.get("python", ""))}
    expected_environment.update({str(name): str(version) for name, version in packages.items()})
    config = ConfirmationConfig(
        source_path=source,
        project_root=root,
        experiment_id=str(document.get("experiment_id", "")).strip(),
        train_dataset=_path(root, train.get("glyphs_npz"), "training.glyphs_npz"),
        train_dataset_sha256=_digest(train.get("sha256"), "training.sha256"),
        train_fonts=_string_tuple(train.get("fonts"), "training.fonts"),
        ideal_mapping=_path(root, ideal.get("path"), "training.ideal_mapping.path"),
        ideal_mapping_sha256=_digest(ideal.get("sha256"), "training.ideal_mapping.sha256"),
        train_mapping_paths=train_paths,
        train_mapping_sha256=train_hashes,
        test_asset_dir=asset_dir,
        test_asset_manifest=_path(asset_dir, test.get("asset_manifest"), "confirmation.asset_manifest"),
        test_asset_manifest_sha256=_digest(test.get("asset_manifest_sha256"), "confirmation.asset_manifest_sha256"),
        test_dataset=_path(asset_dir, test.get("glyphs_npz"), "confirmation.glyphs_npz"),
        test_dataset_sha256=_digest(test.get("glyphs_sha256"), "confirmation.glyphs_sha256"),
        test_fonts=_string_tuple(test.get("fonts"), "confirmation.fonts"),
        test_mapping_paths=test_paths,
        test_mapping_sha256=test_hashes,
        lock_path=_path(root, document.get("confirmation_lock"), "confirmation_lock"),
        output_root=_path(root, document.get("output_root"), "output_root"),
        budget=int(document.get("budget", -1)),
        feature=dict(feature),
        bootstrap_iterations=int(bootstrap.get("iterations", -1)),
        bootstrap_seed=int(bootstrap.get("seed", -1)),
        expected_environment=expected_environment,
    )
    if not config.experiment_id or not all(
        item.isalnum() or item in "-_" for item in config.experiment_id
    ):
        raise MmaConfirmationError("experiment_id must be filesystem-safe")
    if config.budget != 64:
        raise MmaConfirmationError("confirmation budget is frozen at B=64")
    if config.bootstrap_iterations != 100_000 or config.bootstrap_seed < 0:
        raise MmaConfirmationError("bootstrap is frozen at 100000 iterations")
    if len(config.train_fonts) != 6 or len(config.test_fonts) != 2:
        raise MmaConfirmationError("confirmation requires exactly six train and two test fonts")
    if set(config.train_fonts) & set(config.test_fonts):
        raise MmaConfirmationError("training and confirmation fonts overlap")
    return config


def _runtime_identity() -> dict[str, Any]:
    package_names = ("numpy", "scipy", "scikit-image", "scikit-learn", "PyYAML", "threadpoolctl")
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {name: importlib_metadata.version(name) for name in package_names},
    }


def _validate_environment(config: ConfirmationConfig) -> dict[str, Any]:
    identity = _runtime_identity()
    mismatches: list[str] = []
    if identity["python"] != config.expected_environment.get("python"):
        mismatches.append(
            f"python expected {config.expected_environment.get('python')}, found {identity['python']}"
        )
    for name, actual in identity["packages"].items():
        expected = config.expected_environment.get(name)
        if expected != actual:
            mismatches.append(f"{name} expected {expected}, found {actual}")
    if mismatches:
        raise MmaConfirmationError("environment mismatch: " + "; ".join(mismatches))
    return identity


def _archive_font_hashes(path: Path, fonts: Sequence[str]) -> dict[str, str]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"font_ids", "font_sha256"}
        if not required <= set(archive.files):
            raise MmaConfirmationError(f"glyph archive lacks font hash metadata: {path}")
        font_ids = np.asarray(archive["font_ids"], dtype=np.str_)
        font_hashes = np.asarray(archive["font_sha256"], dtype=np.str_)
    if font_ids.shape != font_hashes.shape:
        raise MmaConfirmationError(f"font hash metadata shape mismatch: {path}")
    result: dict[str, str] = {}
    for font in fonts:
        values = np.unique(font_hashes[font_ids == font])
        if values.shape != (1,):
            raise MmaConfirmationError(f"font {font} lacks one stable file hash")
        result[font] = _digest(str(values[0]), f"font hash {font}")
    return result


def _validate_asset_manifest(
    config: ConfirmationConfig, manifest: Mapping[str, Any]
) -> dict[str, str]:
    if manifest.get("status") != "FROZEN_INPUTS_ONLY_NO_PERFORMANCE_READ":
        raise MmaConfirmationError("confirmation asset manifest has an invalid status")
    if (
        manifest.get("prediction_executed") is not False
        or manifest.get("accuracy_read") is not False
    ):
        raise MmaConfirmationError("confirmation assets were not performance-naive")
    source = _mapping(
        manifest.get("source_character_archive"),
        "asset manifest source_character_archive",
    )
    if source.get("sha256") != config.train_dataset_sha256:
        raise MmaConfirmationError("asset character source differs from training archive")
    glyph = _mapping(manifest.get("glyph_archive"), "asset manifest glyph_archive")
    if glyph.get("file_sha256") != config.test_dataset_sha256:
        raise MmaConfirmationError("asset manifest glyph hash differs from config")
    if int(glyph.get("sample_count", -1)) != 360 or int(glyph.get("font_count", -1)) != 2:
        raise MmaConfirmationError("asset manifest glyph counts differ from protocol")

    font_entries = manifest.get("fonts")
    if not isinstance(font_entries, list) or len(font_entries) != 2:
        raise MmaConfirmationError("asset manifest must bind exactly two fonts")
    font_hashes: dict[str, str] = {}
    for item in font_entries:
        entry = _mapping(item, "asset manifest font entry")
        font_id = str(entry.get("font_id", ""))
        if font_id in font_hashes:
            raise MmaConfirmationError("asset manifest contains duplicate font IDs")
        font_hashes[font_id] = _digest(entry.get("file_sha256"), f"font {font_id}")
        if (
            int(entry.get("missing_character_count", -1)) != 0
            or int(entry.get("unique_normalized_glyphs", -1)) != 180
            or int(entry.get("unique_raw_glyph_fingerprints", -1)) != 180
        ):
            raise MmaConfirmationError(f"font QC is not complete for {font_id}")
    if tuple(font_hashes) != config.test_fonts:
        raise MmaConfirmationError("asset manifest font order/IDs differ from config")

    mapping_entries = manifest.get("mappings")
    if not isinstance(mapping_entries, list) or len(mapping_entries) != 5:
        raise MmaConfirmationError("asset manifest must bind exactly five mappings")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for item in mapping_entries:
        entry = _mapping(item, "asset manifest mapping entry")
        seed = int(entry.get("seed", -1))
        if seed in by_seed:
            raise MmaConfirmationError("asset manifest contains duplicate mapping seeds")
        by_seed[seed] = entry
    if tuple(sorted(by_seed)) != tuple(sorted(config.test_mapping_paths)):
        raise MmaConfirmationError("asset manifest mapping seeds differ from config")
    for seed, entry in by_seed.items():
        if entry.get("file_sha256") != config.test_mapping_sha256[seed]:
            raise MmaConfirmationError(f"asset manifest mapping hash differs for seed {seed}")
        expected_relative = config.test_mapping_paths[seed].relative_to(
            config.test_asset_dir
        ).as_posix()
        if entry.get("file") != expected_relative:
            raise MmaConfirmationError(f"asset manifest mapping path differs for seed {seed}")
    return font_hashes


def validate_confirmation_inputs(
    config: ConfirmationConfig, *, require_lock: bool = True
) -> dict[str, Any]:
    identity = _validate_environment(config)
    bound = {
        config.train_dataset: config.train_dataset_sha256,
        config.ideal_mapping: config.ideal_mapping_sha256,
        config.test_asset_manifest: config.test_asset_manifest_sha256,
        config.test_dataset: config.test_dataset_sha256,
    }
    bound.update({config.train_mapping_paths[seed]: digest for seed, digest in config.train_mapping_sha256.items()})
    bound.update({config.test_mapping_paths[seed]: digest for seed, digest in config.test_mapping_sha256.items()})
    for path, expected in bound.items():
        if not path.is_file():
            raise MmaConfirmationError(f"frozen input does not exist: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise MmaConfirmationError(f"frozen input SHA-256 mismatch: {path}")
    manifest = _mapping(
        json.loads(config.test_asset_manifest.read_text(encoding="utf-8")),
        "asset manifest",
    )
    manifest_font_hashes = _validate_asset_manifest(config, manifest)
    train_font_hashes = _archive_font_hashes(config.train_dataset, config.train_fonts)
    test_font_hashes = _archive_font_hashes(config.test_dataset, config.test_fonts)
    if test_font_hashes != manifest_font_hashes:
        raise MmaConfirmationError("test archive font hashes differ from asset manifest")
    if set(train_font_hashes.values()) & set(test_font_hashes.values()):
        raise MmaConfirmationError("training and confirmation font files overlap by hash")
    geometry_hashes: list[tuple[str, str]] = []
    for path in (*config.train_mapping_paths.values(), *config.test_mapping_paths.values()):
        with np.load(path, allow_pickle=False) as archive:
            geometry_hashes.append(
                (
                    array_sha256(np.asarray(archive["centers_xy"], dtype=np.float32)),
                    array_sha256(np.asarray(archive["sigma_px"], dtype=np.float32)),
                )
            )
    if len(set(geometry_hashes)) != len(geometry_hashes):
        raise MmaConfirmationError("a train/test moderate mapping geometry is duplicated")
    lock_sha256 = None
    if require_lock:
        lock_sha256 = validate_confirmation_lock(config)
    return {
        "experiment_id": config.experiment_id,
        "methods": list(METHODS),
        "primary_comparison": f"{PRIMARY_METHOD}-{PRIMARY_BASELINE}",
        "budget": config.budget,
        "train_fonts": list(config.train_fonts),
        "test_fonts": list(config.test_fonts),
        "train_mapping_seeds": sorted(config.train_mapping_paths),
        "test_mapping_seeds": sorted(config.test_mapping_paths),
        "lock_path": str(config.lock_path),
        "lock_sha256": lock_sha256,
        "environment": identity,
        "output_root": str(config.output_root),
    }


def _load_archive(path: Path, selected_fonts: Sequence[str]) -> Archive:
    with np.load(path, allow_pickle=False) as archive:
        required = {"images", "class_ids", "characters", "font_ids", "sample_ids"}
        if not required <= set(archive.files):
            raise MmaConfirmationError(f"glyph archive is incomplete: {path}")
        raw = {name: archive[name] for name in required}
    mask = np.isin(raw["font_ids"], np.asarray(selected_fonts, dtype=np.str_))
    if int(mask.sum()) != len(selected_fonts) * 180:
        raise MmaConfirmationError(f"glyph archive does not contain fonts x 180: {path}")
    result = Archive(
        images=np.ascontiguousarray(raw["images"][mask], dtype=np.float32),
        class_ids=np.ascontiguousarray(raw["class_ids"][mask], dtype=np.int64),
        characters=np.asarray(raw["characters"][mask], dtype=np.str_).copy(),
        font_ids=np.asarray(raw["font_ids"][mask], dtype=np.str_).copy(),
        sample_ids=np.asarray(raw["sample_ids"][mask], dtype=np.str_).copy(),
    )
    if result.images.shape != (len(selected_fonts) * 180, 96, 96):
        raise MmaConfirmationError("glyph image shape differs from protocol")
    if not np.isfinite(result.images).all() or not (
        0.0 <= float(result.images.min()) <= float(result.images.max()) <= 1.0
    ):
        raise MmaConfirmationError("glyph images contain invalid numeric values")
    if len(set(map(str, result.sample_ids))) != result.sample_ids.size:
        raise MmaConfirmationError("glyph sample IDs must be unique")
    if set(map(str, np.unique(result.font_ids))) != set(selected_fonts):
        raise MmaConfirmationError("glyph font identities differ from config")
    for font in selected_fonts:
        labels = np.sort(result.class_ids[result.font_ids == font])
        if not np.array_equal(labels, np.arange(180, dtype=np.int64)):
            raise MmaConfirmationError(f"font {font} does not contain classes 0..179")
    return result


def _class_characters(archive: Archive) -> tuple[str, ...]:
    sequence: list[str] = []
    for class_id in range(180):
        values = np.unique(archive.characters[archive.class_ids == class_id])
        if values.shape != (1,):
            raise MmaConfirmationError(f"ambiguous character for class {class_id}")
        sequence.append(str(values[0]))
    return tuple(sequence)


def _select(images: NDArray[np.float32], ideal_mapping: Any, budget: int) -> NDArray[np.int64]:
    grid = make_regular_grid(rows=16, cols=16, image_shape=(96, 96)).electrodes
    if not np.array_equal(ideal_mapping.centers_xy, grid.centers_xy):
        raise MmaConfirmationError("ideal mapping differs from the regular 16x16 grid")
    encoder = SkeletonTopKEncoder()
    projector = IdealGaussianUtilityProjector(ideal_mapping)
    selected = np.empty((images.shape[0], budget), dtype=np.int64)
    for index, image in enumerate(images):
        selected[index] = encoder.encode(image, grid, budget, projector=projector).selected_indices
    return np.ascontiguousarray(selected)


def _extractor(config: ConfirmationConfig) -> HogFeatureExtractor:
    return HogFeatureExtractor(
        image_shape=(96, 96),
        orientations=int(config.feature.get("orientations", 9)),
        pixels_per_cell=tuple(map(int, config.feature.get("pixels_per_cell", [8, 8]))),
        cells_per_block=tuple(map(int, config.feature.get("cells_per_block", [2, 2]))),
        block_norm=str(config.feature.get("block_norm", "L2-Hys")),
    )


def _features(
    mapping: Any,
    selected: NDArray[np.int64],
    extractor: HogFeatureExtractor,
) -> NDArray[np.float32]:
    amplitudes = np.ones(selected.shape, dtype=np.float32)
    images = GaussianPhospheneForwardModel(mapping).generate_variable_batch(selected, amplitudes)
    return extractor.extract_batch(images)


def _fit(
    features_by_mapping: Mapping[str, NDArray[np.float32]],
    archive: Archive,
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


def _metrics(labels: NDArray[np.int64], prediction: NDArray[np.int64]) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(labels == prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def _bootstrap(
    per_character_difference: NDArray[np.float64],
    *,
    iterations: int,
    seed: int,
    cells_per_character: int | None = None,
) -> dict[str, Any]:
    if per_character_difference.shape != (180,):
        raise MmaConfirmationError("bootstrap requires 180 character clusters")
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    chunk = 10_000
    for start in range(0, iterations, chunk):
        stop = min(iterations, start + chunk)
        indices = rng.integers(0, 180, size=(stop - start, 180))
        values[start:stop] = per_character_difference[indices].mean(axis=1)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "unit": "character_cluster",
        "cluster_count": 180,
        "within_cluster_cells": cells_per_character,
        "iterations": iterations,
        "seed": seed,
        "mean_difference_pp": float(per_character_difference.mean() * 100.0),
        "ci95_pp": [float(lower * 100.0), float(upper * 100.0)],
    }


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="")
    temporary.replace(path)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src").rglob("*.py"))
    paths.extend(sorted(root.glob("run_mma_*.py")))
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _protocol_binding(config: ConfirmationConfig) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "primary_method": PRIMARY_METHOD,
        "primary_baseline": PRIMARY_BASELINE,
        "methods": list(METHODS),
        "budget": config.budget,
        "feature": dict(config.feature),
        "classifier": "multimapping_cosine_prototype",
        "encoder": "ideal_skeleton_topk",
        "train_dataset_sha256": config.train_dataset_sha256,
        "train_fonts": list(config.train_fonts),
        "ideal_mapping_sha256": config.ideal_mapping_sha256,
        "train_mapping_sha256": {
            str(seed): config.train_mapping_sha256[seed]
            for seed in sorted(config.train_mapping_sha256)
        },
        "test_asset_manifest_sha256": config.test_asset_manifest_sha256,
        "test_dataset_sha256": config.test_dataset_sha256,
        "test_fonts": list(config.test_fonts),
        "test_mapping_sha256": {
            str(seed): config.test_mapping_sha256[seed]
            for seed in sorted(config.test_mapping_sha256)
        },
        "single_moderate_seed": 2026,
        "bootstrap": {
            "unit": "character_cluster",
            "iterations": config.bootstrap_iterations,
            "seed": config.bootstrap_seed,
        },
        "expected_environment": dict(config.expected_environment),
        "output_root": str(config.output_root),
        "parameter_search": False,
    }


def validate_confirmation_lock(config: ConfirmationConfig) -> str:
    if not config.lock_path.is_file():
        raise MmaConfirmationError(
            f"confirmation lock is missing; freeze before running: {config.lock_path}"
        )
    document = _mapping(
        json.loads(config.lock_path.read_text(encoding="utf-8")),
        "confirmation lock",
    )
    if document.get("status") != "FROZEN_AWAITING_ONE_SHOT_CONFIRMATION":
        raise MmaConfirmationError("confirmation lock status is invalid")
    if document.get("config_sha256") != sha256_file(config.source_path):
        raise MmaConfirmationError("confirmation config changed after protocol freeze")
    if document.get("source_tree_sha256") != _source_sha256(config.project_root):
        raise MmaConfirmationError("execution source changed after protocol freeze")
    if document.get("protocol_binding") != _protocol_binding(config):
        raise MmaConfirmationError("confirmation protocol differs from frozen lock")
    return sha256_file(config.lock_path)


def freeze_confirmation_lock(config: ConfirmationConfig) -> dict[str, Any]:
    """Bind the execution source and config before any confirmation scoring."""

    preflight = validate_confirmation_inputs(config, require_lock=False)
    completion_path = config.output_root.with_name(
        f"{config.output_root.name}.completion.json"
    )
    if config.output_root.exists() or completion_path.exists():
        raise MmaConfirmationError("cannot freeze after confirmation output exists")
    if config.lock_path.exists():
        raise MmaConfirmationError(
            f"confirmation lock already exists; refusing to replace: {config.lock_path}"
        )
    document = {
        "schema_version": 1,
        "status": "FROZEN_AWAITING_ONE_SHOT_CONFIRMATION",
        "config_path": str(config.source_path),
        "config_sha256": sha256_file(config.source_path),
        "source_tree_sha256": _source_sha256(config.project_root),
        "protocol_binding": _protocol_binding(config),
        "preflight_without_lock": preflight,
    }
    config.lock_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        config.lock_path,
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "status": document["status"],
        "lock_path": str(config.lock_path),
        "lock_sha256": sha256_file(config.lock_path),
        "config_sha256": document["config_sha256"],
        "source_tree_sha256": document["source_tree_sha256"],
        "confirmation_output_exists": False,
    }


def execute_confirmation(config: ConfirmationConfig) -> dict[str, Any]:
    preflight = validate_confirmation_inputs(config)
    completion_path = config.output_root.with_name(
        f"{config.output_root.name}.completion.json"
    )
    if config.output_root.exists() or completion_path.exists():
        raise MmaConfirmationError(
            "confirmation output/completion already exists; one-shot rerun refused"
        )
    train = _load_archive(config.train_dataset, config.train_fonts)
    test = _load_archive(config.test_dataset, config.test_fonts)
    if set(map(str, train.sample_ids)) & set(map(str, test.sample_ids)):
        raise MmaConfirmationError("training and confirmation sample IDs overlap")
    if _class_characters(train) != _class_characters(test):
        raise MmaConfirmationError("training and confirmation character sequences differ")
    ideal = load_mapping(
        config.ideal_mapping,
        condition=ConditionSpec("skeleton_topk", 64, "ideal", 0),
    )
    train_mappings = {
        seed: load_mapping(
            path,
            condition=ConditionSpec("skeleton_topk", 64, "moderate", seed),
        )
        for seed, path in config.train_mapping_paths.items()
    }
    test_mappings = {
        seed: load_mapping(
            path,
            condition=ConditionSpec("skeleton_topk", 64, "moderate", seed),
        )
        for seed, path in config.test_mapping_paths.items()
    }
    source_sha = _source_sha256(config.project_root)
    output_parent = config.output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = output_parent / f".{config.output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        with threadpool_limits(limits=1):
            train_selected = _select(train.images, ideal, config.budget)
            test_selected = _select(test.images, ideal, config.budget)
            extractor = _extractor(config)
            train_features: dict[str, NDArray[np.float32]] = {
                "ideal": _features(ideal, train_selected, extractor)
            }
            for seed, mapping in sorted(train_mappings.items()):
                train_features[f"moderate_seed{seed}"] = _features(
                    mapping, train_selected, extractor
                )
            for seed, mapping in sorted(test_mappings.items()):
                train_features[f"moderate_seed{seed}"] = _features(
                    mapping, train_selected, extractor
                )
            train_names = [f"moderate_seed{seed}" for seed in sorted(train_mappings)]
            fixed_models = {
                "ideal_only": _fit(train_features, train, ["ideal"]),
                "single_moderate": _fit(
                    train_features, train, ["moderate_seed2026"]
                ),
                "multi4": _fit(train_features, train, train_names),
                "ideal_plus_multi4": _fit(
                    train_features, train, ["ideal", *train_names]
                ),
            }
            predictions: list[dict[str, Any]] = []
            fold_metrics: list[dict[str, Any]] = []
            correctness: dict[tuple[str, int], NDArray[np.bool_]] = {}
            for seed, mapping in sorted(test_mappings.items()):
                test_x = _features(mapping, test_selected, extractor)
                oracle = _fit(train_features, train, [f"moderate_seed{seed}"])
                models = fixed_models | {"oracle_matched": oracle}
                for method in METHODS:
                    prediction = np.asarray(models[method].predict(test_x), dtype=np.int64)
                    correct = prediction == test.class_ids
                    correctness[(method, seed)] = correct
                    overall = _metrics(test.class_ids, prediction)
                    fold_metrics.append(
                        {
                            "mapping_seed": seed,
                            "font_id": "ALL",
                            "method": method,
                            "sample_count": int(test.class_ids.size),
                            **overall,
                        }
                    )
                    for font in config.test_fonts:
                        mask = test.font_ids == font
                        fold_metrics.append(
                            {
                                "mapping_seed": seed,
                                "font_id": font,
                                "method": method,
                                "sample_count": int(mask.sum()),
                                **_metrics(test.class_ids[mask], prediction[mask]),
                            }
                        )
                    for index in range(test.class_ids.size):
                        predictions.append(
                            {
                                "mapping_seed": seed,
                                "method": method,
                                "sample_id": str(test.sample_ids[index]),
                                "font_id": str(test.font_ids[index]),
                                "class_id": int(test.class_ids[index]),
                                "character": str(test.characters[index]),
                                "predicted_class_id": int(prediction[index]),
                                "correct": int(correct[index]),
                            }
                        )

        overall_rows = [row for row in fold_metrics if row["font_id"] == "ALL"]
        summary_methods: dict[str, Any] = {}
        for method in METHODS:
            rows = [row for row in overall_rows if row["method"] == method]
            summary_methods[method] = {
                "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
                "std_accuracy": float(np.std([row["accuracy"] for row in rows])),
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
                "per_seed_accuracy": {
                    str(row["mapping_seed"]): row["accuracy"] for row in rows
                },
            }
        differences = []
        for class_id in range(180):
            cells = []
            class_mask = test.class_ids == class_id
            for seed in sorted(test_mappings):
                cells.extend(
                    correctness[(PRIMARY_METHOD, seed)][class_mask].astype(np.float64)
                    - correctness[(PRIMARY_BASELINE, seed)][class_mask].astype(np.float64)
                )
            differences.append(float(np.mean(cells)))
        bootstrap = _bootstrap(
            np.asarray(differences, dtype=np.float64),
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
            cells_per_character=len(config.test_fonts) * len(config.test_mapping_paths),
        )
        seed_differences = {
            str(seed): float(
                correctness[(PRIMARY_METHOD, seed)].mean()
                - correctness[(PRIMARY_BASELINE, seed)].mean()
            )
            for seed in sorted(test_mappings)
        }
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "status": "PASS_COMMITTED",
            "primary_method": PRIMARY_METHOD,
            "primary_baseline": PRIMARY_BASELINE,
            "primary_difference_pp": float(
                (
                    summary_methods[PRIMARY_METHOD]["mean_accuracy"]
                    - summary_methods[PRIMARY_BASELINE]["mean_accuracy"]
                )
                * 100.0
            ),
            "primary_per_seed_difference_pp": {
                seed: value * 100.0 for seed, value in seed_differences.items()
            },
            "paired_bootstrap": bootstrap,
            "methods": summary_methods,
            "interpretation_boundary": (
                "Machine-decodability under simulated mappings only; no claim about "
                "human phosphene perception or implanted-user reading."
            ),
        }
        if not np.isclose(
            bootstrap["mean_difference_pp"],
            summary["primary_difference_pp"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise MmaConfirmationError(
                "bootstrap mean differs from the primary paired difference"
            )
        _write_csv(
            temporary / "predictions.csv",
            (
                "mapping_seed",
                "method",
                "sample_id",
                "font_id",
                "class_id",
                "character",
                "predicted_class_id",
                "correct",
            ),
            predictions,
        )
        _write_csv(
            temporary / "fold_metrics.csv",
            ("mapping_seed", "font_id", "method", "sample_count", "accuracy", "macro_f1"),
            fold_metrics,
        )
        _atomic_text(
            temporary / "summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        output_hashes = {
            name: sha256_file(temporary / name)
            for name in ("predictions.csv", "fold_metrics.csv", "summary.json")
        }
        manifest = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "status": "PASS_COMMITTED",
            "preflight": preflight,
            "config_path": str(config.source_path),
            "config_sha256": sha256_file(config.source_path),
            "source_tree_sha256": source_sha,
            "confirmation_lock_path": str(config.lock_path),
            "confirmation_lock_sha256": sha256_file(config.lock_path),
            "train_dataset_sha256": sha256_file(config.train_dataset),
            "test_asset_manifest_sha256": sha256_file(config.test_asset_manifest),
            "test_dataset_sha256": sha256_file(config.test_dataset),
            "train_selections_sha256": array_sha256(train_selected),
            "test_selections_sha256": array_sha256(test_selected),
            "environment": _runtime_identity(),
            "methods": list(METHODS),
            "primary_comparison": f"{PRIMARY_METHOD}-{PRIMARY_BASELINE}",
            "parameter_search": False,
            "test_font_count": len(config.test_fonts),
            "test_mapping_count": len(config.test_mapping_paths),
            "output_sha256": output_hashes,
        }
        _atomic_text(
            temporary / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        manifest_sha256 = sha256_file(temporary / "manifest.json")
        completion = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "status": "PASS_COMMITTED",
            "manifest_sha256": manifest_sha256,
            "config_sha256": sha256_file(config.source_path),
            "source_tree_sha256": source_sha,
            "confirmation_lock_sha256": sha256_file(config.lock_path),
        }
        _atomic_text(
            temporary / "completion.json",
            json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        temporary.replace(config.output_root)
        _atomic_text(
            completion_path,
            json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return summary | {"output_root": str(config.output_root)}
    except BaseException:
        if temporary.exists():
            if temporary.resolve().parent != output_parent.resolve():
                raise MmaConfirmationError(
                    "temporary confirmation directory escaped the intended output parent"
                )
            shutil.rmtree(temporary)
        raise
