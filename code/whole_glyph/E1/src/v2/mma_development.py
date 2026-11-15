"""Development-only cross-validation for multi-mapping cosine prototypes.

The D2 protocol uses only the six development fonts.  Every glyph is encoded
once with the ideal Skeleton-TopK selector at B=64, then rendered and described
with frozen HOG under the ideal mapping and five moderate mappings.  The
resulting feature bank is reused for the 6-font x 5-mapping cross-validation.

Old test fonts may occur in the source archive so their metadata can be
identified and excluded, but their image rows are never selected into the
development array, training data, predictions, or metrics.
"""

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from src.deterministic_io import array_sha256, sha256_file
from src.encoders.skeleton_topk import SkeletonTopKEncoder
from src.features import HogFeatureExtractor
from src.grid import make_regular_grid
from src.spv_forward import (
    GaussianPhospheneForwardModel,
    IdealGaussianUtilityProjector,
    expected_output_energy,
)
from src.v2.experiment import ConditionSpec, load_ideal_mapping, load_mapping
from src.v2.multimapping_prototypes import (
    MultiMappingCosinePrototypeClassifier,
    MultiMappingPrototypeError,
)


MMA_METHODS = (
    "ideal_only",
    "single_moderate",
    "multi4",
    "ideal_plus_multi4",
    "oracle_matched",
)
MMA_MAPPING_SEEDS = (2026, 2027, 2028, 2029, 2030)
MMA_BUDGET = 64
EXPECTED_IMAGE_SHAPE = (96, 96)
EXPECTED_CANDIDATES = 256
EXPECTED_DEVELOPMENT_FONTS = 6
KNOWN_OLD_TEST_FONTS = ("long_cang", "zhi_mang_xing")


class MmaDevelopmentError(ValueError):
    """Raised when the frozen D2 development protocol is violated."""


@dataclass(frozen=True, slots=True)
class MmaDevelopmentConfig:
    source_path: Path
    project_root: Path
    experiment_id: str
    legacy_root: Path
    dataset_path: Path
    expected_dataset_sha256: str
    mapping_dir: Path
    expected_mapping_sha256: Mapping[str, str]
    output_root: Path
    development_fonts: tuple[str, ...]
    locked_test_fonts: tuple[str, ...]
    mapping_seeds: tuple[int, ...]
    methods: tuple[str, ...]
    budget: int
    feature: Mapping[str, Any]
    max_blas_threads: int
    enforce_environment: bool
    expected_environment: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MmaDevelopmentDataset:
    images: NDArray[np.float32]
    class_ids: NDArray[np.int64]
    sample_ids: NDArray[np.str_]
    characters: NDArray[np.str_]
    font_ids: NDArray[np.str_]
    source_indices: NDArray[np.int64]
    locked_test_metadata_count: int


@dataclass(frozen=True, slots=True)
class MmaFold:
    fold_id: str
    heldout_font: str
    heldout_mapping_seed: int


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MmaDevelopmentError(f"{name} must be a mapping")
    return value


def _require_string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise MmaDevelopmentError(f"{name} must be a non-empty list")
    result = tuple(map(str, value))
    if any(not item or item.strip() != item for item in result):
        raise MmaDevelopmentError(f"{name} contains an empty or padded value")
    if len(set(result)) != len(result):
        raise MmaDevelopmentError(f"{name} contains duplicate values")
    return result


def _require_sha256(value: Any, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
        raise MmaDevelopmentError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _resolve_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MmaDevelopmentError(f"{name} must be a non-empty path string")
    raw = value.strip()
    if os.name != "nt" and len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
        candidate = Path("/mnt") / raw[0].lower() / raw[3:].replace("\\", "/")
    else:
        candidate = Path(raw)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def load_mma_development_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> MmaDevelopmentConfig:
    """Load the D2 YAML without accessing dataset pixels or writing outputs."""

    source = Path(path).resolve()
    if not source.is_file():
        raise MmaDevelopmentError(f"config file does not exist: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    document = _require_mapping(raw, "config")
    if int(document.get("schema_version", -1)) != 1:
        raise MmaDevelopmentError("schema_version must equal 1")
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    experiment_id = str(document.get("experiment_id", source.stem)).strip()
    if not experiment_id or not all(item.isalnum() or item in "-_" for item in experiment_id):
        raise MmaDevelopmentError("experiment_id must be filesystem-safe")

    legacy_root = _resolve_path(root, document.get("legacy_root"), "legacy_root")
    dataset_cfg = _require_mapping(document.get("dataset"), "dataset")
    mapping_cfg = _require_mapping(document.get("mapping"), "mapping")
    protocol_cfg = _require_mapping(document.get("protocol"), "protocol")
    feature_cfg = _require_mapping(document.get("feature"), "feature")
    runtime_cfg = _require_mapping(document.get("runtime", {}), "runtime")
    environment_cfg = _require_mapping(document.get("environment", {}), "environment")

    dataset_path = _resolve_path(
        legacy_root, dataset_cfg.get("glyphs_npz"), "dataset.glyphs_npz"
    )
    dataset_hash = _require_sha256(dataset_cfg.get("sha256"), "dataset.sha256")
    development_fonts = _require_string_list(
        dataset_cfg.get("development_fonts"), "dataset.development_fonts"
    )
    locked_test_fonts = _require_string_list(
        dataset_cfg.get("locked_test_fonts"), "dataset.locked_test_fonts"
    )
    if len(development_fonts) != EXPECTED_DEVELOPMENT_FONTS:
        raise MmaDevelopmentError("D2 requires exactly six development fonts")
    if set(development_fonts) & set(locked_test_fonts):
        raise MmaDevelopmentError("development and locked-test fonts overlap")
    if tuple(locked_test_fonts) != KNOWN_OLD_TEST_FONTS:
        raise MmaDevelopmentError(
            f"locked_test_fonts must equal the frozen old-test list {KNOWN_OLD_TEST_FONTS}"
        )

    mapping_dir = _resolve_path(
        legacy_root, mapping_cfg.get("directory"), "mapping.directory"
    )
    raw_seeds = mapping_cfg.get("moderate_seeds")
    if not isinstance(raw_seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds
    ):
        raise MmaDevelopmentError("mapping.moderate_seeds must be an integer list")
    mapping_seeds = tuple(raw_seeds)
    if mapping_seeds != MMA_MAPPING_SEEDS:
        raise MmaDevelopmentError(
            f"D2 moderate seeds must equal {MMA_MAPPING_SEEDS} in this order"
        )
    raw_mapping_hashes = _require_mapping(mapping_cfg.get("sha256"), "mapping.sha256")
    mapping_hashes = {
        str(filename): _require_sha256(value, f"mapping.sha256.{filename}")
        for filename, value in raw_mapping_hashes.items()
    }
    expected_mapping_names = {
        "ideal_mapping.npz",
        *(f"moderate_seed_{seed}.npz" for seed in mapping_seeds),
    }
    if set(mapping_hashes) != expected_mapping_names:
        raise MmaDevelopmentError(
            "mapping.sha256 must bind exactly ideal plus seeds 2026--2030"
        )

    if int(protocol_cfg.get("budget", -1)) != MMA_BUDGET:
        raise MmaDevelopmentError("D2 budget must equal 64")
    if str(protocol_cfg.get("encoder", "")) != "ideal_skeleton_topk":
        raise MmaDevelopmentError("D2 encoder must be ideal_skeleton_topk")
    if str(protocol_cfg.get("feature", "")) != "hog":
        raise MmaDevelopmentError("D2 protocol feature must be hog")
    if str(protocol_cfg.get("classifier", "")) != "multimapping_cosine_prototype":
        raise MmaDevelopmentError(
            "D2 classifier must be multimapping_cosine_prototype"
        )
    methods = _require_string_list(protocol_cfg.get("methods"), "protocol.methods")
    if methods != MMA_METHODS:
        raise MmaDevelopmentError(
            f"D2 methods must equal {MMA_METHODS} in this order"
        )

    output_root = _resolve_path(root, document.get("output_root"), "output_root")
    if output_root == root or not _is_relative_to(output_root, root):
        raise MmaDevelopmentError("output_root must be a child of the project root")
    max_blas_threads = int(runtime_cfg.get("max_blas_threads", 1))
    if max_blas_threads <= 0:
        raise MmaDevelopmentError("runtime.max_blas_threads must be positive")
    enforce_environment = environment_cfg.get("enforce", False)
    if not isinstance(enforce_environment, bool):
        raise MmaDevelopmentError("environment.enforce must be true or false")
    expected_environment: dict[str, str] = {}
    if "python" in environment_cfg:
        expected_environment["python"] = str(environment_cfg["python"])
    packages = _require_mapping(environment_cfg.get("packages", {}), "environment.packages")
    expected_environment.update(
        {f"package:{name}": str(version) for name, version in packages.items()}
    )
    if enforce_environment and not expected_environment:
        raise MmaDevelopmentError("enforced environment has no version bindings")

    return MmaDevelopmentConfig(
        source_path=source,
        project_root=root,
        experiment_id=experiment_id,
        legacy_root=legacy_root,
        dataset_path=dataset_path,
        expected_dataset_sha256=dataset_hash,
        mapping_dir=mapping_dir,
        expected_mapping_sha256=mapping_hashes,
        output_root=output_root,
        development_fonts=development_fonts,
        locked_test_fonts=locked_test_fonts,
        mapping_seeds=mapping_seeds,
        methods=methods,
        budget=MMA_BUDGET,
        feature=dict(feature_cfg),
        max_blas_threads=max_blas_threads,
        enforce_environment=enforce_environment,
        expected_environment=expected_environment,
    )


# Short aliases make the module convenient without weakening the explicit API.
load_development_config = load_mma_development_config


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
    value = dict(identity) if identity is not None else _runtime_identity()
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_runtime_environment(config: MmaDevelopmentConfig) -> dict[str, Any]:
    identity = _runtime_identity()
    if not config.enforce_environment:
        return identity
    mismatches: list[str] = []
    for key, expected in config.expected_environment.items():
        if key == "python":
            actual = str(identity["python"])
        elif key.startswith("package:"):
            actual = str(identity["packages"].get(key.split(":", 1)[1], "MISSING"))
        else:
            raise MmaDevelopmentError(f"unsupported environment binding: {key}")
        if actual != expected:
            mismatches.append(f"{key} expected {expected}, found {actual}")
    if mismatches:
        raise MmaDevelopmentError("environment mismatch: " + "; ".join(mismatches))
    return identity


def _mapping_path(config: MmaDevelopmentConfig, seed: int | None) -> Path:
    if seed is None:
        return config.mapping_dir / "ideal_mapping.npz"
    if seed not in config.mapping_seeds:
        raise MmaDevelopmentError(f"unfrozen moderate mapping seed: {seed}")
    return config.mapping_dir / f"moderate_seed_{seed}.npz"


def build_cv_folds(config: MmaDevelopmentConfig) -> tuple[MmaFold, ...]:
    folds = tuple(
        MmaFold(
            fold_id=f"font-{font_id}__map-{seed}",
            heldout_font=font_id,
            heldout_mapping_seed=seed,
        )
        for font_id in config.development_fonts
        for seed in config.mapping_seeds
    )
    if len(folds) != 30 or len({fold.fold_id for fold in folds}) != 30:
        raise MmaDevelopmentError("D2 must produce 30 unique CV folds")
    return folds


def training_mapping_ids(
    method: str,
    heldout_mapping_seed: int,
    mapping_seeds: Sequence[int] = MMA_MAPPING_SEEDS,
) -> tuple[str, ...]:
    """Resolve the frozen prototype sources for one mapping-held-out fold."""

    seeds = tuple(int(seed) for seed in mapping_seeds)
    if seeds != MMA_MAPPING_SEEDS:
        raise MmaDevelopmentError("mapping seed order differs from D2")
    if heldout_mapping_seed not in seeds:
        raise MmaDevelopmentError("heldout mapping seed is not in D2")
    nonheldout = tuple(seed for seed in seeds if seed != heldout_mapping_seed)
    moderate_ids = tuple(f"moderate_seed{seed}" for seed in nonheldout)
    heldout_id = f"moderate_seed{heldout_mapping_seed}"
    if method == "ideal_only":
        return ("ideal",)
    if method == "single_moderate":
        return (moderate_ids[0],)
    if method == "multi4":
        return moderate_ids
    if method == "ideal_plus_multi4":
        return ("ideal", *moderate_ids)
    if method == "oracle_matched":
        return (heldout_id,)
    raise MmaDevelopmentError(f"unsupported MMA method: {method}")


def _input_hashes(config: MmaDevelopmentConfig) -> dict[str, str]:
    result = {
        "config_sha256": sha256_file(config.source_path),
        "dataset_sha256": sha256_file(config.dataset_path),
    }
    for seed in (None, *config.mapping_seeds):
        path = _mapping_path(config, seed)
        result[f"mapping:{path.name}"] = sha256_file(path)
    return result


def validate_mma_plan_files(config: MmaDevelopmentConfig) -> dict[str, Any]:
    """Validate environment, paths and all frozen hashes without writing."""

    identity = validate_runtime_environment(config)
    if not config.legacy_root.is_dir():
        raise MmaDevelopmentError(f"legacy_root does not exist: {config.legacy_root}")
    if not config.dataset_path.is_file():
        raise MmaDevelopmentError(
            f"development glyph archive does not exist: {config.dataset_path}"
        )
    if sha256_file(config.dataset_path) != config.expected_dataset_sha256:
        raise MmaDevelopmentError("glyph archive SHA-256 differs from D2 config")
    mapping_hashes: dict[str, str] = {}
    for seed in (None, *config.mapping_seeds):
        path = _mapping_path(config, seed)
        if not path.is_file():
            raise MmaDevelopmentError(f"mapping file does not exist: {path}")
        actual = sha256_file(path)
        expected = config.expected_mapping_sha256.get(path.name)
        if expected != actual:
            raise MmaDevelopmentError(f"mapping SHA-256 differs from D2 config: {path.name}")
        mapping_hashes[path.name] = actual
    folds = build_cv_folds(config)
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "VALID_DRY_RUN_PLAN",
        "output_root": str(config.output_root),
        "budget": config.budget,
        "development_fonts": list(config.development_fonts),
        "locked_test_fonts_excluded": list(config.locked_test_fonts),
        "mapping_seeds": list(config.mapping_seeds),
        "methods": list(config.methods),
        "fold_count": len(folds),
        "folds": [
            {
                "fold_id": fold.fold_id,
                "heldout_font": fold.heldout_font,
                "heldout_mapping_seed": fold.heldout_mapping_seed,
                "training_mappings": {
                    method: list(
                        training_mapping_ids(method, fold.heldout_mapping_seed)
                    )
                    for method in config.methods
                },
            }
            for fold in folds
        ],
        "hashes": {
            "dataset": config.expected_dataset_sha256,
            "mappings": mapping_hashes,
        },
        "environment_fingerprint_sha256": _runtime_fingerprint(identity),
    }


validate_plan_files = validate_mma_plan_files


def load_development_dataset(config: MmaDevelopmentConfig) -> MmaDevelopmentDataset:
    """Load only rows belonging to the six development fonts."""

    with np.load(config.dataset_path, allow_pickle=False) as archive:
        required = {"images", "class_ids", "sample_ids", "characters", "font_ids"}
        if not required <= set(archive.files):
            raise MmaDevelopmentError("glyph archive is missing required arrays")
        raw_class_ids = archive["class_ids"]
        if raw_class_ids.dtype != np.int64 or raw_class_ids.ndim != 1:
            raise MmaDevelopmentError("class_ids must be a one-dimensional int64 array")
        class_ids_all = np.ascontiguousarray(raw_class_ids)
        sample_ids_all = np.asarray(archive["sample_ids"], dtype=np.str_)
        characters_all = np.asarray(archive["characters"], dtype=np.str_)
        font_ids_all = np.asarray(archive["font_ids"], dtype=np.str_)
        count = class_ids_all.size
        if any(
            array.shape != (count,)
            for array in (sample_ids_all, characters_all, font_ids_all)
        ):
            raise MmaDevelopmentError("glyph metadata lengths differ")
        if count == 0 or len(set(map(str, sample_ids_all))) != count:
            raise MmaDevelopmentError("sample IDs are empty or non-unique")
        observed_fonts = set(map(str, np.unique(font_ids_all)))
        expected_or_excluded = set(config.development_fonts) | set(config.locked_test_fonts)
        unexpected = observed_fonts - expected_or_excluded
        if unexpected:
            raise MmaDevelopmentError(
                "glyph archive contains fonts outside development/locked-test bindings: "
                + ", ".join(sorted(unexpected))
            )
        missing_development = set(config.development_fonts) - observed_fonts
        if missing_development:
            raise MmaDevelopmentError(
                "glyph archive is missing development fonts: "
                + ", ".join(sorted(missing_development))
            )
        development_mask = np.isin(font_ids_all, config.development_fonts)
        locked_mask = np.isin(font_ids_all, config.locked_test_fonts)
        if np.any(development_mask & locked_mask):
            raise MmaDevelopmentError("development and locked-test masks overlap")
        source_indices = np.flatnonzero(development_mask).astype(np.int64)
        if source_indices.size == 0:
            raise MmaDevelopmentError("development selection is empty")

        image_member = archive["images"]
        if (
            image_member.dtype != np.float32
            or image_member.ndim != 3
            or image_member.shape != (count, *EXPECTED_IMAGE_SHAPE)
        ):
            raise MmaDevelopmentError("glyph images must be float32 (N,96,96)")
        # This is the sole image selection: only the prevalidated development rows.
        images = np.ascontiguousarray(image_member[source_indices], dtype=np.float32)
        class_ids = np.ascontiguousarray(class_ids_all[source_indices], dtype=np.int64)
        sample_ids = np.asarray(sample_ids_all[source_indices], dtype=np.str_).copy()
        characters = np.asarray(characters_all[source_indices], dtype=np.str_).copy()
        font_ids = np.asarray(font_ids_all[source_indices], dtype=np.str_).copy()

    if np.any(np.isin(font_ids, config.locked_test_fonts)):
        raise MmaDevelopmentError("a locked-test font entered the development array")
    if not np.isfinite(images).all() or float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise MmaDevelopmentError("development images contain invalid pixels")
    if np.any(images.max(axis=(1, 2)) <= 0.0):
        raise MmaDevelopmentError("development images contain an empty glyph")

    class_order = np.unique(class_ids)
    if class_order.size < 2 or np.any(class_order < 0):
        raise MmaDevelopmentError("development data requires at least two non-negative classes")
    expected_pairs = {
        (font_id, int(class_id))
        for font_id in config.development_fonts
        for class_id in class_order
    }
    observed_pairs = [
        (str(font_id), int(class_id))
        for font_id, class_id in zip(font_ids, class_ids, strict=True)
    ]
    if set(observed_pairs) != expected_pairs or len(observed_pairs) != len(expected_pairs):
        raise MmaDevelopmentError(
            "each development font must contain exactly one glyph for every class"
        )
    for class_id in class_order:
        class_characters = np.unique(characters[class_ids == class_id])
        if class_characters.shape != (1,):
            raise MmaDevelopmentError(f"class {int(class_id)} has inconsistent characters")

    return MmaDevelopmentDataset(
        images=images,
        class_ids=class_ids,
        sample_ids=sample_ids,
        characters=characters,
        font_ids=font_ids,
        source_indices=np.ascontiguousarray(source_indices),
        locked_test_metadata_count=int(np.count_nonzero(locked_mask)),
    )


def _load_protocol_mappings(config: MmaDevelopmentConfig) -> dict[str, Any]:
    ideal = load_ideal_mapping(_mapping_path(config, None))
    grid = make_regular_grid(
        rows=16, cols=16, image_shape=EXPECTED_IMAGE_SHAPE
    ).electrodes
    if not np.array_equal(ideal.centers_xy, grid.centers_xy):
        raise MmaDevelopmentError("ideal mapping differs from the regular 16x16 grid")
    mappings: dict[str, Any] = {ideal.mapping_id: ideal}
    for seed in config.mapping_seeds:
        condition = ConditionSpec(
            method="skeleton_topk",
            budget=MMA_BUDGET,
            mapping_type="moderate",
            mapping_seed=seed,
        )
        mapping = load_mapping(_mapping_path(config, seed), condition=condition)
        expected_id = f"moderate_seed{seed}"
        if mapping.mapping_id != expected_id:
            raise MmaDevelopmentError(
                f"mapping identity differs from D2 protocol: {mapping.mapping_id}"
            )
        mappings[mapping.mapping_id] = mapping
    expected_ids = {"ideal", *(f"moderate_seed{seed}" for seed in config.mapping_seeds)}
    if set(mappings) != expected_ids:
        raise MmaDevelopmentError("loaded mapping IDs differ from D2 protocol")
    return mappings


def _build_hog(config: MmaDevelopmentConfig) -> HogFeatureExtractor:
    pixels = tuple(map(int, config.feature.get("pixels_per_cell", (8, 8))))
    cells = tuple(map(int, config.feature.get("cells_per_block", (2, 2))))
    return HogFeatureExtractor(
        image_shape=EXPECTED_IMAGE_SHAPE,
        orientations=int(config.feature.get("orientations", 9)),
        pixels_per_cell=pixels,
        cells_per_block=cells,
        block_norm=str(config.feature.get("block_norm", "L2-Hys")),
    )


def _ideal_skeleton_selections(
    dataset: MmaDevelopmentDataset,
    ideal_mapping: Any,
) -> NDArray[np.int64]:
    grid = make_regular_grid(
        rows=16, cols=16, image_shape=EXPECTED_IMAGE_SHAPE
    ).electrodes
    encoder = SkeletonTopKEncoder()
    projector = IdealGaussianUtilityProjector(ideal_mapping)
    selections = np.empty((dataset.images.shape[0], MMA_BUDGET), dtype=np.int64)
    for index, image in enumerate(dataset.images):
        result = encoder.encode(image, grid, MMA_BUDGET, projector=projector)
        selections[index] = result.selected_indices
    if any(np.unique(row).size != MMA_BUDGET for row in selections):
        raise MmaDevelopmentError("SkeletonTopK produced duplicate electrodes")
    if np.any(selections < 0) or np.any(selections >= EXPECTED_CANDIDATES):
        raise MmaDevelopmentError("SkeletonTopK produced an out-of-range electrode")
    return np.ascontiguousarray(selections, dtype=np.int64)


def build_feature_bank(
    config: MmaDevelopmentConfig,
    dataset: MmaDevelopmentDataset,
    mappings: Mapping[str, Any],
) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
    """Build the six mapping feature matrices used by every CV fold."""

    ideal = mappings.get("ideal")
    if ideal is None:
        raise MmaDevelopmentError("feature bank requires the ideal mapping")
    selections = _ideal_skeleton_selections(dataset, ideal)
    amplitudes = np.ones(selections.shape, dtype=np.float32)
    extractor = _build_hog(config)
    mapping_order = ("ideal", *(f"moderate_seed{seed}" for seed in config.mapping_seeds))
    feature_bank: dict[str, NDArray[np.float32]] = {}
    feature_hashes: dict[str, str] = {}
    energy_audit: dict[str, Any] = {}
    for mapping_id in mapping_order:
        mapping = mappings.get(mapping_id)
        if mapping is None:
            raise MmaDevelopmentError(f"feature mapping is missing: {mapping_id}")
        forward = GaussianPhospheneForwardModel(mapping)
        spv_images = forward.generate_variable_batch(selections, amplitudes)
        energy = spv_images.sum(axis=(1, 2), dtype=np.float64)
        expected = expected_output_energy(
            amplitudes, amplitude_scale=forward.amplitude_scale
        )
        if not np.allclose(energy, expected, rtol=0.0, atol=2e-5):
            raise MmaDevelopmentError(f"SPV energy audit failed for {mapping_id}")
        features = extractor.extract_batch(spv_images)
        if np.any(np.linalg.norm(features.astype(np.float64), axis=1) == 0.0):
            raise MmaDevelopmentError(f"HOG returned a zero vector for {mapping_id}")
        feature_bank[mapping_id] = np.ascontiguousarray(features, dtype=np.float32)
        feature_hashes[mapping_id] = array_sha256(features)
        energy_audit[mapping_id] = {
            "expected": float(MMA_BUDGET),
            "observed_min": float(energy.min()),
            "observed_max": float(energy.max()),
        }
    return feature_bank, {
        "encoder": "ideal_skeleton_topk",
        "budget": MMA_BUDGET,
        "selection_sha256": array_sha256(selections),
        "feature": "hog",
        "feature_length": extractor.feature_length,
        "feature_sha256": feature_hashes,
        "energy": energy_audit,
    }


def _prototype_training_arrays(
    dataset: MmaDevelopmentDataset,
    feature_bank: Mapping[str, NDArray[np.float32]],
    *,
    train_indices: NDArray[np.int64],
    mapping_ids: Sequence[str],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int64],
    NDArray[np.str_],
    NDArray[np.str_],
]:
    matrices: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.int64]] = []
    font_ids: list[NDArray[np.str_]] = []
    mapping_rows: list[NDArray[np.str_]] = []
    for mapping_id in mapping_ids:
        features = feature_bank.get(mapping_id)
        if features is None:
            raise MmaDevelopmentError(f"prototype feature mapping is missing: {mapping_id}")
        matrices.append(np.ascontiguousarray(features[train_indices], dtype=np.float32))
        labels.append(np.ascontiguousarray(dataset.class_ids[train_indices], dtype=np.int64))
        font_ids.append(np.asarray(dataset.font_ids[train_indices], dtype=np.str_))
        mapping_rows.append(
            np.asarray([mapping_id] * int(train_indices.size), dtype=np.str_)
        )
    return (
        np.ascontiguousarray(np.concatenate(matrices, axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(labels), dtype=np.int64),
        np.ascontiguousarray(np.concatenate(font_ids), dtype=np.str_),
        np.ascontiguousarray(np.concatenate(mapping_rows), dtype=np.str_),
    )


def run_cross_validation(
    config: MmaDevelopmentConfig,
    dataset: MmaDevelopmentDataset,
    feature_bank: Mapping[str, NDArray[np.float32]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run all 150 method-fold evaluations and return deterministic rows."""

    class_order = tuple(map(int, np.unique(dataset.class_ids)))
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in build_cv_folds(config):
        train_indices = np.flatnonzero(dataset.font_ids != fold.heldout_font).astype(np.int64)
        validation_indices = np.flatnonzero(dataset.font_ids == fold.heldout_font).astype(np.int64)
        if set(map(str, np.unique(dataset.font_ids[train_indices]))) != (
            set(config.development_fonts) - {fold.heldout_font}
        ):
            raise MmaDevelopmentError("fold training fonts differ from the five-font complement")
        if set(map(str, np.unique(dataset.font_ids[validation_indices]))) != {fold.heldout_font}:
            raise MmaDevelopmentError("fold validation contains a non-heldout font")
        heldout_mapping_id = f"moderate_seed{fold.heldout_mapping_seed}"
        validation_features = np.ascontiguousarray(
            feature_bank[heldout_mapping_id][validation_indices], dtype=np.float32
        )
        truth = np.ascontiguousarray(
            dataset.class_ids[validation_indices], dtype=np.int64
        )
        for method in config.methods:
            source_mapping_ids = training_mapping_ids(
                method, fold.heldout_mapping_seed, config.mapping_seeds
            )
            training = _prototype_training_arrays(
                dataset,
                feature_bank,
                train_indices=train_indices,
                mapping_ids=source_mapping_ids,
            )
            classifier = MultiMappingCosinePrototypeClassifier(class_order)
            try:
                classifier.fit(*training)
                predictions = classifier.predict(validation_features)
            except MultiMappingPrototypeError as error:
                raise MmaDevelopmentError(
                    f"prototype failure in {fold.fold_id}/{method}: {error}"
                ) from error
            accuracy = float(np.mean(truth == predictions))
            macro_f1 = float(
                f1_score(truth, predictions, average="macro", zero_division=0)
            )
            metric_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "method": method,
                    "heldout_font": fold.heldout_font,
                    "heldout_mapping_seed": fold.heldout_mapping_seed,
                    "heldout_mapping_id": heldout_mapping_id,
                    "training_mapping_ids": "|".join(source_mapping_ids),
                    "train_font_count": 5,
                    "train_mapping_count": len(source_mapping_ids),
                    "train_sample_count": int(training[0].shape[0]),
                    "validation_sample_count": int(validation_indices.size),
                    "class_count": len(class_order),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                }
            )
            for source_index, prediction in zip(
                validation_indices, predictions, strict=True
            ):
                true_label = int(dataset.class_ids[source_index])
                predicted_label = int(prediction)
                prediction_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "method": method,
                        "heldout_font": fold.heldout_font,
                        "heldout_mapping_seed": fold.heldout_mapping_seed,
                        "heldout_mapping_id": heldout_mapping_id,
                        "training_mapping_ids": "|".join(source_mapping_ids),
                        "sample_id": str(dataset.sample_ids[source_index]),
                        "character": str(dataset.characters[source_index]),
                        "font_id": str(dataset.font_ids[source_index]),
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "correct": int(true_label == predicted_label),
                    }
                )

    expected_metrics = 30 * len(config.methods)
    expected_predictions = expected_metrics * len(class_order)
    if len(metric_rows) != expected_metrics or len(prediction_rows) != expected_predictions:
        raise MmaDevelopmentError("cross-validation row counts differ from the D2 design")
    if any(row["font_id"] in config.locked_test_fonts for row in prediction_rows):
        raise MmaDevelopmentError("a locked-test font entered D2 predictions")

    method_summary: dict[str, Any] = {}
    for method in config.methods:
        rows = [row for row in metric_rows if row["method"] == method]
        accuracy = np.asarray([row["accuracy"] for row in rows], dtype=np.float64)
        macro_f1 = np.asarray([row["macro_f1"] for row in rows], dtype=np.float64)
        method_summary[method] = {
            "fold_count": len(rows),
            "mean_accuracy": float(accuracy.mean()),
            "std_accuracy_population": float(accuracy.std(ddof=0)),
            "min_accuracy": float(accuracy.min()),
            "max_accuracy": float(accuracy.max()),
            "mean_macro_f1": float(macro_f1.mean()),
            "std_macro_f1_population": float(macro_f1.std(ddof=0)),
            "min_macro_f1": float(macro_f1.min()),
            "max_macro_f1": float(macro_f1.max()),
        }
    summary = {
        "schema_version": 1,
        "status": "PASS_DEVELOPMENT_CV",
        "experiment_id": config.experiment_id,
        "protocol": {
            "development_only": True,
            "development_font_count": EXPECTED_DEVELOPMENT_FONTS,
            "heldout_font_count_per_fold": 1,
            "moderate_mapping_seeds": list(config.mapping_seeds),
            "heldout_mapping_count_per_fold": 1,
            "fold_count": 30,
            "budget": MMA_BUDGET,
            "encoder": "ideal_skeleton_topk",
            "feature": "hog",
            "classifier": "multimapping_cosine_prototype",
            "fold_aggregation": "unweighted_mean_over_30_font_x_mapping_folds",
            "oracle_matched_role": "mapping-matched diagnostic upper control",
        },
        "methods": method_summary,
    }
    return prediction_rows, metric_rows, summary


def _json_bytes(value: Any) -> bytes:
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


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _source_tree_sha256(project_root: Path) -> str:
    paths = sorted((project_root / "src").rglob("*.py"))
    entrypoint = project_root / "run_mma_development.py"
    if entrypoint.is_file():
        paths.append(entrypoint)
    if not paths:
        raise MmaDevelopmentError("no Python sources found for source binding")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _safe_remove_existing(path: Path, project_root: Path) -> None:
    resolved = path.resolve()
    root = project_root.resolve()
    if resolved == root or not _is_relative_to(resolved, root) or not resolved.name:
        raise MmaDevelopmentError("refusing to remove output outside project root")
    shutil.rmtree(resolved)


def execute_mma_development(
    config: MmaDevelopmentConfig,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute D2 and atomically commit exactly four auditable outputs."""

    validate_mma_plan_files(config)
    runtime_before = validate_runtime_environment(config)
    runtime_fingerprint_before = _runtime_fingerprint(runtime_before)
    source_root = Path(__file__).resolve().parents[2]
    source_hash_before = _source_tree_sha256(source_root)
    input_hashes_before = _input_hashes(config)
    output_root = config.output_root
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        if not overwrite:
            raise MmaDevelopmentError(
                f"D2 output already exists (use --overwrite deliberately): {output_root}"
            )
        _safe_remove_existing(output_root, config.project_root)
    staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        dataset = load_development_dataset(config)
        mappings = _load_protocol_mappings(config)
        with threadpool_limits(limits=config.max_blas_threads):
            feature_bank, feature_audit = build_feature_bank(
                config, dataset, mappings
            )
            predictions, fold_metrics, summary = run_cross_validation(
                config, dataset, feature_bank
            )
        elapsed = time.perf_counter() - started
        summary = dict(summary)
        summary["elapsed_seconds"] = elapsed
        _write_csv_atomic(
            staging / "predictions.csv",
            (
                "fold_id",
                "method",
                "heldout_font",
                "heldout_mapping_seed",
                "heldout_mapping_id",
                "training_mapping_ids",
                "sample_id",
                "character",
                "font_id",
                "true_label",
                "predicted_label",
                "correct",
            ),
            predictions,
        )
        _write_csv_atomic(
            staging / "fold_metrics.csv",
            (
                "fold_id",
                "method",
                "heldout_font",
                "heldout_mapping_seed",
                "heldout_mapping_id",
                "training_mapping_ids",
                "train_font_count",
                "train_mapping_count",
                "train_sample_count",
                "validation_sample_count",
                "class_count",
                "accuracy",
                "macro_f1",
            ),
            fold_metrics,
        )
        _write_json_atomic(staging / "summary.json", summary)

        if _input_hashes(config) != input_hashes_before:
            raise MmaDevelopmentError("a frozen D2 input changed during execution")
        if _source_tree_sha256(source_root) != source_hash_before:
            raise MmaDevelopmentError("Python source changed during D2 execution")
        runtime_after = validate_runtime_environment(config)
        if _runtime_fingerprint(runtime_after) != runtime_fingerprint_before:
            raise MmaDevelopmentError("runtime environment changed during D2 execution")
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        if set(output_hashes) != {
            "predictions.csv",
            "fold_metrics.csv",
            "summary.json",
        }:
            raise MmaDevelopmentError("pre-manifest D2 output set is incomplete")
        manifest = {
            "schema_version": 1,
            "status": "PASS_COMMITTED",
            "experiment_id": config.experiment_id,
            "started_utc": started_utc,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "protocol": summary["protocol"],
            "paths": {
                "config": str(config.source_path),
                "dataset": str(config.dataset_path),
                "mapping_directory": str(config.mapping_dir),
                "output": str(output_root),
            },
            "dataset_audit": {
                "development_fonts": list(config.development_fonts),
                "development_sample_count": int(dataset.images.shape[0]),
                "development_class_count": int(np.unique(dataset.class_ids).size),
                "development_source_indices_sha256": array_sha256(dataset.source_indices),
                "locked_test_fonts": list(config.locked_test_fonts),
                "locked_test_metadata_count": dataset.locked_test_metadata_count,
                "locked_test_image_rows_selected": 0,
                "locked_test_prediction_rows": 0,
            },
            "feature_audit": feature_audit,
            "row_counts": {
                "predictions": len(predictions),
                "fold_metrics": len(fold_metrics),
            },
            "hashes": {
                **input_hashes_before,
                "source_tree_sha256": source_hash_before,
                "outputs_before_manifest": output_hashes,
            },
            "environment": {
                **runtime_before,
                "python_build": sys.version,
                "fingerprint_sha256": runtime_fingerprint_before,
                "pid": os.getpid(),
            },
        }
        _write_json_atomic(staging / "manifest.json", manifest)
        if {path.name for path in staging.iterdir() if path.is_file()} != {
            "predictions.csv",
            "fold_metrics.csv",
            "summary.json",
            "manifest.json",
        }:
            raise MmaDevelopmentError("final D2 staging output set is not exact")
        staging.replace(output_root)
        return {
            "schema_version": 1,
            "status": "PASS_COMMITTED",
            "experiment_id": config.experiment_id,
            "output_root": str(output_root),
            "fold_count": 30,
            "method_fold_count": len(fold_metrics),
            "prediction_count": len(predictions),
            "elapsed_seconds": elapsed,
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


execute_development = execute_mma_development
