"""Stage-9 development-only EB weight selection and formal freeze helpers.

This module is intentionally outside ``src/encoders``.  It applies the only
parameters that Stage 5.5 allowed to vary (alpha_E and alpha_J) while reusing
the frozen topology, projector, Top-K, forward model, HOG, and LinearSVC code.
Locked-test glyphs and formal mapping seeds never enter the functions below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from math import log2
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.classifier import LinearSvcClassifier, classification_accuracy
from src.config import ExperimentConfig, load_config
from src.contracts import ElectrodeMapping, FloatImage
from src.deterministic_io import array_sha256, sha256_file, tree_sha256
from src.eb_drive import EbDriveComponents, build_eb_drive_components, max_normalize_nonzero
from src.encoders.base import deterministic_topk
from src.features import HogFeatureExtractor
from src.formal_benchmark import (
    BENCHMARK_ID,
    CHARACTER_SEQUENCE_SHA256,
    DEVELOPMENT_MAPPING_SEED,
    FORMAL_BUDGETS,
    FORMAL_FONT_SPECS,
    FORMAL_MAPPING_SEEDS,
    FORMAL_METHODS,
    STAGE5_5_TRUTH_TABLE_SHA256,
    validate_formal_config,
)
from src.grid import RegularGrid, make_regular_grid
from src.protocol import validate_stage5_5_truth_table
from src.spv_forward import (
    GaussianPhospheneForwardModel,
    IdealGaussianUtilityProjector,
    expected_output_energy,
)


ALPHA_VALUES = (0.5, 1.0, 2.0)
PREREGISTERED_ALPHA_PAIRS = tuple(
    (alpha_e, alpha_j)
    for alpha_e in ALPHA_VALUES
    for alpha_j in ALPHA_VALUES
)
DEVELOPMENT_TRAIN_FONTS = tuple(
    spec.font_id
    for spec in FORMAL_FONT_SPECS
    if spec.split_role == "development_train"
)
DEVELOPMENT_VALIDATION_FONTS = tuple(
    spec.font_id
    for spec in FORMAL_FONT_SPECS
    if spec.split_role == "development_validation"
)
LOCKED_TEST_FONTS = tuple(
    spec.font_id for spec in FORMAL_FONT_SPECS if spec.split_role == "locked_test"
)
STAGE8_MANIFEST_SHA256 = (
    "452b514b4a2c7e5dd7b10c13a37656571cf56181f0115b0801ef89287ea369ab"
)
STAGE8_ENCODER_BUNDLE_SHA256 = (
    "4183a48cf6c9e9001fceb0ad0bc39223f4bf16e06328d27db24e9cc8c4baa367"
)
STAGE8_FONT_SPLIT_SHA256 = (
    "15d1c65919160943e628d7b4d5ef084338a509e9990a15db0d84cdbbe200d3e4"
)
STAGE8_GLYPH_ARCHIVE_SHA256 = (
    "8427140836b2358aa48ff9c7bfc07cfdbe0f02ee166fb8b2e671ff7cb8a8d582"
)
DEVELOPMENT_MAPPING_SHA256 = (
    "0b93dcf767e9e8673d26f1db646aa7fb87127c997c8cf5a59182cf86fc4ab89f"
)
IDEAL_MAPPING_SHA256 = (
    "267cd5ebfaa9ff4093f950fbe6df66442cf3f5115419c677808c1db34c617d5a"
)


class Stage9Error(ValueError):
    """Raised before any Stage-9 protocol violation can affect a result."""


@dataclass(frozen=True, slots=True)
class Stage9Preflight:
    benchmark_id: str
    benchmark_manifest_sha256: str
    truth_table_sha256: str
    encoder_bundle_sha256: str
    character_sequence_sha256: str
    font_split_sha256: str
    glyph_archive_sha256: str
    ideal_mapping_sha256: str
    development_mapping_sha256: str
    formal_mapping_performance_access_count: int


@dataclass(frozen=True, slots=True)
class DevelopmentDataset:
    """Only the six development fonts; no locked-test image is retained."""

    images: NDArray[np.float32] = field(repr=False)
    class_ids: NDArray[np.int64] = field(repr=False)
    sample_ids: NDArray[np.str_] = field(repr=False)
    characters: NDArray[np.str_] = field(repr=False)
    font_ids: NDArray[np.str_] = field(repr=False)
    splits: NDArray[np.str_] = field(repr=False)
    train_indices: NDArray[np.int64] = field(repr=False)
    validation_indices: NDArray[np.int64] = field(repr=False)

    def __post_init__(self) -> None:
        if self.images.shape != (1080, 96, 96) or self.images.dtype != np.float32:
            raise Stage9Error("development image array must be float32 (1080,96,96)")
        if set(map(str, self.font_ids)) & set(LOCKED_TEST_FONTS):
            raise Stage9Error("locked-test font entered the development array")
        if set(map(str, self.splits)) != {"train", "validation"}:
            raise Stage9Error("development array contains an unexpected split")
        if self.train_indices.shape != (900,) or self.validation_indices.shape != (180,):
            raise Stage9Error("development split must contain exactly 900/180 samples")
        if np.intersect1d(self.train_indices, self.validation_indices).size:
            raise Stage9Error("development train and validation indices overlap")
        if set(map(int, self.class_ids[self.train_indices])) != set(range(180)):
            raise Stage9Error("development training fonts do not cover all 180 classes")
        if set(map(int, self.class_ids[self.validation_indices])) != set(range(180)):
            raise Stage9Error("development validation font does not cover all 180 classes")


@dataclass(frozen=True, slots=True)
class WeightedEbRepresentation:
    """One formally weighted D_EB and its shared ideal-projector utility."""

    drive_map: FloatImage = field(repr=False)
    candidate_scores: NDArray[np.float32] = field(repr=False)

    def __post_init__(self) -> None:
        if self.drive_map.shape != (96, 96) or self.drive_map.dtype != np.float32:
            raise Stage9Error("weighted D_EB must be a float32 96x96 image")
        if self.candidate_scores.shape != (256,) or self.candidate_scores.dtype != np.float32:
            raise Stage9Error("weighted candidate utility must be float32 length 256")
        if not np.isfinite(self.drive_map).all() or not np.isfinite(self.candidate_scores).all():
            raise Stage9Error("weighted EB representation contains non-finite values")
        if float(self.drive_map.min()) < 0.0 or float(self.drive_map.max()) != 1.0:
            raise Stage9Error("weighted D_EB must be non-negative and max-normalized")
        if float(self.candidate_scores.min()) < 0.0:
            raise Stage9Error("weighted candidate utility must be non-negative")


def encoder_bundle_paths(project_root: Path) -> list[Path]:
    """Return exactly the Stage-8 encoder bundle path set."""

    root = project_root.resolve()
    return [
        *sorted((root / "src" / "encoders").glob("*.py")),
        root / "src" / "coverage.py",
        root / "src" / "eb_drive.py",
        root / "src" / "eb_representation.py",
        root / "src" / "topology.py",
        root / "src" / "spv_forward.py",
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage9Error(f"cannot read required JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Stage9Error(f"JSON root must be an object: {path}")
    return value


def validate_stage9_preflight(
    project_root: str | Path,
    *,
    require_parameter_flag_absent: bool,
) -> Stage9Preflight:
    """Validate all frozen identities before any development computation."""

    root = Path(project_root).resolve()
    manifest_path = root / "data" / "benchmark" / "benchmark_manifest.json"
    manifest_sha_path = root / "data" / "benchmark" / "benchmark_manifest.sha256"
    actual_manifest_sha = sha256_file(manifest_path)
    sidecar_manifest_sha = manifest_sha_path.read_text(encoding="ascii").split()[0]
    if actual_manifest_sha != sidecar_manifest_sha or actual_manifest_sha != STAGE8_MANIFEST_SHA256:
        raise Stage9Error("Stage-8 benchmark manifest hash mismatch")
    manifest = _read_json(manifest_path)
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise Stage9Error("Stage-8 benchmark ID mismatch")
    if any(manifest["stage8_prohibited_actions"].values()):
        raise Stage9Error("Stage-8 manifest records a prohibited performance action")

    truth = validate_stage5_5_truth_table()
    if truth.sha256 != STAGE5_5_TRUTH_TABLE_SHA256:
        raise Stage9Error("Stage-5.5 truth-table hash mismatch")
    actual_encoder_sha = tree_sha256(root, encoder_bundle_paths(root))
    expected_encoder_sha = manifest["protocol"]["encoder_implementation_bundle_sha256"]
    if actual_encoder_sha != expected_encoder_sha or actual_encoder_sha != STAGE8_ENCODER_BUNDLE_SHA256:
        raise Stage9Error("encoder implementation bundle changed after Stage 8")

    config = load_config(root / "configs" / "formal_benchmark.yaml")
    validate_formal_config(config)
    if tuple(config.get("dataset.split.train_fonts")) != DEVELOPMENT_TRAIN_FONTS:
        raise Stage9Error("development-train font identities changed")
    if tuple(config.get("dataset.split.validation_fonts")) != DEVELOPMENT_VALIDATION_FONTS:
        raise Stage9Error("development-validation font identity changed")
    if tuple(config.get("dataset.split.test_fonts")) != LOCKED_TEST_FONTS:
        raise Stage9Error("locked-test font identities changed")

    font_split_path = root / "data" / "registry" / "font_split_formal.json"
    glyph_path = root / "data" / "processed" / "benchmark" / "glyphs.npz"
    ideal_mapping_path = root / "data" / "mappings" / "formal" / "ideal_mapping.npz"
    development_mapping_path = (
        root / "data" / "mappings" / "formal" / "moderate_seed_2101.npz"
    )
    checks = {
        "font split": (sha256_file(font_split_path), STAGE8_FONT_SPLIT_SHA256),
        "glyph archive": (sha256_file(glyph_path), STAGE8_GLYPH_ARCHIVE_SHA256),
        "ideal mapping": (sha256_file(ideal_mapping_path), IDEAL_MAPPING_SHA256),
        "development mapping": (
            sha256_file(development_mapping_path),
            DEVELOPMENT_MAPPING_SHA256,
        ),
    }
    changed = [name for name, (actual, expected) in checks.items() if actual != expected]
    if changed:
        raise Stage9Error(f"frozen Stage-8 asset hashes changed: {changed}")

    ledger = _read_json(
        root / "data" / "mappings" / "formal" / "mapping_access_ledger.json"
    )
    access_count = int(ledger.get("formal_mapping_performance_access_count", -1))
    if access_count != 0:
        raise Stage9Error("formal mapping seed performance access count is not zero")
    for item in ledger.get("formal_mappings", []):
        if int(item.get("seed", -1)) not in FORMAL_MAPPING_SEEDS:
            raise Stage9Error("formal mapping ledger contains an unknown seed")
        if item.get("accuracy_runs") or item.get("parameter_search_runs") or item.get("method_comparison_runs"):
            raise Stage9Error("formal mapping ledger contains a performance access")

    parameter_flag = root / "configs" / "formal_parameters_frozen.json"
    if require_parameter_flag_absent and parameter_flag.exists():
        raise Stage9Error("formal parameters already exist; one-time Stage-9 runner refuses overwrite")

    return Stage9Preflight(
        benchmark_id=BENCHMARK_ID,
        benchmark_manifest_sha256=actual_manifest_sha,
        truth_table_sha256=truth.sha256,
        encoder_bundle_sha256=actual_encoder_sha,
        character_sequence_sha256=CHARACTER_SEQUENCE_SHA256,
        font_split_sha256=sha256_file(font_split_path),
        glyph_archive_sha256=sha256_file(glyph_path),
        ideal_mapping_sha256=sha256_file(ideal_mapping_path),
        development_mapping_sha256=sha256_file(development_mapping_path),
        formal_mapping_performance_access_count=access_count,
    )


def load_development_dataset(project_root: str | Path) -> DevelopmentDataset:
    """Build a strict development-only array after validating all split metadata."""

    root = Path(project_root).resolve()
    path = root / "data" / "processed" / "benchmark" / "glyphs.npz"
    if sha256_file(path) != STAGE8_GLYPH_ARCHIVE_SHA256:
        raise Stage9Error("glyph archive changed before development loading")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "images",
            "class_ids",
            "sample_ids",
            "characters",
            "font_ids",
            "splits",
        }
        if not required <= set(archive.files):
            raise Stage9Error("formal glyph archive is missing required arrays")
        font_ids_all = np.asarray(archive["font_ids"])
        splits_all = np.asarray(archive["splits"])
        class_ids_all = np.asarray(archive["class_ids"])
        sample_ids_all = np.asarray(archive["sample_ids"])
        characters_all = np.asarray(archive["characters"])
        if font_ids_all.shape != (1440,) or splits_all.shape != (1440,):
            raise Stage9Error("formal glyph metadata must contain 1440 samples")

        train_mask = np.isin(font_ids_all, DEVELOPMENT_TRAIN_FONTS)
        validation_mask = np.isin(font_ids_all, DEVELOPMENT_VALIDATION_FONTS)
        locked_mask = np.isin(font_ids_all, LOCKED_TEST_FONTS)
        if int(train_mask.sum()) != 900 or int(validation_mask.sum()) != 180 or int(locked_mask.sum()) != 360:
            raise Stage9Error("font whitelist did not produce the frozen 900/180/360 counts")
        if np.any(train_mask & validation_mask) or np.any((train_mask | validation_mask) & locked_mask):
            raise Stage9Error("font-role masks overlap")
        if not np.all(splits_all[train_mask] == "train"):
            raise Stage9Error("a development-train font has the wrong split label")
        if not np.all(splits_all[validation_mask] == "validation"):
            raise Stage9Error("the development-validation font has the wrong split label")
        if not np.all(splits_all[locked_mask] == "test"):
            raise Stage9Error("a locked-test font has the wrong split label")

        # The locked-test mask is validated above, then intentionally discarded.
        # Only these 1080 whitelisted indices are ever used to index the image array.
        development_indices = np.flatnonzero(train_mask | validation_mask).astype(np.int64)
        images = np.ascontiguousarray(archive["images"][development_indices], dtype=np.float32)
        class_ids = np.ascontiguousarray(class_ids_all[development_indices], dtype=np.int64)
        sample_ids = np.asarray(sample_ids_all[development_indices]).copy()
        characters = np.asarray(characters_all[development_indices]).copy()
        font_ids = np.asarray(font_ids_all[development_indices]).copy()
        splits = np.asarray(splits_all[development_indices]).copy()

    if not np.isfinite(images).all() or float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise Stage9Error("development glyph pixels are invalid")
    if np.any(images.max(axis=(1, 2)) <= 0.0):
        raise Stage9Error("development glyph array contains an empty image")
    train_indices = np.flatnonzero(splits == "train").astype(np.int64)
    validation_indices = np.flatnonzero(splits == "validation").astype(np.int64)
    return DevelopmentDataset(
        images=images,
        class_ids=class_ids,
        sample_ids=sample_ids,
        characters=characters,
        font_ids=font_ids,
        splits=splits,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def load_frozen_mapping(path: str | Path, *, expected_seed: int) -> ElectrodeMapping:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        seed = int(np.asarray(archive["seed"])[0])
        mapping_id = str(np.asarray(archive["mapping_id"])[0])
        benchmark_id = str(np.asarray(archive["benchmark_id"])[0])
        image_shape_values = np.asarray(archive["image_shape"], dtype=np.int64)
        centers = np.ascontiguousarray(archive["centers_xy"], dtype=np.float32)
        sigma = np.ascontiguousarray(archive["sigma_px"], dtype=np.float32)
    if seed != expected_seed or benchmark_id != BENCHMARK_ID:
        raise Stage9Error("mapping identity does not match the frozen benchmark")
    expected_id = "ideal" if expected_seed == 0 else f"moderate_seed{expected_seed}"
    if mapping_id != expected_id:
        raise Stage9Error("mapping ID does not match its seed")
    image_shape = tuple(map(int, image_shape_values.tolist()))
    if image_shape != (96, 96):
        raise Stage9Error("mapping image shape changed")
    return ElectrodeMapping(
        mapping_id=mapping_id,
        seed=seed,
        centers_xy=centers,
        sigma_px=sigma,
        image_shape=(96, 96),
    )


def validate_alpha_pair(alpha_e: float, alpha_j: float) -> tuple[float, float]:
    pair = (float(alpha_e), float(alpha_j))
    if pair not in PREREGISTERED_ALPHA_PAIRS:
        raise Stage9Error(f"alpha pair is outside the preregistered grid: {pair}")
    return pair


def build_weighted_eb_representation_from_components(
    components: EbDriveComponents,
    projector: IdealGaussianUtilityProjector,
    *,
    alpha_e: float,
    alpha_j: float,
) -> WeightedEbRepresentation:
    """Apply D_EB=N(P_S+alpha_E P_E+alpha_J P_J), then frozen u_i."""

    alpha_e, alpha_j = validate_alpha_pair(alpha_e, alpha_j)
    combined = (
        components.p_s.astype(np.float64)
        + alpha_e * components.p_e.astype(np.float64)
        + alpha_j * components.p_j.astype(np.float64)
    )
    drive_map = max_normalize_nonzero(combined)
    scores = projector.project(drive_map)
    return WeightedEbRepresentation(
        drive_map=drive_map,
        candidate_scores=np.ascontiguousarray(scores, dtype=np.float32),
    )


def build_weighted_eb_representation(
    image: FloatImage,
    projector: IdealGaussianUtilityProjector,
    *,
    alpha_e: float,
    alpha_j: float,
) -> WeightedEbRepresentation:
    """Shared formal representation path for both EB-TopK and EB-SSE."""

    components = build_eb_drive_components(image)
    return build_weighted_eb_representation_from_components(
        components,
        projector,
        alpha_e=alpha_e,
        alpha_j=alpha_j,
    )


def build_hog(config: ExperimentConfig) -> HogFeatureExtractor:
    return HogFeatureExtractor(
        image_shape=config.image_size,
        orientations=int(config.get("feature.orientations")),
        pixels_per_cell=tuple(config.get("feature.pixels_per_cell")),  # type: ignore[arg-type]
        cells_per_block=tuple(config.get("feature.cells_per_block")),  # type: ignore[arg-type]
        block_norm=str(config.get("feature.block_norm")),
    )


def build_classifier(config: ExperimentConfig) -> LinearSvcClassifier:
    return LinearSvcClassifier(
        c=float(config.get("classifier.c")),
        dual=str(config.get("classifier.dual")),
        max_iter=int(config.get("classifier.max_iter")),
        random_state=int(config.get("classifier.random_state")),
    )


def _score_complexity(alpha_e: float, alpha_j: float) -> float:
    return abs(log2(alpha_e)) + abs(log2(alpha_j))


def select_preregistered_pair(
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply exact score, centrality, alpha_E, alpha_J tie rules in order."""

    if len(results) != 9:
        raise Stage9Error("selection requires all nine preregistered results")
    by_pair: dict[tuple[float, float], Mapping[str, Any]] = {}
    for result in results:
        pair = validate_alpha_pair(float(result["alpha_e"]), float(result["alpha_j"]))
        if pair in by_pair:
            raise Stage9Error(f"duplicate result for alpha pair {pair}")
        score = float(result["score"])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise Stage9Error("validation score must lie in [0,1]")
        by_pair[pair] = result
    if set(by_pair) != set(PREREGISTERED_ALPHA_PAIRS):
        raise Stage9Error("result set does not cover the preregistered 3x3 grid")

    best_score = max(float(item["score"]) for item in results)
    exact_score_ties = [item for item in results if float(item["score"]) == best_score]
    minimum_complexity = min(
        _score_complexity(float(item["alpha_e"]), float(item["alpha_j"]))
        for item in exact_score_ties
    )
    central_ties = [
        item
        for item in exact_score_ties
        if _score_complexity(float(item["alpha_e"]), float(item["alpha_j"]))
        == minimum_complexity
    ]
    ordered = sorted(
        central_ties,
        key=lambda item: (float(item["alpha_e"]), float(item["alpha_j"])),
    )
    selected = dict(ordered[0])
    tie_report = {
        "best_score": best_score,
        "exact_score_tie_count": len(exact_score_ties),
        "exact_score_tied_pairs": [
            [float(item["alpha_e"]), float(item["alpha_j"])]
            for item in sorted(
                exact_score_ties,
                key=lambda item: (float(item["alpha_e"]), float(item["alpha_j"])),
            )
        ],
        "minimum_abs_log2_complexity": minimum_complexity,
        "pairs_after_tie_break_1": [
            [float(item["alpha_e"]), float(item["alpha_j"])]
            for item in sorted(
                central_ties,
                key=lambda item: (float(item["alpha_e"]), float(item["alpha_j"])),
            )
        ],
        "tie_break_2_order": "alpha_e_ascending_then_alpha_j_ascending",
        "selected_pair": [float(selected["alpha_e"]), float(selected["alpha_j"])],
    }
    return selected, tie_report


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    temporary.replace(path)


def result_array_hashes(
    *,
    selections: NDArray[np.int64],
    spv: NDArray[np.float32],
    features: NDArray[np.float32],
    predictions: NDArray[np.int64],
) -> dict[str, str]:
    return {
        "selections_sha256": array_sha256(selections),
        "spv_sha256": array_sha256(spv),
        "hog_features_sha256": array_sha256(features),
        "validation_predictions_sha256": array_sha256(predictions),
    }


def make_grid_and_models(
    project_root: Path,
    config: ExperimentConfig,
) -> tuple[
    RegularGrid,
    IdealGaussianUtilityProjector,
    GaussianPhospheneForwardModel,
    HogFeatureExtractor,
]:
    grid = make_regular_grid(rows=16, cols=16, image_shape=(96, 96))
    ideal_path = project_root / "data" / "mappings" / "formal" / "ideal_mapping.npz"
    moderate_path = project_root / "data" / "mappings" / "formal" / "moderate_seed_2101.npz"
    if sha256_file(ideal_path) != IDEAL_MAPPING_SHA256:
        raise Stage9Error("ideal mapping hash changed")
    if sha256_file(moderate_path) != DEVELOPMENT_MAPPING_SHA256:
        raise Stage9Error("development mapping hash changed")
    ideal_mapping = load_frozen_mapping(ideal_path, expected_seed=0)
    moderate_mapping = load_frozen_mapping(
        moderate_path,
        expected_seed=DEVELOPMENT_MAPPING_SEED,
    )
    if not np.array_equal(ideal_mapping.centers_xy, grid.centers_xy):
        raise Stage9Error("ideal mapping centers differ from the frozen candidate grid")
    projector = IdealGaussianUtilityProjector(ideal_mapping)
    forward = GaussianPhospheneForwardModel(
        moderate_mapping,
        amplitude_scale=float(config.get("spv.amplitude")),
        kernel_normalization=str(config.get("spv.kernel_normalization")),
    )
    return grid, projector, forward, build_hog(config)


def make_unit_amplitudes(sample_count: int, budget: int) -> NDArray[np.float32]:
    amplitudes = np.ones((sample_count, budget), dtype=np.float32)
    if np.count_nonzero(amplitudes) != sample_count * budget:
        raise Stage9Error("unit stimulation did not activate every selected electrode")
    return amplitudes


def evaluate_one_condition(
    *,
    dataset: DevelopmentDataset,
    selections: NDArray[np.int64],
    budget: int,
    forward: GaussianPhospheneForwardModel,
    extractor: HogFeatureExtractor,
    config: ExperimentConfig,
) -> tuple[dict[str, Any], NDArray[np.int64], dict[str, str]]:
    if selections.shape != (1080, budget) or selections.dtype != np.int64:
        raise Stage9Error("selection matrix has the wrong shape or dtype")
    if any(np.unique(row).size != budget for row in selections):
        raise Stage9Error("a sample does not contain exactly B unique electrodes")
    amplitudes = make_unit_amplitudes(1080, budget)
    spv = forward.generate_variable_batch(selections, amplitudes)
    expected_energy = expected_output_energy(amplitudes, 1.0)
    actual_energy = spv.sum(axis=(1, 2), dtype=np.float64)
    absolute_error = np.abs(actual_energy - expected_energy)
    relative_error = absolute_error / np.maximum(expected_energy, np.finfo(np.float64).eps)
    tolerance = float(config.get("validation.energy_relative_tolerance"))
    if float(relative_error.max()) > tolerance:
        raise Stage9Error("development SPV energy exceeded the frozen tolerance")
    features = extractor.extract_batch(spv)
    classifier = build_classifier(config)
    classifier.fit(
        features[dataset.train_indices],
        dataset.class_ids[dataset.train_indices],
    )
    predictions = classifier.predict(features[dataset.validation_indices])
    accuracy = classification_accuracy(
        dataset.class_ids[dataset.validation_indices],
        predictions,
    )
    correct = int(np.sum(dataset.class_ids[dataset.validation_indices] == predictions))
    metrics = {
        "budget": budget,
        "train_sample_count": int(dataset.train_indices.size),
        "validation_sample_count": int(dataset.validation_indices.size),
        "locked_test_sample_count": 0,
        "correct_count": correct,
        "accuracy": accuracy,
        "active_electrodes_per_sample": budget,
        "unit_amplitude": True,
        "theoretical_energy": float(budget),
        "actual_energy_min": float(actual_energy.min()),
        "actual_energy_max": float(actual_energy.max()),
        "max_energy_absolute_error": float(absolute_error.max()),
        "max_energy_relative_error": float(relative_error.max()),
        "hog_feature_length": int(features.shape[1]),
        "linear_svc_n_iter": classifier.n_iter,
    }
    hashes = result_array_hashes(
        selections=selections,
        spv=spv,
        features=features,
        predictions=predictions,
    )
    return metrics, predictions, hashes


def prediction_rows(
    dataset: DevelopmentDataset,
    predictions: NDArray[np.int64],
) -> list[dict[str, Any]]:
    character_by_class = {
        int(class_id): str(character)
        for class_id, character in zip(dataset.class_ids, dataset.characters, strict=True)
    }
    rows: list[dict[str, Any]] = []
    for local_index, prediction in zip(
        dataset.validation_indices,
        predictions,
        strict=True,
    ):
        true_class = int(dataset.class_ids[local_index])
        predicted_class = int(prediction)
        rows.append(
            {
                "sample_id": str(dataset.sample_ids[local_index]),
                "font_id": str(dataset.font_ids[local_index]),
                "split": "validation",
                "true_class_id": true_class,
                "true_character": str(dataset.characters[local_index]),
                "predicted_class_id": predicted_class,
                "predicted_character": character_by_class[predicted_class],
                "correct": true_class == predicted_class,
            }
        )
    return rows


def format_alpha_token(value: float) -> str:
    return str(value).replace(".", "p")


def synthetic_tie_rule_self_check() -> dict[str, Any]:
    """Exercise both preregistered tie stages with artificial result grids."""

    def grid(default_score: float = 0.1) -> list[dict[str, float]]:
        return [
            {"alpha_e": alpha_e, "alpha_j": alpha_j, "score": default_score}
            for alpha_e, alpha_j in PREREGISTERED_ALPHA_PAIRS
        ]

    all_equal = grid(0.5)
    selected_all, report_all = select_preregistered_pair(all_equal)
    if (selected_all["alpha_e"], selected_all["alpha_j"]) != (1.0, 1.0):
        raise Stage9Error("tie-break 1 failed to prefer the central pair")

    alpha_e_tie = grid(0.1)
    for item in alpha_e_tie:
        if (item["alpha_e"], item["alpha_j"]) in {(0.5, 1.0), (2.0, 1.0)}:
            item["score"] = 0.9
    selected_e, report_e = select_preregistered_pair(alpha_e_tie)
    if (selected_e["alpha_e"], selected_e["alpha_j"]) != (0.5, 1.0):
        raise Stage9Error("tie-break 2 failed alpha_E ascending order")

    alpha_j_tie = grid(0.1)
    for item in alpha_j_tie:
        if (item["alpha_e"], item["alpha_j"]) in {(1.0, 0.5), (1.0, 2.0)}:
            item["score"] = 0.9
    selected_j, report_j = select_preregistered_pair(alpha_j_tie)
    if (selected_j["alpha_e"], selected_j["alpha_j"]) != (1.0, 0.5):
        raise Stage9Error("tie-break 2 failed alpha_J ascending order")

    primary = grid(0.1)
    for item in primary:
        if (item["alpha_e"], item["alpha_j"]) == (2.0, 2.0):
            item["score"] = 0.2
    selected_primary, _ = select_preregistered_pair(primary)
    if (selected_primary["alpha_e"], selected_primary["alpha_j"]) != (2.0, 2.0):
        raise Stage9Error("primary score was not applied before tie rules")
    return {
        "status": "PASS",
        "all_equal_selected": [selected_all["alpha_e"], selected_all["alpha_j"]],
        "alpha_e_second_tie_selected": [selected_e["alpha_e"], selected_e["alpha_j"]],
        "alpha_j_second_tie_selected": [selected_j["alpha_e"], selected_j["alpha_j"]],
        "primary_score_selected": [selected_primary["alpha_e"], selected_primary["alpha_j"]],
        "reports": [report_all, report_e, report_j],
    }
