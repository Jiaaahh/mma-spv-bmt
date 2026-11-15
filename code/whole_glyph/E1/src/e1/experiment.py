"""Development and confirmation computation for the independent E1 study."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score
import yaml

from src.contracts import ElectrodeMapping
from src.e1.classifiers import BalancedLinearSVC
from src.e1.contour_encoder import ScialomStyleGridContourEncoder
from src.encoders.skeleton_topk import SkeletonTopKEncoder
from src.features import HogFeatureExtractor
from src.grid import make_regular_grid
from src.mapping import make_ideal_mapping
from src.spv_forward import GaussianPhospheneForwardModel, IdealGaussianUtilityProjector
from src.v2.multimapping_prototypes import MultiMappingCosinePrototypeClassifier


IMAGE_SHAPE = (96, 96)
BUDGET = 64
TRAINING_FONT_COUNT = 6
CLASS_COUNT = 180


class E1ExperimentError(RuntimeError):
    """Raised when an E1 protocol or computation violates a frozen invariant."""


@dataclass(frozen=True, slots=True)
class GlyphDataset:
    images: NDArray[np.float32]
    class_ids: NDArray[np.int64]
    sample_ids: NDArray[np.str_]
    characters: NDArray[np.str_]
    font_ids: NDArray[np.str_]


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise E1ExperimentError("configuration root must be a mapping")
    value["_config_path"] = str(source)
    value["_project_root"] = str(source.parent.parent)
    return value


def _resolve(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(str(config["_project_root"])) / path
    return path.resolve()


def load_glyph_dataset(
    path: str | Path,
    *,
    allowed_fonts: Sequence[str] | None = None,
) -> GlyphDataset:
    source = Path(path)
    if not source.is_file():
        raise E1ExperimentError(f"glyph archive does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {"images", "class_ids", "sample_ids", "characters", "font_ids"}
        if not required <= set(archive.files):
            raise E1ExperimentError("glyph archive is missing required arrays")
        images_all = archive["images"]
        class_ids_all = archive["class_ids"]
        sample_ids_all = np.asarray(archive["sample_ids"], dtype=np.str_)
        characters_all = np.asarray(archive["characters"], dtype=np.str_)
        font_ids_all = np.asarray(archive["font_ids"], dtype=np.str_)
        count = int(class_ids_all.size)
        if (
            images_all.dtype != np.float32
            or images_all.shape != (count, *IMAGE_SHAPE)
            or class_ids_all.dtype != np.int64
            or class_ids_all.shape != (count,)
            or any(
                array.shape != (count,)
                for array in (sample_ids_all, characters_all, font_ids_all)
            )
        ):
            raise E1ExperimentError("glyph archive arrays violate the E1 contract")
        if allowed_fonts is None:
            indices = np.arange(count, dtype=np.int64)
        else:
            expected_fonts = tuple(map(str, allowed_fonts))
            observed_fonts = set(map(str, np.unique(font_ids_all)))
            missing = set(expected_fonts) - observed_fonts
            if missing:
                raise E1ExperimentError(
                    "glyph archive is missing fonts: " + ", ".join(sorted(missing))
                )
            indices = np.flatnonzero(np.isin(font_ids_all, expected_fonts)).astype(
                np.int64
            )
        images = np.ascontiguousarray(images_all[indices], dtype=np.float32)
        class_ids = np.ascontiguousarray(class_ids_all[indices], dtype=np.int64)
        sample_ids = np.asarray(sample_ids_all[indices], dtype=np.str_)
        characters = np.asarray(characters_all[indices], dtype=np.str_)
        font_ids = np.asarray(font_ids_all[indices], dtype=np.str_)

    if (
        images.shape[0] == 0
        or not np.isfinite(images).all()
        or float(images.min()) < 0.0
        or float(images.max()) > 1.0
        or np.any(images.max(axis=(1, 2)) <= 0.0)
    ):
        raise E1ExperimentError("glyph images are empty or invalid")
    classes = np.unique(class_ids)
    fonts = tuple(map(str, np.unique(font_ids)))
    if classes.size != CLASS_COUNT or not np.array_equal(classes, np.arange(CLASS_COUNT)):
        raise E1ExperimentError("E1 requires class IDs 0..179")
    expected_pairs = {
        (font_id, class_id) for font_id in fonts for class_id in range(CLASS_COUNT)
    }
    observed_pairs = [
        (str(font_id), int(class_id))
        for font_id, class_id in zip(font_ids, class_ids, strict=True)
    ]
    if len(observed_pairs) != len(expected_pairs) or set(observed_pairs) != expected_pairs:
        raise E1ExperimentError("every font requires one glyph for every class")
    if len(set(map(str, sample_ids))) != images.shape[0]:
        raise E1ExperimentError("sample IDs must be unique")
    return GlyphDataset(images, class_ids, sample_ids, characters, font_ids)


def load_mapping(path: str | Path) -> ElectrodeMapping:
    source = Path(path)
    if not source.is_file():
        raise E1ExperimentError(f"mapping file does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {"centers_xy", "sigma_px", "image_shape", "mapping_id", "seed"}
        if not required <= set(archive.files):
            raise E1ExperimentError(f"mapping is missing arrays: {source}")
        mapping_id = str(np.asarray(archive["mapping_id"]).reshape(-1)[0])
        seed = int(np.asarray(archive["seed"]).reshape(-1)[0])
        image_shape = tuple(map(int, np.asarray(archive["image_shape"]).reshape(-1)))
        centers = np.ascontiguousarray(archive["centers_xy"], dtype=np.float32)
        sigma = np.ascontiguousarray(archive["sigma_px"], dtype=np.float32)
    return ElectrodeMapping(
        mapping_id=mapping_id,
        seed=seed,
        centers_xy=centers,
        sigma_px=sigma,
        image_shape=image_shape,
    )


def build_hog(config: Mapping[str, Any]) -> HogFeatureExtractor:
    feature = config.get("feature", {})
    return HogFeatureExtractor(
        image_shape=IMAGE_SHAPE,
        orientations=int(feature.get("orientations", 9)),
        pixels_per_cell=tuple(map(int, feature.get("pixels_per_cell", (8, 8)))),
        cells_per_block=tuple(map(int, feature.get("cells_per_block", (2, 2)))),
        block_norm=str(feature.get("block_norm", "L2-Hys")),
    )


def compute_selections(
    images: NDArray[np.float32],
    *,
    selector: str,
    workers: int,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    regular = make_regular_grid(rows=16, cols=16, image_shape=IMAGE_SHAPE)
    ideal = make_ideal_mapping(regular)
    projector = IdealGaussianUtilityProjector(ideal)
    if selector == "skeleton":
        encoder: Any = SkeletonTopKEncoder()
    elif selector == "scialom_contour":
        encoder = ScialomStyleGridContourEncoder()
    else:
        raise E1ExperimentError(f"unknown selector: {selector}")

    def encode_one(image: NDArray[np.float32]) -> tuple[NDArray[np.int64], int]:
        result = encoder.encode(
            np.ascontiguousarray(image, dtype=np.float32),
            regular.electrodes,
            BUDGET,
            projector=projector,
        )
        backfill = int(result.metadata.get("threshold_backfill_count", 0))
        return result.selected_indices, backfill

    start = time.perf_counter()
    if workers <= 1:
        rows = [encode_one(image) for image in images]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(encode_one, images))
    selections = np.ascontiguousarray(
        np.stack([row[0] for row in rows]), dtype=np.int64
    )
    if (
        selections.shape != (images.shape[0], BUDGET)
        or any(np.unique(row).size != BUDGET for row in selections)
        or np.any(selections < 0)
        or np.any(selections >= 256)
    ):
        raise E1ExperimentError(f"{selector} produced invalid selections")
    return selections, {
        "selector": selector,
        "sample_count": int(images.shape[0]),
        "selection_seconds": time.perf_counter() - start,
        "backfill_total": int(sum(row[1] for row in rows)),
        "backfill_max": int(max((row[1] for row in rows), default=0)),
    }


def build_features(
    selections: NDArray[np.int64],
    mapping: ElectrodeMapping,
    extractor: HogFeatureExtractor,
    *,
    chunk_size: int = 12,
) -> tuple[NDArray[np.float32], dict[str, float]]:
    """Render unit-amplitude SPV images in bounded-memory chunks and run HOG."""

    if selections.dtype != np.int64 or selections.ndim != 2 or selections.shape[1] != BUDGET:
        raise E1ExperimentError("selections must be int64 (N,64)")
    forward = GaussianPhospheneForwardModel(mapping)
    kernels = forward.kernel_bank.reshape(mapping.candidate_count, -1)
    features = np.empty((selections.shape[0], extractor.feature_length), dtype=np.float32)
    observed_min = float("inf")
    observed_max = float("-inf")
    start = time.perf_counter()
    for begin in range(0, selections.shape[0], chunk_size):
        end = min(begin + chunk_size, selections.shape[0])
        selected_kernels = kernels[selections[begin:end]]
        flat = selected_kernels.sum(axis=1, dtype=np.float32)
        images = np.ascontiguousarray(
            flat.reshape(end - begin, *IMAGE_SHAPE), dtype=np.float32
        )
        energy = images.sum(axis=(1, 2), dtype=np.float64)
        observed_min = min(observed_min, float(energy.min()))
        observed_max = max(observed_max, float(energy.max()))
        features[begin:end] = extractor.extract_batch(images)
    if not np.allclose(
        [observed_min, observed_max], [BUDGET, BUDGET], rtol=0.0, atol=2e-4
    ):
        raise E1ExperimentError(
            f"SPV energy audit failed for {mapping.mapping_id}: "
            f"{observed_min}, {observed_max}"
        )
    if np.any(np.linalg.norm(features.astype(np.float64), axis=1) == 0.0):
        raise E1ExperimentError(f"HOG returned a zero vector for {mapping.mapping_id}")
    return np.ascontiguousarray(features), {
        "feature_seconds": time.perf_counter() - start,
        "energy_min": observed_min,
        "energy_max": observed_max,
    }


def training_arrays(
    dataset: GlyphDataset,
    feature_bank: Mapping[str, NDArray[np.float32]],
    *,
    indices: NDArray[np.int64],
    mapping_ids: Sequence[str],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int64],
    NDArray[np.str_],
    NDArray[np.str_],
]:
    matrices: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.int64]] = []
    fonts: list[NDArray[np.str_]] = []
    mapping_rows: list[NDArray[np.str_]] = []
    for mapping_id in mapping_ids:
        matrix = feature_bank.get(mapping_id)
        if matrix is None:
            raise E1ExperimentError(f"feature bank is missing {mapping_id}")
        matrices.append(np.ascontiguousarray(matrix[indices], dtype=np.float32))
        labels.append(np.ascontiguousarray(dataset.class_ids[indices], dtype=np.int64))
        fonts.append(np.asarray(dataset.font_ids[indices], dtype=np.str_))
        mapping_rows.append(
            np.asarray([mapping_id] * int(indices.size), dtype=np.str_)
        )
    return (
        np.ascontiguousarray(np.concatenate(matrices), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(labels), dtype=np.int64),
        np.ascontiguousarray(np.concatenate(fonts), dtype=np.str_),
        np.ascontiguousarray(np.concatenate(mapping_rows), dtype=np.str_),
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".write_tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise E1ExperimentError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".write_tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _prepare_output(path: Path, *, force: bool) -> Path:
    if path.exists():
        if not force:
            raise E1ExperimentError(f"output already exists: {path}")
        shutil.rmtree(path)
    temporary = path.with_name(path.name + ".running_tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    return temporary


def run_development(config_path: str | Path, *, force: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    development = config["development"]
    training = config["training"]
    output = _resolve(config, development["output_dir"])
    temporary = _prepare_output(output, force=force)
    dataset = load_glyph_dataset(
        _resolve(config, training["glyphs_npz"]),
        allowed_fonts=tuple(training["fonts"]),
    )
    if len(np.unique(dataset.font_ids)) != TRAINING_FONT_COUNT:
        raise E1ExperimentError("development requires exactly six fonts")
    workers = int(config.get("runtime", {}).get("selection_workers", 1))
    selections, selection_audit = compute_selections(
        dataset.images, selector="skeleton", workers=workers
    )
    extractor = build_hog(config)
    mapping_specs = {
        "ideal": training["ideal_mapping"],
        **{
            f"moderate_seed{seed}": path
            for seed, path in training["moderate_mappings"].items()
        },
    }
    feature_bank: dict[str, NDArray[np.float32]] = {}
    feature_audit: dict[str, Any] = {}
    for expected_id, path in mapping_specs.items():
        mapping = load_mapping(_resolve(config, path))
        if mapping.mapping_id != expected_id:
            raise E1ExperimentError(
                f"mapping ID mismatch: expected {expected_id}, got {mapping.mapping_id}"
            )
        feature_bank[expected_id], feature_audit[expected_id] = build_features(
            selections, mapping, extractor
        )

    train_mapping_ids = (
        "ideal",
        *(f"moderate_seed{seed}" for seed in development["training_mapping_seeds"]),
    )
    validation_mapping_id = f"moderate_seed{int(development['validation_mapping_seed'])}"
    c_grid = tuple(map(float, development["c_grid"]))
    class_order = tuple(range(CLASS_COUNT))
    metric_rows: list[dict[str, Any]] = []
    fonts = tuple(map(str, training["fonts"]))
    for heldout_font in fonts:
        train_indices = np.flatnonzero(dataset.font_ids != heldout_font).astype(np.int64)
        validation_indices = np.flatnonzero(dataset.font_ids == heldout_font).astype(np.int64)
        arrays = training_arrays(
            dataset,
            feature_bank,
            indices=train_indices,
            mapping_ids=train_mapping_ids,
        )
        validation_features = feature_bank[validation_mapping_id][validation_indices]
        truth = dataset.class_ids[validation_indices]
        for c_value in c_grid:
            fit_start = time.perf_counter()
            classifier = BalancedLinearSVC(class_order, c=c_value).fit(*arrays)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            predictions = classifier.predict(validation_features)
            predict_seconds = time.perf_counter() - predict_start
            metric_rows.append(
                {
                    "heldout_font": heldout_font,
                    "validation_mapping_id": validation_mapping_id,
                    "c": c_value,
                    "accuracy": float(np.mean(predictions == truth)),
                    "macro_f1": float(
                        f1_score(truth, predictions, average="macro", zero_division=0)
                    ),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "n_iter": classifier.n_iter_,
                    "model_size_bytes": classifier.model_size_bytes,
                }
            )
    aggregate: list[dict[str, Any]] = []
    for c_value in c_grid:
        rows = [row for row in metric_rows if float(row["c"]) == c_value]
        aggregate.append(
            {
                "c": c_value,
                "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
                "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
            }
        )
    selected = sorted(aggregate, key=lambda row: (-row["mean_macro_f1"], row["c"]))[0]
    summary = {
        "status": "D3_COMPLETE_DEVELOPMENT_ONLY",
        "selected_c": selected["c"],
        "selection_rule": "max_mean_macro_f1_tie_smallest_c",
        "fold_count": len(fonts),
        "c_grid": list(c_grid),
        "training_mapping_ids": list(train_mapping_ids),
        "validation_mapping_id": validation_mapping_id,
        "aggregate": aggregate,
        "selection_audit": selection_audit,
        "feature_audit": feature_audit,
        "e1_test_assets_read": False,
        "e1_predictions_executed": False,
    }
    _write_csv(temporary / "fold_metrics.csv", metric_rows)
    _atomic_json(temporary / "summary.json", summary)
    _atomic_json(
        temporary / "selected_c.json",
        {
            "selected_c": selected["c"],
            "selection_rule": summary["selection_rule"],
            "e1_test_assets_read": False,
        },
    )
    temporary.replace(output)
    return summary


def _fit_confirmation_models(
    dataset: GlyphDataset,
    feature_banks: Mapping[str, Mapping[str, NDArray[np.float32]]],
    *,
    selected_c: float,
) -> tuple[dict[str, tuple[str, Any]], list[dict[str, Any]]]:
    all_indices = np.arange(dataset.images.shape[0], dtype=np.int64)
    class_order = tuple(range(CLASS_COUNT))
    definitions: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
        ("proto_skeleton_ideal_only", "skeleton", "prototype", ("ideal",)),
        (
            "proto_skeleton_ideal_plus_moderate1",
            "skeleton",
            "prototype",
            ("ideal", "moderate_seed2026"),
        ),
        (
            "proto_skeleton_ideal_plus_moderate2",
            "skeleton",
            "prototype",
            ("ideal", "moderate_seed2026", "moderate_seed2027"),
        ),
        (
            "proto_skeleton_ideal_plus_multi4",
            "skeleton",
            "prototype",
            (
                "ideal",
                "moderate_seed2026",
                "moderate_seed2027",
                "moderate_seed2028",
                "moderate_seed2029",
            ),
        ),
        (
            "proto_skeleton_multi4_no_ideal",
            "skeleton",
            "prototype",
            (
                "moderate_seed2026",
                "moderate_seed2027",
                "moderate_seed2028",
                "moderate_seed2029",
            ),
        ),
        ("svc_skeleton_ideal_only", "skeleton", "svc", ("ideal",)),
        (
            "svc_skeleton_ideal_plus_multi4",
            "skeleton",
            "svc",
            (
                "ideal",
                "moderate_seed2026",
                "moderate_seed2027",
                "moderate_seed2028",
                "moderate_seed2029",
            ),
        ),
        (
            "proto_scialom_contour_ideal_only",
            "scialom_contour",
            "prototype",
            ("ideal",),
        ),
        (
            "proto_scialom_contour_ideal_plus_multi4",
            "scialom_contour",
            "prototype",
            (
                "ideal",
                "moderate_seed2026",
                "moderate_seed2027",
                "moderate_seed2028",
                "moderate_seed2029",
            ),
        ),
    )
    fitted: dict[str, tuple[str, Any]] = {}
    efficiency: list[dict[str, Any]] = []
    for method, selector, classifier_kind, mapping_ids in definitions:
        arrays = training_arrays(
            dataset,
            feature_banks[selector],
            indices=all_indices,
            mapping_ids=mapping_ids,
        )
        start = time.perf_counter()
        if classifier_kind == "prototype":
            classifier: Any = MultiMappingCosinePrototypeClassifier(class_order)
        else:
            classifier = BalancedLinearSVC(class_order, c=selected_c)
        classifier.fit(*arrays)
        fit_seconds = time.perf_counter() - start
        if classifier_kind == "prototype":
            model_size = int(classifier.prototypes_.nbytes)
            n_iter: int | None = None
        else:
            model_size = classifier.model_size_bytes
            n_iter = classifier.n_iter_
        fitted[method] = (selector, classifier)
        efficiency.append(
            {
                "method": method,
                "selector": selector,
                "classifier": classifier_kind,
                "training_mapping_ids": "|".join(mapping_ids),
                "training_sample_count": int(arrays[0].shape[0]),
                "fit_seconds": fit_seconds,
                "model_size_bytes": model_size,
                "n_iter": n_iter,
                "predict_seconds": 0.0,
                "predicted_sample_count": 0,
            }
        )
        del arrays
    return fitted, efficiency


def _prediction_rows(
    dataset: GlyphDataset,
    predictions: NDArray[np.int64],
    *,
    condition: str,
    mapping_seed: int,
    method: str,
) -> list[dict[str, Any]]:
    if predictions.shape != dataset.class_ids.shape or predictions.dtype != np.int64:
        raise E1ExperimentError("prediction vector violates E1 contract")
    rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        truth = int(dataset.class_ids[index])
        predicted = int(prediction)
        rows.append(
            {
                "method": method,
                "condition": condition,
                "mapping_seed": mapping_seed,
                "mapping_id": f"e1_{condition}_seed{mapping_seed}",
                "sample_id": str(dataset.sample_ids[index]),
                "font_id": str(dataset.font_ids[index]),
                "true_label": truth,
                "predicted_label": predicted,
                "character": str(dataset.characters[index]),
                "correct": int(truth == predicted),
            }
        )
    return rows


def run_confirmation(
    config_path: str | Path,
    *,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run all E1 conditions without aggregating or printing performance."""

    from src.e1.integrity import sha256_file, validate_lock

    config = load_yaml(config_path)
    confirmation = config["confirmation"]
    project_root = Path(str(config["_project_root"]))
    lock_source = _resolve(
        config, lock_path or confirmation.get("lock", "locks/e1_confirmation.lock.json")
    )
    lock_audit = validate_lock(lock_source, project_root=project_root)
    output = _resolve(config, confirmation["output_dir"])
    temporary = _prepare_output(output, force=False)
    completion_marker = output.with_suffix(".completion.json")
    if completion_marker.exists():
        raise E1ExperimentError(
            f"formal E1 completion marker already exists: {completion_marker}"
        )

    training = config["training"]
    train_dataset = load_glyph_dataset(
        _resolve(config, training["glyphs_npz"]),
        allowed_fonts=tuple(training["fonts"]),
    )
    asset_dir = _resolve(config, confirmation["asset_dir"])
    test_dataset = load_glyph_dataset(asset_dir / confirmation["glyphs_npz"])
    if len(np.unique(test_dataset.font_ids)) != 4:
        raise E1ExperimentError("E1 confirmation requires exactly four test fonts")
    if set(map(str, np.unique(test_dataset.font_ids))) & set(training["fonts"]):
        raise E1ExperimentError("training and E1 test font IDs overlap")

    selected_c_path = _resolve(config, confirmation["selected_c_file"])
    selected_document = json.loads(selected_c_path.read_text(encoding="utf-8"))
    selected_c = float(selected_document["selected_c"])
    if selected_document.get("e1_test_assets_read") is not False:
        raise E1ExperimentError("C selection was not isolated from E1 test assets")

    workers = int(config.get("runtime", {}).get("selection_workers", 1))
    train_selections: dict[str, NDArray[np.int64]] = {}
    test_selections: dict[str, NDArray[np.int64]] = {}
    selection_audit: dict[str, Any] = {}
    for selector in ("skeleton", "scialom_contour"):
        train_selections[selector], selection_audit[f"training_{selector}"] = (
            compute_selections(
                train_dataset.images, selector=selector, workers=workers
            )
        )
        test_selections[selector], selection_audit[f"test_{selector}"] = (
            compute_selections(
                test_dataset.images, selector=selector, workers=workers
            )
        )

    extractor = build_hog(config)
    training_mapping_paths = {
        "ideal": training["ideal_mapping"],
        **{
            f"moderate_seed{seed}": path
            for seed, path in training["moderate_mappings"].items()
            if int(seed) in (2026, 2027, 2028, 2029)
        },
    }
    expected_training_ids = {
        "ideal",
        "moderate_seed2026",
        "moderate_seed2027",
        "moderate_seed2028",
        "moderate_seed2029",
    }
    if set(training_mapping_paths) != expected_training_ids:
        raise E1ExperimentError("confirmation training mappings differ from freeze")
    feature_banks: dict[str, dict[str, NDArray[np.float32]]] = {
        "skeleton": {},
        "scialom_contour": {},
    }
    feature_audit: dict[str, Any] = {}
    for mapping_id, path in training_mapping_paths.items():
        mapping = load_mapping(_resolve(config, path))
        if mapping.mapping_id != mapping_id:
            raise E1ExperimentError(f"training mapping ID mismatch: {mapping_id}")
        for selector in feature_banks:
            key = f"training_{selector}_{mapping_id}"
            feature_banks[selector][mapping_id], feature_audit[key] = build_features(
                train_selections[selector], mapping, extractor
            )

    fitted, efficiency_rows = _fit_confirmation_models(
        train_dataset, feature_banks, selected_c=selected_c
    )
    efficiency_by_method = {row["method"]: row for row in efficiency_rows}
    conditions = tuple(map(str, confirmation["conditions"]))
    seeds = tuple(map(int, confirmation["mapping_seeds"]))
    if conditions != ("mild", "moderate", "severe", "position_only", "scale_only"):
        raise E1ExperimentError("confirmation conditions differ from frozen order")
    if seeds != (3101, 3102, 3103, 3104, 3105):
        raise E1ExperimentError("confirmation mapping seeds differ from frozen order")
    core_methods = (
        "proto_skeleton_ideal_only",
        "proto_skeleton_ideal_plus_multi4",
        "svc_skeleton_ideal_only",
        "svc_skeleton_ideal_plus_multi4",
        "proto_scialom_contour_ideal_only",
        "proto_scialom_contour_ideal_plus_multi4",
    )
    ablation_methods = (
        "proto_skeleton_ideal_plus_moderate1",
        "proto_skeleton_ideal_plus_moderate2",
        "proto_skeleton_multi4_no_ideal",
    )
    prediction_rows: list[dict[str, Any]] = []
    condition_progress: list[dict[str, Any]] = []
    import psutil

    process = psutil.Process()
    observed_peak_rss = int(process.memory_info().rss)
    for condition in conditions:
        for mapping_seed in seeds:
            mapping_path = asset_dir / "mappings" / f"e1_{condition}_seed{mapping_seed}.npz"
            mapping = load_mapping(mapping_path)
            expected_mapping_id = f"e1_{condition}_seed{mapping_seed}"
            if mapping.mapping_id != expected_mapping_id:
                raise E1ExperimentError(
                    f"test mapping ID mismatch: {expected_mapping_id}"
                )
            test_features: dict[str, NDArray[np.float32]] = {}
            for selector in ("skeleton", "scialom_contour"):
                audit_key = f"test_{selector}_{expected_mapping_id}"
                test_features[selector], feature_audit[audit_key] = build_features(
                    test_selections[selector], mapping, extractor
                )
            methods = core_methods + (ablation_methods if condition == "moderate" else ())
            for method in methods:
                selector, classifier = fitted[method]
                start = time.perf_counter()
                predictions = classifier.predict(test_features[selector])
                elapsed = time.perf_counter() - start
                efficiency_by_method[method]["predict_seconds"] += elapsed
                efficiency_by_method[method]["predicted_sample_count"] += int(
                    predictions.size
                )
                prediction_rows.extend(
                    _prediction_rows(
                        test_dataset,
                        predictions,
                        condition=condition,
                        mapping_seed=mapping_seed,
                        method=method,
                    )
                )
            observed_peak_rss = max(
                observed_peak_rss, int(process.memory_info().rss)
            )
            condition_progress.append(
                {
                    "condition": condition,
                    "mapping_seed": mapping_seed,
                    "mapping_id": expected_mapping_id,
                    "method_count": len(methods),
                    "prediction_rows_committed_in_memory": len(methods)
                    * test_dataset.images.shape[0],
                }
            )
            print(
                f"PASS_PREDICTIONS_BUFFERED {condition} seed={mapping_seed} "
                f"methods={len(methods)}"
            )

    expected_rows = (
        len(core_methods) * len(conditions) * len(seeds) * test_dataset.images.shape[0]
        + len(ablation_methods) * len(seeds) * test_dataset.images.shape[0]
    )
    if len(prediction_rows) != expected_rows:
        raise E1ExperimentError(
            f"prediction row count {len(prediction_rows)} != {expected_rows}"
        )
    _write_csv(temporary / "predictions.csv", prediction_rows)
    _write_csv(temporary / "efficiency.csv", efficiency_rows)
    _atomic_json(temporary / "condition_progress.json", condition_progress)
    manifest = {
        "status": "E1_PREDICTIONS_COMPLETE_AWAITING_UNBLINDED_ANALYSIS",
        "experiment_id": config["experiment_id"],
        "budget": BUDGET,
        "class_count": CLASS_COUNT,
        "training_font_count": int(len(np.unique(train_dataset.font_ids))),
        "test_font_count": int(len(np.unique(test_dataset.font_ids))),
        "conditions": list(conditions),
        "mapping_seeds": list(seeds),
        "core_methods": list(core_methods),
        "moderate_only_ablation_methods": list(ablation_methods),
        "selected_linear_svc_c": selected_c,
        "prediction_row_count": len(prediction_rows),
        "condition_run_count": len(condition_progress),
        "selection_audit": selection_audit,
        "feature_audit": feature_audit,
        "observed_peak_rss_bytes": observed_peak_rss,
        "lock_audit": lock_audit,
        "predictions_sha256": sha256_file(temporary / "predictions.csv"),
        "accuracy_aggregated": False,
        "bootstrap_executed": False,
    }
    _atomic_json(temporary / "manifest.json", manifest)
    _atomic_json(
        temporary / "completion.json",
        {
            "status": "PASS_E1_PREDICTIONS_COMMITTED",
            "prediction_row_count": len(prediction_rows),
            "predictions_sha256": manifest["predictions_sha256"],
            "lock_sha256": lock_audit["lock_sha256"],
            "accuracy_aggregated": False,
        },
    )
    temporary.replace(output)
    completion = {
        "status": "PASS_E1_PREDICTIONS_COMMITTED",
        "output_dir": str(output),
        "prediction_row_count": len(prediction_rows),
        "predictions_sha256": manifest["predictions_sha256"],
        "lock_sha256": lock_audit["lock_sha256"],
    }
    _atomic_json(completion_marker, completion)
    return completion


def analyze_confirmation(config_path: str | Path) -> dict[str, Any]:
    """Unblind only after all predictions have been atomically committed."""

    from src.e1.statistics import build_statistics_report, write_statistics_outputs

    config = load_yaml(config_path)
    output = _resolve(config, config["confirmation"]["output_dir"])
    completion = output / "completion.json"
    if not completion.is_file():
        raise E1ExperimentError("E1 predictions are not fully committed")
    with (output / "predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    comparisons: list[tuple[str, str, str]] = [
        (
            "moderate",
            "proto_skeleton_ideal_plus_moderate1",
            "proto_skeleton_ideal_only",
        ),
        (
            "moderate",
            "proto_skeleton_ideal_plus_moderate2",
            "proto_skeleton_ideal_only",
        ),
        (
            "moderate",
            "proto_skeleton_multi4_no_ideal",
            "proto_skeleton_ideal_plus_multi4",
        ),
        (
            "moderate",
            "svc_skeleton_ideal_plus_multi4",
            "svc_skeleton_ideal_only",
        ),
        (
            "moderate",
            "proto_scialom_contour_ideal_plus_multi4",
            "proto_skeleton_ideal_plus_multi4",
        ),
    ]
    for condition in ("mild", "severe", "position_only", "scale_only"):
        comparisons.extend(
            [
                (
                    condition,
                    "proto_skeleton_ideal_plus_multi4",
                    "proto_skeleton_ideal_only",
                ),
                (
                    condition,
                    "svc_skeleton_ideal_plus_multi4",
                    "svc_skeleton_ideal_only",
                ),
                (
                    condition,
                    "proto_scialom_contour_ideal_plus_multi4",
                    "proto_skeleton_ideal_plus_multi4",
                ),
            ]
        )
    report = build_statistics_report(rows, comparisons=comparisons)
    report["status"] = "E1_ANALYSIS_COMPLETE"
    report["interpretation_rule"] = {
        "primary_success": "character_cluster_bootstrap_ci95_lower_gt_0",
        "primary_failure": "otherwise",
        "secondary_results": "descriptive_unless_explicitly_adjusted",
    }
    write_statistics_outputs(output, report)
    return report
