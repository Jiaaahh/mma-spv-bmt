"""Result-blind Stage-10B formal-condition execution orchestration.

This module binds the already-frozen scientific components into one auditable
condition runner.  It does not change encoder, projector, SPV, HOG, classifier,
mapping, split, or parameter implementations.  Formal pixels are read only
after :func:`record_formal_run_started` has committed the session access event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from src.classifier import LinearSvcClassifier, classification_accuracy
from src.contracts import ElectrodeGrid
from src.coverage import build_ideal_overlap_matrix, greedy_coverage_order
from src.deterministic_io import (
    array_sha256,
    sha256_file,
    tree_sha256,
    write_deterministic_npz,
)
from src.encoders import build_encoder
from src.encoders.base import deterministic_topk
from src.features import HogFeatureExtractor
from src.grid import make_regular_grid
from src.spv_forward import (
    GaussianPhospheneForwardModel,
    IdealGaussianUtilityProjector,
    expected_output_energy,
)
from src.stage9_development import (
    build_weighted_eb_representation,
    encoder_bundle_paths,
    load_frozen_mapping,
)
from src.stage10a_preflight import (
    BENCHMARK_ID,
    DEVELOPMENT_FONTS,
    LOCKED_TEST_FONTS,
)
from src.stage10b_session import (
    ACTIVE_SESSION_POINTER,
    ANALYSIS_PLAN_PATH,
    EXPECTED_ENCODER_BUNDLE_SHA256,
    EXPECTED_FROZEN_FILES,
    EXPECTED_RUN_PLAN_SHA256,
    REQUIRED_PREDICTION_COLUMNS,
    RUN_PLAN_PATH,
    WORK_ROOT,
    Stage10BSessionError,
    commit_completed_run,
    create_run_staging_dir,
    load_resume_state,
    record_formal_run_started,
    utc_now,
    write_json_atomic,
)


FORMAL_GLYPH_PATH = Path("data/processed/benchmark/glyphs.npz")
FORMAL_CONFIG_PATH = Path("configs/formal_experiment_frozen.json")
FORMAL_PARAMETER_PATH = Path("configs/formal_parameters_frozen.json")
FORMAL_MAPPING_ROOT = Path("data/mappings/formal")
EXPECTED_IMAGE_SHAPE = (96, 96)
EXPECTED_SAMPLE_COUNT = 1440
EXPECTED_TRAIN_COUNT = 1080
EXPECTED_TEST_COUNT = 360
EXPECTED_CLASS_COUNT = 180
EXPECTED_CANDIDATE_COUNT = 256
EXPECTED_HOG_LENGTH = 4356
ENERGY_RELATIVE_TOLERANCE = 1.0e-5
FORMAL_METHODS = (
    "raw_topk",
    "edge_topk",
    "skeleton_topk",
    "eb_topk",
    "eb_sse",
)


class Stage10BRunnerError(ValueError):
    """Raised before a condition with inconsistent formal bindings can commit."""


@dataclass(frozen=True, slots=True)
class FormalDataset:
    images: NDArray[np.float32] = field(repr=False)
    class_ids: NDArray[np.int64] = field(repr=False)
    sample_ids: NDArray[np.str_] = field(repr=False)
    characters: NDArray[np.str_] = field(repr=False)
    font_ids: NDArray[np.str_] = field(repr=False)
    splits: NDArray[np.str_] = field(repr=False)
    train_indices: NDArray[np.int64] = field(repr=False)
    test_indices: NDArray[np.int64] = field(repr=False)

    def __post_init__(self) -> None:
        if self.images.shape != (EXPECTED_SAMPLE_COUNT, *EXPECTED_IMAGE_SHAPE):
            raise Stage10BRunnerError("formal images must have shape (1440,96,96)")
        if self.images.dtype != np.float32:
            raise Stage10BRunnerError("formal images must use float32")
        for name in ("class_ids", "sample_ids", "characters", "font_ids", "splits"):
            value = getattr(self, name)
            if value.shape != (EXPECTED_SAMPLE_COUNT,):
                raise Stage10BRunnerError(f"formal {name} must contain 1440 rows")
        if self.class_ids.dtype != np.int64:
            raise Stage10BRunnerError("formal class IDs must use int64")
        if self.train_indices.shape != (EXPECTED_TRAIN_COUNT,):
            raise Stage10BRunnerError("formal refit split must contain 1080 rows")
        if self.test_indices.shape != (EXPECTED_TEST_COUNT,):
            raise Stage10BRunnerError("locked-test split must contain 360 rows")


@dataclass(frozen=True, slots=True)
class ConditionExecutionResult:
    run_id: str
    record_sha256: str
    output_path: str
    output_file_count: int
    completed_count: int
    next_run_id: str | None
    hash_verified: bool


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage10BRunnerError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise Stage10BRunnerError(f"JSON root must be an object: {path}")
    return value


def _write_predictions_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_PREDICTION_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    temporary.replace(path)


def _active_session_document(project_root: Path) -> dict[str, Any]:
    pointer = _read_json(project_root / ACTIVE_SESSION_POINTER)
    session_path = project_root / str(pointer.get("session_record_path", ""))
    return _read_json(session_path)


def _reserve_or_resume_staging(
    project_root: Path,
    run_id: str,
) -> tuple[Path, str, bool]:
    """Start a new condition or reuse its auditable interrupted partial dir.

    A partial directory may be reused only when the access-ledger contains one
    start event and no completion event for the exact current run.  The start
    event is never duplicated, so counters and seed access remain monotonic.
    """

    state = load_resume_state(project_root)
    if run_id != state.next_run_id:
        raise Stage10BRunnerError("only the exact next run can be staged or resumed")
    session = _active_session_document(project_root)
    ledger_path = project_root / str(session["paths"]["access_ledger"])
    ledger = _read_json(ledger_path)
    events = ledger.get("events")
    if not isinstance(events, list):
        raise Stage10BRunnerError("formal access-ledger events are invalid")
    starts = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event_type") == "FORMAL_RUN_STARTED"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("run_id") == run_id
    ]
    completions = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event_type") == "FORMAL_RUN_COMPLETED_AND_CHECKPOINTED"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("run_id") == run_id
    ]
    partial = (
        project_root
        / WORK_ROOT
        / state.formal_session_id
        / f"{run_id}.partial"
    )
    if partial.exists():
        if not partial.is_dir():
            raise Stage10BRunnerError("formal partial staging path is not a directory")
        if len(starts) != 1 or completions:
            raise Stage10BRunnerError(
                "partial staging is not bound to one interrupted run-start event"
            )
        return partial, str(starts[0]["timestamp_utc"]), True
    if starts or completions:
        raise Stage10BRunnerError(
            "run has ledger history but its interrupted staging directory is missing"
        )
    partial = create_run_staging_dir(project_root, run_id)
    record_formal_run_started(project_root, run_id)
    session = _active_session_document(project_root)
    ledger = _read_json(project_root / str(session["paths"]["access_ledger"]))
    events = ledger.get("events")
    if not isinstance(events, list) or not events:
        raise Stage10BRunnerError("formal run-start event was not recorded")
    last = events[-1]
    if (
        last.get("event_type") != "FORMAL_RUN_STARTED"
        or last.get("payload", {}).get("run_id") != run_id
    ):
        raise Stage10BRunnerError("formal run-start event does not bind the run ID")
    return partial, str(last["timestamp_utc"]), False


def validate_runtime_binding(project_root: str | Path) -> dict[str, Any]:
    """Require the exact Python/package environment captured by this session."""

    root = Path(project_root).resolve()
    session = _active_session_document(root)
    expected = session.get("environment")
    if not isinstance(expected, dict):
        raise Stage10BRunnerError("formal session lacks an environment snapshot")
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "scikit-image", "scikit-learn", "Pillow"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise Stage10BRunnerError(f"required package is missing: {name}") from exc
    actual = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable_name": Path(sys.executable).name,
        "package_versions": packages,
        "cpu_only_protocol": True,
    }
    for key in ("python_version", "python_implementation", "executable_name"):
        if actual[key] != expected.get(key):
            raise Stage10BRunnerError(
                f"runtime {key} differs from the formal-session snapshot"
            )
    if packages != expected.get("package_versions"):
        raise Stage10BRunnerError("runtime package versions changed")
    if expected.get("cpu_only_protocol") is not True:
        raise Stage10BRunnerError("formal session is not bound to CPU-only execution")
    return actual


def _load_plan(project_root: Path) -> dict[str, Any]:
    plan = _read_json(project_root / RUN_PLAN_PATH)
    if plan.get("benchmark_id") != BENCHMARK_ID:
        raise Stage10BRunnerError("formal run plan benchmark changed")
    return plan


def _condition_for_next_run(
    project_root: Path,
    run_id: str,
) -> tuple[dict[str, Any], str]:
    state = load_resume_state(project_root)
    if state.formal_session_id != _read_json(
        project_root / ACTIVE_SESSION_POINTER
    ).get("formal_session_id"):
        raise Stage10BRunnerError("active formal-session identity changed")
    if run_id != state.next_run_id:
        raise Stage10BRunnerError("requested run is not the exact next condition")
    plan = _load_plan(project_root)
    condition = dict(plan["conditions"][state.completed_count])
    if condition.get("run_id") != run_id:
        raise Stage10BRunnerError("run-plan condition index and run ID differ")
    if condition.get("method") not in FORMAL_METHODS:
        raise Stage10BRunnerError("run-plan condition contains an unknown method")
    if condition.get("budget") not in {32, 64, 128}:
        raise Stage10BRunnerError("run-plan condition contains an unknown budget")
    return condition, state.formal_session_id


def preflight_batch(
    project_root: str | Path,
    run_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate an exact run-plan prefix without reading formal image pixels."""

    root = Path(project_root).resolve()
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise Stage10BRunnerError("batch run IDs must be non-empty and unique")
    state = load_resume_state(root)
    expected = state.remaining_run_ids[: len(run_ids)]
    if tuple(run_ids) != expected:
        raise Stage10BRunnerError("batch run IDs are not the exact remaining-plan prefix")
    runtime = validate_runtime_binding(root)
    return {
        "formal_session_id": state.formal_session_id,
        "run_plan_sha256": EXPECTED_RUN_PLAN_SHA256,
        "completed_before": state.completed_count,
        "requested_run_ids": list(run_ids),
        "runtime": runtime,
        "locked_test_images_array_read": False,
        "status": "PASS_READY_TO_EXECUTE_EXACT_PREFIX",
    }


def _load_formal_dataset(
    project_root: Path,
    condition: Mapping[str, Any],
) -> FormalDataset:
    """Read all formal pixels after the caller has committed a run-start event."""

    path = project_root / FORMAL_GLYPH_PATH
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
            raise Stage10BRunnerError("formal glyph archive is missing required arrays")
        raw_images = archive["images"]
        if raw_images.shape != (EXPECTED_SAMPLE_COUNT, *EXPECTED_IMAGE_SHAPE):
            raise Stage10BRunnerError("formal image array shape changed")
        if raw_images.dtype != np.float32:
            raise Stage10BRunnerError("formal image array dtype changed")
        images = np.ascontiguousarray(raw_images, dtype=np.float32)
        class_ids = np.ascontiguousarray(archive["class_ids"], dtype=np.int64)
        sample_ids = np.asarray(archive["sample_ids"]).copy()
        characters = np.asarray(archive["characters"]).copy()
        font_ids = np.asarray(archive["font_ids"]).copy()
        splits = np.asarray(archive["splits"]).copy()

    if not np.isfinite(images).all() or float(images.min()) < 0.0 or float(images.max()) > 1.0:
        raise Stage10BRunnerError("formal images contain invalid values")
    if np.any(images.max(axis=(1, 2)) <= 0.0):
        raise Stage10BRunnerError("formal glyph archive contains an empty image")
    if len(set(map(str, sample_ids))) != EXPECTED_SAMPLE_COUNT:
        raise Stage10BRunnerError("formal sample IDs are not unique")
    if set(map(int, class_ids)) != set(range(EXPECTED_CLASS_COUNT)):
        raise Stage10BRunnerError("formal class IDs are not exactly 0..179")

    train_fonts = tuple(map(str, condition["train_font_ids"]))
    test_fonts = tuple(map(str, condition["test_font_ids"]))
    if train_fonts != DEVELOPMENT_FONTS or test_fonts != LOCKED_TEST_FONTS:
        raise Stage10BRunnerError("condition font split differs from the frozen split")
    train_mask = np.isin(font_ids, train_fonts)
    test_mask = np.isin(font_ids, test_fonts)
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise Stage10BRunnerError("formal train/test masks overlap or omit samples")
    if int(train_mask.sum()) != EXPECTED_TRAIN_COUNT or int(test_mask.sum()) != EXPECTED_TEST_COUNT:
        raise Stage10BRunnerError("formal train/test sample counts changed")
    if not np.all(np.isin(splits[train_mask], ("train", "validation"))):
        raise Stage10BRunnerError("formal refit rows contain an invalid split label")
    if not np.all(splits[test_mask] == "test"):
        raise Stage10BRunnerError("locked-test rows contain an invalid split label")
    for font_id in (*train_fonts, *test_fonts):
        if int(np.sum(font_ids == font_id)) != EXPECTED_CLASS_COUNT:
            raise Stage10BRunnerError(f"font {font_id} does not contain 180 classes")
    for class_id in range(EXPECTED_CLASS_COUNT):
        class_mask = class_ids == class_id
        if len(set(map(str, characters[class_mask]))) != 1:
            raise Stage10BRunnerError("one class ID maps to multiple characters")
        if int(np.sum(class_mask & test_mask)) != len(test_fonts):
            raise Stage10BRunnerError("locked-test class/font coverage is incomplete")

    return FormalDataset(
        images=images,
        class_ids=class_ids,
        sample_ids=sample_ids,
        characters=characters,
        font_ids=font_ids,
        splits=splits,
        train_indices=np.flatnonzero(train_mask).astype(np.int64),
        test_indices=np.flatnonzero(test_mask).astype(np.int64),
    )


def _load_grid_projector_forward(
    project_root: Path,
    condition: Mapping[str, Any],
) -> tuple[
    ElectrodeGrid,
    IdealGaussianUtilityProjector,
    GaussianPhospheneForwardModel,
    dict[str, Any],
]:
    ideal_path = project_root / FORMAL_MAPPING_ROOT / "ideal_mapping.npz"
    ideal_mapping = load_frozen_mapping(ideal_path, expected_seed=0)
    regular_grid = make_regular_grid(
        rows=16,
        cols=16,
        image_shape=EXPECTED_IMAGE_SHAPE,
    )
    if not np.array_equal(ideal_mapping.centers_xy, regular_grid.centers_xy):
        raise Stage10BRunnerError("ideal mapping centers differ from the 16x16 grid")
    # ``RegularGrid`` stores the same frozen row-major geometry but is a helper
    # record rather than the public encoder contract type.  Adapt the geometry
    # without modifying values or any frozen encoder implementation.
    grid = ElectrodeGrid(
        centers_xy=np.ascontiguousarray(regular_grid.centers_xy, dtype=np.float32),
        image_shape=regular_grid.image_shape,
    )
    projector = IdealGaussianUtilityProjector(ideal_mapping)

    mapping_type = str(condition["mapping_type"])
    mapping_seed = int(condition["mapping_seed"])
    if mapping_type == "moderate":
        target_path = (
            project_root
            / FORMAL_MAPPING_ROOT
            / f"moderate_seed_{mapping_seed}.npz"
        )
    elif mapping_type == "ideal" and mapping_seed == 0:
        target_path = ideal_path
    else:
        raise Stage10BRunnerError("condition has an invalid mapping binding")
    target_mapping = load_frozen_mapping(target_path, expected_seed=mapping_seed)
    if target_mapping.candidate_count != EXPECTED_CANDIDATE_COUNT:
        raise Stage10BRunnerError("target mapping candidate count changed")
    forward = GaussianPhospheneForwardModel(
        target_mapping,
        amplitude_scale=1.0,
        kernel_normalization="l1",
    )

    manifest = _read_json(
        project_root / FORMAL_MAPPING_ROOT / "mapping_bank_manifest.json"
    )
    entries = [
        item
        for item in manifest.get("mappings", [])
        if isinstance(item, dict) and int(item.get("seed", -1)) == mapping_seed
    ]
    if len(entries) != 1:
        raise Stage10BRunnerError("mapping manifest seed binding is ambiguous")
    target_entry = entries[0]
    actual_kernel_hash = array_sha256(forward.kernel_bank)
    if actual_kernel_hash != target_entry.get("kernel_bank_sha256"):
        raise Stage10BRunnerError("target mapping kernel-bank hash changed")
    ideal_entries = [
        item
        for item in manifest.get("mappings", [])
        if isinstance(item, dict) and int(item.get("seed", -1)) == 0
    ]
    if len(ideal_entries) != 1 or array_sha256(projector.kernel_bank) != ideal_entries[0].get(
        "kernel_bank_sha256"
    ):
        raise Stage10BRunnerError("ideal projector kernel-bank hash changed")
    return grid, projector, forward, {
        "target_mapping_path": target_path.relative_to(project_root).as_posix(),
        "target_mapping_file_sha256": sha256_file(target_path),
        "target_kernel_bank_sha256": actual_kernel_hash,
        "ideal_projector_kernel_bank_sha256": array_sha256(projector.kernel_bank),
    }


def _build_hog_and_classifier(
    project_root: Path,
) -> tuple[HogFeatureExtractor, LinearSvcClassifier, dict[str, Any]]:
    config = _read_json(project_root / FORMAL_CONFIG_PATH)
    feature = config.get("feature")
    classifier = config.get("classifier")
    if not isinstance(feature, dict) or not isinstance(classifier, dict):
        raise Stage10BRunnerError("formal experiment config is incomplete")
    extractor = HogFeatureExtractor(
        image_shape=EXPECTED_IMAGE_SHAPE,
        orientations=int(feature["orientations"]),
        pixels_per_cell=tuple(map(int, feature["pixels_per_cell"])),
        cells_per_block=tuple(map(int, feature["cells_per_block"])),
        block_norm=str(feature["block_norm"]),
    )
    decoder = LinearSvcClassifier(
        c=float(classifier["C"]),
        dual=str(classifier["dual"]),
        max_iter=int(classifier["max_iter"]),
        random_state=int(classifier["random_state"]),
    )
    if extractor.feature_length != int(feature["feature_length"]):
        raise Stage10BRunnerError("formal HOG length binding changed")
    if float(classifier.get("tol")) != 0.0001:
        raise Stage10BRunnerError("formal LinearSVC tolerance binding changed")
    return extractor, decoder, config


def _encode_all(
    dataset: FormalDataset,
    grid: ElectrodeGrid,
    projector: IdealGaussianUtilityProjector,
    *,
    method: str,
    budget: int,
    alpha_e: float,
    alpha_j: float,
) -> tuple[NDArray[np.int64], dict[str, Any]]:
    sample_count = dataset.images.shape[0]
    selections = np.empty((sample_count, budget), dtype=np.int64)
    scores = np.empty((sample_count, EXPECTED_CANDIDATE_COUNT), dtype=np.float32)
    drive_maps = np.empty((sample_count, *EXPECTED_IMAGE_SHAPE), dtype=np.float32)
    encoder_metadata_keys: set[str] | None = None
    overlap = build_ideal_overlap_matrix(grid) if method == "eb_sse" else None
    encoder = build_encoder(method) if method not in {"eb_topk", "eb_sse"} else None

    for sample_index, image in enumerate(dataset.images):
        if method in {"eb_topk", "eb_sse"}:
            representation = build_weighted_eb_representation(
                image,
                projector,
                alpha_e=alpha_e,
                alpha_j=alpha_j,
            )
            if method == "eb_topk":
                selected = deterministic_topk(representation.candidate_scores, budget)
            else:
                assert overlap is not None
                trace = greedy_coverage_order(
                    representation.candidate_scores,
                    overlap,
                    max_steps=128,
                )
                selected = np.ascontiguousarray(trace.order[:budget], dtype=np.int64)
            candidate_scores = representation.candidate_scores
            drive_map = representation.drive_map
            metadata_keys = {
                "alpha_e",
                "alpha_j",
                "candidate_projection",
                "drive_map",
                "selection",
            }
        else:
            assert encoder is not None
            result = encoder.encode(image, grid, budget, projector=projector)
            selected = result.selected_indices
            candidate_scores = result.candidate_scores
            drive_map = result.drive_map
            metadata_keys = set(map(str, result.metadata.keys()))
        selections[sample_index] = selected
        scores[sample_index] = candidate_scores
        drive_maps[sample_index] = drive_map
        if encoder_metadata_keys is None:
            encoder_metadata_keys = metadata_keys
        elif encoder_metadata_keys != metadata_keys:
            raise Stage10BRunnerError("encoder metadata schema changed between samples")

    if selections.dtype != np.int64 or selections.shape != (sample_count, budget):
        raise Stage10BRunnerError("formal selection matrix shape or dtype changed")
    if any(np.unique(row).size != budget for row in selections):
        raise Stage10BRunnerError("a formal sample has duplicate selected electrodes")
    if np.any(selections < 0) or np.any(selections >= EXPECTED_CANDIDATE_COUNT):
        raise Stage10BRunnerError("a formal selection contains an invalid electrode")
    return selections, {
        "schema_version": 1,
        "method": method,
        "budget": budget,
        "sample_count": sample_count,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "active_electrodes_per_sample_min": int(
            np.min([np.unique(row).size for row in selections])
        ),
        "active_electrodes_per_sample_max": int(
            np.max([np.unique(row).size for row in selections])
        ),
        "all_selected_indices_unique_and_in_range": True,
        "stimulation_amplitude_rule": "unit_selected_electrode",
        "candidate_utility_rule": "l1_ideal_gaussian_projection",
        "formal_alpha_s": 1.0,
        "formal_alpha_e": alpha_e,
        "formal_alpha_j": alpha_j,
        "encoder_metadata_keys": sorted(encoder_metadata_keys or ()),
        "selections_sha256": array_sha256(selections),
        "candidate_utilities_sha256": array_sha256(scores),
        "drive_maps_sha256": array_sha256(drive_maps),
    }


def _prediction_rows(
    dataset: FormalDataset,
    predictions: NDArray[np.int64],
    condition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if predictions.shape != (EXPECTED_TEST_COUNT,) or predictions.dtype != np.int64:
        raise Stage10BRunnerError("formal prediction array shape or dtype changed")
    if np.any(predictions < 0) or np.any(predictions >= EXPECTED_CLASS_COUNT):
        raise Stage10BRunnerError("formal classifier returned an invalid class label")
    rows: list[dict[str, Any]] = []
    for local_index, predicted in zip(dataset.test_indices, predictions, strict=True):
        true_label = int(dataset.class_ids[local_index])
        predicted_label = int(predicted)
        rows.append(
            {
                "benchmark_id": BENCHMARK_ID,
                "run_id": str(condition["run_id"]),
                "method": str(condition["method"]),
                "budget": int(condition["budget"]),
                "mapping_type": str(condition["mapping_type"]),
                "mapping_seed": int(condition["mapping_seed"]),
                "char_id": f"c{true_label:03d}",
                "font_id": str(dataset.font_ids[local_index]),
                "true_label": true_label,
                "pred_label": predicted_label,
                "correct": int(true_label == predicted_label),
            }
        )
    if len({(row["char_id"], row["font_id"]) for row in rows}) != EXPECTED_TEST_COUNT:
        raise Stage10BRunnerError("formal predictions contain missing or duplicate samples")
    return rows


def execute_one_condition(
    project_root: str | Path,
    run_id: str,
) -> ConditionExecutionResult:
    """Execute, atomically commit, and re-verify the exact next formal run."""

    root = Path(project_root).resolve()
    condition, formal_session_id = _condition_for_next_run(root, run_id)
    validate_runtime_binding(root)
    # This is intentionally the last operation before formal mappings/pixels are
    # loaded.  From this point onward, any failure is an auditable engineering
    # interruption in the same irreversible formal session.  An existing
    # partial directory resumes that same event without incrementing counters.
    staging, started_utc, _resumed = _reserve_or_resume_staging(root, run_id)

    dataset = _load_formal_dataset(root, condition)
    grid, projector, forward, mapping_metadata = _load_grid_projector_forward(
        root, condition
    )
    parameters = _read_json(root / FORMAL_PARAMETER_PATH)
    alpha_e = float(parameters["alpha_e"])
    alpha_j = float(parameters["alpha_j"])
    if (alpha_e, alpha_j) != (0.5, 0.5):
        raise Stage10BRunnerError("formal EB alpha pair changed")

    budget = int(condition["budget"])
    method = str(condition["method"])
    selections, selection_metadata = _encode_all(
        dataset,
        grid,
        projector,
        method=method,
        budget=budget,
        alpha_e=alpha_e,
        alpha_j=alpha_j,
    )
    selection_npz_sha = write_deterministic_npz(
        staging / "selections.npz",
        {
            "sample_ids": dataset.sample_ids,
            "selected_indices": selections,
        },
    )
    selection_metadata["selections_file_sha256"] = selection_npz_sha
    write_json_atomic(staging / "selection_metadata.json", selection_metadata)

    amplitudes = np.ones((EXPECTED_SAMPLE_COUNT, budget), dtype=np.float32)
    if np.count_nonzero(amplitudes) != EXPECTED_SAMPLE_COUNT * budget:
        raise Stage10BRunnerError("formal stimulation amplitudes are not all one")
    spv = forward.generate_variable_batch(selections, amplitudes)
    expected_energy = expected_output_energy(amplitudes, amplitude_scale=1.0)
    actual_energy = spv.sum(axis=(1, 2), dtype=np.float64)
    absolute_error = np.abs(actual_energy - expected_energy)
    relative_error = absolute_error / np.maximum(
        expected_energy, np.finfo(np.float64).eps
    )
    max_relative_error = float(relative_error.max())
    if max_relative_error > ENERGY_RELATIVE_TOLERANCE:
        raise Stage10BRunnerError("formal SPV energy exceeds the frozen tolerance")
    spv_sha = array_sha256(spv)
    energy_audit = {
        "schema_version": 1,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "budget": budget,
        "active_electrodes_per_sample": budget,
        "unit_amplitude": True,
        "amplitude_min": float(amplitudes.min()),
        "amplitude_max": float(amplitudes.max()),
        "expected_energy_per_sample": float(budget),
        "actual_energy_min": float(actual_energy.min()),
        "actual_energy_max": float(actual_energy.max()),
        "max_energy_absolute_error": float(absolute_error.max()),
        "max_energy_relative_error": max_relative_error,
        "relative_tolerance": ENERGY_RELATIVE_TOLERANCE,
        "status": "PASS",
        "spv_array_sha256": spv_sha,
    }
    write_json_atomic(staging / "energy_audit.json", energy_audit)

    extractor, classifier, formal_config = _build_hog_and_classifier(root)
    features = extractor.extract_batch(spv)
    del spv
    if features.shape != (EXPECTED_SAMPLE_COUNT, EXPECTED_HOG_LENGTH):
        raise Stage10BRunnerError("formal HOG array shape changed")
    if not np.isfinite(features).all():
        raise Stage10BRunnerError("formal HOG array contains NaN or Inf")
    hog_sha = array_sha256(features)
    hog_metadata = {
        "schema_version": 1,
        "name": extractor.name,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "feature_length": extractor.feature_length,
        "orientations": extractor.orientations,
        "pixels_per_cell": list(extractor.pixels_per_cell),
        "cells_per_block": list(extractor.cells_per_block),
        "block_norm": extractor.block_norm,
        "transform_sqrt": False,
        "finite": True,
        "hog_features_sha256": hog_sha,
    }
    write_json_atomic(staging / "hog_metadata.json", hog_metadata)

    classifier.fit(
        features[dataset.train_indices],
        dataset.class_ids[dataset.train_indices],
    )
    predictions = classifier.predict(features[dataset.test_indices])
    true_labels = dataset.class_ids[dataset.test_indices]
    accuracy = classification_accuracy(true_labels, predictions)
    macro_f1 = float(
        f1_score(
            true_labels,
            predictions,
            labels=np.arange(EXPECTED_CLASS_COUNT, dtype=np.int64),
            average="macro",
            zero_division=0,
        )
    )
    prediction_array_sha = array_sha256(predictions)
    classifier_metadata = {
        "schema_version": 1,
        "name": classifier.name,
        "C": classifier.c,
        "dual": classifier.dual,
        "max_iter": classifier.max_iter,
        "random_state": classifier.random_state,
        "tol": float(formal_config["classifier"]["tol"]),
        "convergence_warning_policy": formal_config["classifier"][
            "convergence_warning_policy"
        ],
        "n_iter": classifier.n_iter,
        "train_sample_count": int(dataset.train_indices.size),
        "train_class_count": int(
            np.unique(dataset.class_ids[dataset.train_indices]).size
        ),
        "feature_length": int(features.shape[1]),
        "status": "PASS_CONVERGED",
    }
    write_json_atomic(staging / "classifier_metadata.json", classifier_metadata)

    rows = _prediction_rows(dataset, predictions, condition)
    _write_predictions_atomic(staging / "predictions.csv", rows)
    predictions_file_sha = sha256_file(staging / "predictions.csv")

    correct_count = int(np.sum(true_labels == predictions))
    metrics = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "run_id": run_id,
        "method": method,
        "budget": budget,
        "mapping_type": str(condition["mapping_type"]),
        "mapping_seed": int(condition["mapping_seed"]),
        "train_sample_count": int(dataset.train_indices.size),
        "test_sample_count": int(dataset.test_indices.size),
        "prediction_row_count": len(rows),
        "class_count": EXPECTED_CLASS_COUNT,
        "correct_count": correct_count,
        "top1_accuracy": accuracy,
        "macro_f1": macro_f1,
        "active_electrodes_per_sample": budget,
        "unit_amplitude": True,
        "max_energy_relative_error": max_relative_error,
        "hog_feature_length": int(features.shape[1]),
        "linear_svc_n_iter": classifier.n_iter,
        "all_integrity_checks_passed": True,
    }
    write_json_atomic(staging / "metrics.json", metrics)

    frozen_input_hashes = {
        name: sha256_file(root / relative)
        for name, (relative, _expected) in EXPECTED_FROZEN_FILES.items()
    }
    frozen_input_hashes.update(
        {
            "run_plan": sha256_file(root / RUN_PLAN_PATH),
            "analysis_plan": sha256_file(root / ANALYSIS_PLAN_PATH),
            "encoder_implementation_bundle": tree_sha256(
                root, encoder_bundle_paths(root)
            ),
        }
    )
    if frozen_input_hashes["encoder_implementation_bundle"] != EXPECTED_ENCODER_BUNDLE_SHA256:
        raise Stage10BRunnerError("encoder implementation bundle changed during run")
    output_array_hashes = {
        "selections_sha256": selection_metadata["selections_sha256"],
        "candidate_utilities_sha256": selection_metadata[
            "candidate_utilities_sha256"
        ],
        "drive_maps_sha256": selection_metadata["drive_maps_sha256"],
        "spv_sha256": spv_sha,
        "hog_features_sha256": hog_sha,
        "predictions_array_sha256": prediction_array_sha,
        "predictions_csv_sha256": predictions_file_sha,
    }
    input_output_hashes = {
        "schema_version": 1,
        "run_id": run_id,
        "frozen_inputs": frozen_input_hashes,
        "mapping": mapping_metadata,
        "outputs": output_array_hashes,
    }
    write_json_atomic(staging / "input_output_hashes.json", input_output_hashes)

    completed_utc = utc_now()
    run_manifest = {
        "schema_version": 1,
        "benchmark_id": BENCHMARK_ID,
        "formal_session_id": formal_session_id,
        "run_id": run_id,
        "run_plan_sha256": EXPECTED_RUN_PLAN_SHA256,
        "method": method,
        "budget": budget,
        "mapping_type": str(condition["mapping_type"]),
        "mapping_seed": int(condition["mapping_seed"]),
        "planned_output_path": str(condition["planned_output_path"]),
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "train_font_ids": list(map(str, condition["train_font_ids"])),
        "test_font_ids": list(map(str, condition["test_font_ids"])),
        "train_sample_count": int(dataset.train_indices.size),
        "test_sample_count": int(dataset.test_indices.size),
        "formal_parameters": {
            "alpha_s": 1.0,
            "alpha_e": alpha_e,
            "alpha_j": alpha_j,
        },
        "protocol": {
            "candidate_utility": "l1_ideal_gaussian_projection",
            "stimulation_amplitude": "unit_selected_electrode",
            "spv_kernel_normalization": "l1_after_canvas_boundary",
            "spv_output_normalization": "none",
            "classifier": "LinearSVC",
            "feature": "HOG",
        },
        "runtime": validate_runtime_binding(root),
        "frozen_input_hashes": frozen_input_hashes,
        "mapping_hashes": mapping_metadata,
        "output_array_hashes": output_array_hashes,
        "integrity": {
            "train_test_font_leakage": False,
            "missing_or_duplicate_predictions": False,
            "nan_or_inf": False,
            "active_electrode_count_pass": True,
            "unit_amplitude_pass": True,
            "energy_pass": True,
            "hog_dimension_pass": True,
            "classifier_convergence_pass": True,
            "status": "PASS_READY_FOR_ATOMIC_COMMIT",
        },
    }
    write_json_atomic(staging / "run_manifest.json", run_manifest)
    del features

    record = commit_completed_run(root, run_id)
    post_state = load_resume_state(root)
    if post_state.completed_count < 1 or post_state.completed_run_ids[-1] != run_id:
        raise Stage10BRunnerError("post-commit checkpoint did not advance")
    return ConditionExecutionResult(
        run_id=run_id,
        record_sha256=str(record["record_sha256"]),
        output_path=str(record["output_path"]),
        output_file_count=len(record["files"]),
        completed_count=post_state.completed_count,
        next_run_id=post_state.next_run_id,
        hash_verified=True,
    )
