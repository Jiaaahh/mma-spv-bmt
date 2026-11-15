"""Small, strict YAML configuration layer for the Hanzi-SPV benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing, malformed, or inconsistent."""


# Exact keys catch misspellings before an experiment silently uses a default.
SECTION_KEYS: dict[str, set[str]] = {
    "project": {"name", "stage", "description"},
    "runtime": {"python", "device", "seed", "deterministic", "max_workers"},
    "paths": {
        "project_root",
        "data_dir",
        "registry_dir",
        "processed_dir",
        "output_dir",
        "log_dir",
    },
    "dataset": {"character_count", "character_file", "font_manifest", "split"},
    "dataset.split": {
        "train_fonts",
        "validation_fonts",
        "test_fonts",
        "enforce_font_disjointness",
    },
    "rendering": {
        "image_size",
        "render_canvas_size",
        "font_size",
        "content_margin",
        "grayscale_range",
        "foreground_value",
        "background_value",
        "preserve_aspect_ratio",
    },
    "electrode_grid": {"rows", "cols"},
    "experiment": {"encoders", "budgets"},
    "protocol": {
        "candidate_utility",
        "stimulation_amplitude",
        "topology_detection",
        "result_status",
    },
    "mapping": {"names", "seeds", "moderate"},
    "mapping.moderate": {
        "position_jitter_std_grid",
        "sigma_scale_std",
        "sigma_scale_clip",
        "clip_centers_to_canvas",
    },
    "spv": {
        "base_sigma_grid",
        "amplitude",
        "kernel_normalization",
        "output_normalization",
        "output_dtype",
    },
    "feature": {
        "name",
        "orientations",
        "pixels_per_cell",
        "cells_per_block",
        "block_norm",
    },
    "classifier": {"name", "c", "dual", "max_iter", "random_state"},
    "storage": {"save_spv", "save_selection", "cache_features", "overwrite"},
    "validation": {
        "expected_candidate_count",
        "expected_hog_length",
        "energy_relative_tolerance",
        "require_unique_electrodes",
    },
}

ROOT_KEYS = {"schema_version", *SECTION_KEYS.keys()} - {
    "dataset.split",
    "mapping.moderate",
}


def _at(values: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = values
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"missing configuration key: {dotted_key}")
        current = current[part]
    return current


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _check_exact_keys(node: Mapping[str, Any], field: str, expected: set[str]) -> None:
    missing = sorted(expected - set(node))
    unknown = sorted(set(node) - expected)
    if missing:
        raise ConfigError(f"{field} is missing keys: {missing}")
    if unknown:
        raise ConfigError(f"{field} contains unknown keys: {unknown}")


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{field} must be >= {minimum}")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be numeric")
    return float(value)


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be true or false")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _list_of_strings(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{field} must be a YAML list")
    result = tuple(_string(item, f"{field}[{index}]") for index, item in enumerate(value))
    if not result and not allow_empty:
        raise ConfigError(f"{field} must not be empty")
    if len(set(result)) != len(result):
        raise ConfigError(f"{field} must not contain duplicates")
    return result


def _list_of_ints(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty YAML list")
    result = tuple(_integer(item, f"{field}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ConfigError(f"{field} must not contain duplicates")
    return result


def _pair(value: Any, field: str, converter: Any) -> tuple[Any, Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{field} must contain exactly two values")
    return converter(value[0], f"{field}[0]"), converter(value[1], f"{field}[1]")


def _safe_project_path(root: Path, value: Any, field: str) -> Path:
    relative = Path(_string(value, field))
    if relative.is_absolute():
        raise ConfigError(f"{field} must be relative to paths.project_root")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"{field} must not escape paths.project_root") from exc
    return resolved


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def calculate_hog_length(
    image_size: tuple[int, int],
    orientations: int,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
) -> int:
    """Calculate the feature length produced by skimage.feature.hog."""

    height, width = image_size
    cell_h, cell_w = pixels_per_cell
    block_h, block_w = cells_per_block
    if height % cell_h or width % cell_w:
        raise ConfigError("image_size must be divisible by pixels_per_cell")
    blocks_h = height // cell_h - block_h + 1
    blocks_w = width // cell_w - block_w + 1
    if blocks_h <= 0 or blocks_w <= 0:
        raise ConfigError("cells_per_block is larger than the HOG cell grid")
    return blocks_h * blocks_w * block_h * block_w * orientations


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Immutable view of one validated YAML file."""

    source_path: Path
    project_root: Path
    values: Mapping[str, Any]

    def get(self, dotted_key: str) -> Any:
        return _at(self.values, dotted_key)

    def project_path(self, dotted_key: str) -> Path:
        return _safe_project_path(self.project_root, self.get(dotted_key), dotted_key)

    @property
    def stage(self) -> str:
        return str(self.get("project.stage"))

    @property
    def device(self) -> str:
        return str(self.get("runtime.device"))

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(self.get("rendering.image_size"))  # type: ignore[return-value]

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.get("electrode_grid.rows")), int(self.get("electrode_grid.cols"))

    @property
    def candidate_count(self) -> int:
        rows, cols = self.grid_shape
        return rows * cols

    @property
    def budgets(self) -> tuple[int, ...]:
        return tuple(self.get("experiment.budgets"))

    @property
    def encoders(self) -> tuple[str, ...]:
        return tuple(self.get("experiment.encoders"))

    @property
    def mapping_names(self) -> tuple[str, ...]:
        return tuple(self.get("mapping.names"))

    @property
    def mapping_seeds(self) -> tuple[int, ...]:
        return tuple(self.get("mapping.seeds"))

    @property
    def hog_feature_length(self) -> int:
        return calculate_hog_length(
            self.image_size,
            int(self.get("feature.orientations")),
            tuple(self.get("feature.pixels_per_cell")),  # type: ignore[arg-type]
            tuple(self.get("feature.cells_per_block")),  # type: ignore[arg-type]
        )


def _validate(raw: Mapping[str, Any], source_path: Path) -> Path:
    _check_exact_keys(raw, "root", ROOT_KEYS)
    for field, expected_keys in SECTION_KEYS.items():
        _check_exact_keys(_mapping(_at(raw, field), field), field, expected_keys)

    if _integer(raw["schema_version"], "schema_version") != 1:
        raise ConfigError("schema_version must be 1")

    stage = _string(_at(raw, "project.stage"), "project.stage").lower()
    if stage not in {
        "mvp",
        "stage4_5",
        "stage5",
        "stage6",
        "stage7",
        "development",
        "benchmark",
    }:
        raise ConfigError(
            "project.stage must be mvp, stage4_5, stage5, stage6, stage7, "
            "development, or benchmark"
        )
    _string(_at(raw, "project.name"), "project.name")
    _string(_at(raw, "project.description"), "project.description")

    if _string(_at(raw, "runtime.python"), "runtime.python") != "3.11":
        raise ConfigError("runtime.python must be exactly '3.11'")
    if _string(_at(raw, "runtime.device"), "runtime.device").lower() != "cpu":
        raise ConfigError("runtime.device must be 'cpu'; GPU execution is out of scope")
    seed = _integer(_at(raw, "runtime.seed"), "runtime.seed", minimum=0)
    if not _boolean(_at(raw, "runtime.deterministic"), "runtime.deterministic"):
        raise ConfigError("runtime.deterministic must be true")
    max_workers = _integer(_at(raw, "runtime.max_workers"), "runtime.max_workers", minimum=1)

    root_spec = Path(_string(_at(raw, "paths.project_root"), "paths.project_root"))
    if root_spec.is_absolute():
        raise ConfigError("paths.project_root must be relative to the YAML file")
    project_root = (source_path.parent / root_spec).resolve()
    for key in (
        "paths.data_dir",
        "paths.registry_dir",
        "paths.processed_dir",
        "paths.output_dir",
        "paths.log_dir",
        "dataset.character_file",
        "dataset.font_manifest",
    ):
        _safe_project_path(project_root, _at(raw, key), key)

    character_count = _integer(
        _at(raw, "dataset.character_count"), "dataset.character_count", minimum=2
    )
    train_fonts = _list_of_strings(_at(raw, "dataset.split.train_fonts"), "train_fonts")
    validation_fonts = _list_of_strings(
        _at(raw, "dataset.split.validation_fonts"), "validation_fonts", allow_empty=True
    )
    test_fonts = _list_of_strings(_at(raw, "dataset.split.test_fonts"), "test_fonts")
    if not _boolean(
        _at(raw, "dataset.split.enforce_font_disjointness"), "enforce_font_disjointness"
    ):
        raise ConfigError("dataset.split.enforce_font_disjointness must be true")
    split_sets = {"train": set(train_fonts), "validation": set(validation_fonts), "test": set(test_fonts)}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sorted(split_sets[left] & split_sets[right])
        if overlap:
            raise ConfigError(f"font leakage: {left} and {right} overlap: {overlap}")

    image_size = _pair(_at(raw, "rendering.image_size"), "image_size", _integer)
    if any(value <= 0 for value in image_size):
        raise ConfigError("rendering.image_size values must be positive")
    render_canvas_size = _pair(
        _at(raw, "rendering.render_canvas_size"), "render_canvas_size", _integer
    )
    if any(value <= 0 for value in render_canvas_size):
        raise ConfigError("render_canvas_size values must be positive")
    font_size = _integer(_at(raw, "rendering.font_size"), "font_size", minimum=1)
    content_margin = _integer(
        _at(raw, "rendering.content_margin"), "content_margin", minimum=0
    )
    if 2 * content_margin >= min(image_size):
        raise ConfigError("content_margin leaves no drawable output area")
    if font_size > min(render_canvas_size):
        raise ConfigError("font_size must not exceed the smallest render canvas dimension")
    gray_low, gray_high = _pair(
        _at(raw, "rendering.grayscale_range"), "grayscale_range", _number
    )
    foreground = _number(_at(raw, "rendering.foreground_value"), "foreground_value")
    background = _number(_at(raw, "rendering.background_value"), "background_value")
    if not (0 <= gray_low < gray_high <= 1):
        raise ConfigError("grayscale_range must satisfy 0 <= low < high <= 1")
    if not (gray_low <= foreground <= gray_high and gray_low <= background <= gray_high):
        raise ConfigError("foreground/background values must lie inside grayscale_range")
    if foreground != 1.0 or background != 0.0:
        raise ConfigError("MVP rendering requires white foreground (1.0) on black background (0.0)")
    if not _boolean(
        _at(raw, "rendering.preserve_aspect_ratio"), "preserve_aspect_ratio"
    ):
        raise ConfigError("rendering.preserve_aspect_ratio must be true")

    rows = _integer(_at(raw, "electrode_grid.rows"), "electrode_grid.rows", minimum=1)
    cols = _integer(_at(raw, "electrode_grid.cols"), "electrode_grid.cols", minimum=1)
    candidate_count = rows * cols
    expected_candidates = _integer(
        _at(raw, "validation.expected_candidate_count"), "expected_candidate_count", minimum=1
    )
    if candidate_count != expected_candidates:
        raise ConfigError("rows*cols does not equal expected_candidate_count")

    encoders = _list_of_strings(_at(raw, "experiment.encoders"), "experiment.encoders")
    allowed_encoders = {
        "raw_topk",
        "edge_topk",
        "skeleton_topk",
        "eb_sse",
        "eb_topk",
    }
    unknown_encoders = sorted(set(encoders) - allowed_encoders)
    if unknown_encoders:
        raise ConfigError(f"unknown encoder names: {unknown_encoders}")
    budgets = _list_of_ints(_at(raw, "experiment.budgets"), "experiment.budgets")
    if budgets != tuple(sorted(budgets)) or any(not 0 < budget <= candidate_count for budget in budgets):
        raise ConfigError("budgets must be increasing and within the candidate count")

    candidate_utility = _string(
        _at(raw, "protocol.candidate_utility"), "protocol.candidate_utility"
    ).lower()
    stimulation_amplitude = _string(
        _at(raw, "protocol.stimulation_amplitude"),
        "protocol.stimulation_amplitude",
    ).lower()
    topology_detection = _string(
        _at(raw, "protocol.topology_detection"), "protocol.topology_detection"
    ).lower()
    result_status = _string(
        _at(raw, "protocol.result_status"), "protocol.result_status"
    ).lower()

    mapping_names = _list_of_strings(_at(raw, "mapping.names"), "mapping.names")
    if set(mapping_names) - {"ideal", "moderate"}:
        raise ConfigError("mapping.names may contain only ideal and moderate")
    mapping_seeds = _list_of_ints(_at(raw, "mapping.seeds"), "mapping.seeds")
    if any(item < 0 for item in mapping_seeds):
        raise ConfigError("mapping.seeds must be non-negative")
    jitter = _number(
        _at(raw, "mapping.moderate.position_jitter_std_grid"), "position_jitter_std_grid"
    )
    sigma_std = _number(_at(raw, "mapping.moderate.sigma_scale_std"), "sigma_scale_std")
    sigma_low, sigma_high = _pair(
        _at(raw, "mapping.moderate.sigma_scale_clip"), "sigma_scale_clip", _number
    )
    if jitter < 0 or sigma_std < 0 or not 0 < sigma_low <= sigma_high:
        raise ConfigError("moderate mapping parameters are invalid")
    if not _boolean(
        _at(raw, "mapping.moderate.clip_centers_to_canvas"), "clip_centers_to_canvas"
    ):
        raise ConfigError("clip_centers_to_canvas must be true")

    if _number(_at(raw, "spv.base_sigma_grid"), "base_sigma_grid") <= 0:
        raise ConfigError("spv.base_sigma_grid must be positive")
    if _number(_at(raw, "spv.amplitude"), "amplitude") <= 0:
        raise ConfigError("spv.amplitude must be positive")
    required_spv = {"kernel_normalization": "l1", "output_normalization": "none", "output_dtype": "float32"}
    for key, expected in required_spv.items():
        if _string(_at(raw, f"spv.{key}"), f"spv.{key}").lower() != expected:
            raise ConfigError(f"spv.{key} must be {expected!r}")

    if _string(_at(raw, "feature.name"), "feature.name").lower() != "hog":
        raise ConfigError("feature.name must be 'hog'")
    orientations = _integer(_at(raw, "feature.orientations"), "orientations", minimum=1)
    pixels_per_cell = _pair(_at(raw, "feature.pixels_per_cell"), "pixels_per_cell", _integer)
    cells_per_block = _pair(_at(raw, "feature.cells_per_block"), "cells_per_block", _integer)
    if any(value <= 0 for value in (*pixels_per_cell, *cells_per_block)):
        raise ConfigError("HOG cell and block values must be positive")
    if _string(_at(raw, "feature.block_norm"), "block_norm") != "L2-Hys":
        raise ConfigError("feature.block_norm must be 'L2-Hys'")
    hog_length = calculate_hog_length(image_size, orientations, pixels_per_cell, cells_per_block)
    expected_hog = _integer(
        _at(raw, "validation.expected_hog_length"), "expected_hog_length", minimum=1
    )
    if hog_length != expected_hog:
        raise ConfigError(f"HOG geometry yields {hog_length} features, not {expected_hog}")

    if _string(_at(raw, "classifier.name"), "classifier.name").lower() != "linear_svc":
        raise ConfigError("classifier.name must be 'linear_svc'")
    if _number(_at(raw, "classifier.c"), "classifier.c") <= 0:
        raise ConfigError("classifier.c must be positive")
    if _string(_at(raw, "classifier.dual"), "classifier.dual").lower() != "auto":
        raise ConfigError("classifier.dual must be 'auto'")
    _integer(_at(raw, "classifier.max_iter"), "max_iter", minimum=1)
    if _integer(_at(raw, "classifier.random_state"), "random_state") != seed:
        raise ConfigError("classifier.random_state must equal runtime.seed")

    for key in ("save_spv", "save_selection", "cache_features", "overwrite"):
        _boolean(_at(raw, f"storage.{key}"), f"storage.{key}")
    tolerance = _number(
        _at(raw, "validation.energy_relative_tolerance"), "energy_relative_tolerance"
    )
    if not 0 < tolerance < 1:
        raise ConfigError("energy_relative_tolerance must be between 0 and 1")
    if not _boolean(
        _at(raw, "validation.require_unique_electrodes"), "require_unique_electrodes"
    ):
        raise ConfigError("require_unique_electrodes must be true")

    if stage in {"mvp", "stage4_5"}:
        frozen_checks = {
            "character_count": character_count == 20,
            "font split": (len(train_fonts), len(validation_fonts), len(test_fonts)) == (1, 0, 1),
            "encoders": encoders == ("raw_topk", "edge_topk"),
            "budgets": budgets == (64,),
            "mappings": mapping_names == ("ideal", "moderate"),
            "mapping_seeds": mapping_seeds == (2101,),
            "max_workers": max_workers == 1,
            "save_spv": _at(raw, "storage.save_spv") is True,
            "overwrite": _at(raw, "storage.overwrite") is False,
        }
        changed = [name for name, valid in frozen_checks.items() if not valid]
        if changed:
            raise ConfigError(f"MVP frozen constraints were changed: {changed}")
    if stage == "mvp":
        legacy_protocol = {
            "candidate_utility": candidate_utility == "bilinear_center_sampling",
            "stimulation_amplitude": stimulation_amplitude == "raw_glyph_sample",
            "topology_detection": topology_detection == "not_frozen",
            "result_status": result_status == "engineering_validation_only",
        }
        changed = [name for name, valid in legacy_protocol.items() if not valid]
        if changed:
            raise ConfigError(f"legacy MVP protocol markers were changed: {changed}")
    if stage == "stage4_5":
        fairness_protocol = {
            "candidate_utility": candidate_utility
            == "l1_ideal_gaussian_projection",
            "stimulation_amplitude": stimulation_amplitude
            == "unit_selected_electrode",
            "topology_detection": topology_detection == "crossing_number",
            "result_status": result_status == "fairness_repaired_mvp",
        }
        changed = [name for name, valid in fairness_protocol.items() if not valid]
        if changed:
            raise ConfigError(f"Stage 4.5 fairness protocol was changed: {changed}")
    if stage == "stage5":
        stage5_checks = {
            "character_count": character_count == 20,
            "font split": (len(train_fonts), len(validation_fonts), len(test_fonts))
            == (1, 0, 1),
            "encoders": encoders
            == ("raw_topk", "edge_topk", "skeleton_topk"),
            "budgets": budgets == (32, 64, 128),
            "mappings": mapping_names == ("ideal", "moderate"),
            "mapping_seeds": mapping_seeds == (2101,),
            "max_workers": max_workers == 1,
            "save_spv": _at(raw, "storage.save_spv") is True,
            "overwrite": _at(raw, "storage.overwrite") is False,
            "candidate_utility": candidate_utility
            == "l1_ideal_gaussian_projection",
            "stimulation_amplitude": stimulation_amplitude
            == "unit_selected_electrode",
            "topology_detection": topology_detection == "crossing_number",
            "result_status": result_status == "skeleton_topk_mvp",
        }
        changed = [name for name, valid in stage5_checks.items() if not valid]
        if changed:
            raise ConfigError(f"Stage 5 frozen protocol was changed: {changed}")
    if stage == "stage6":
        stage6_checks = {
            "character_count": character_count == 20,
            "font split": (len(train_fonts), len(validation_fonts), len(test_fonts))
            == (1, 0, 1),
            "encoders": encoders
            == ("raw_topk", "edge_topk", "skeleton_topk", "eb_topk"),
            "budgets": budgets == (32, 64, 128),
            "mappings": mapping_names == ("ideal", "moderate"),
            "mapping_seeds": mapping_seeds == (2101,),
            "max_workers": max_workers == 1,
            "save_spv": _at(raw, "storage.save_spv") is True,
            "overwrite": _at(raw, "storage.overwrite") is False,
            "candidate_utility": candidate_utility
            == "l1_ideal_gaussian_projection",
            "stimulation_amplitude": stimulation_amplitude
            == "unit_selected_electrode",
            "topology_detection": topology_detection == "crossing_number",
            "result_status": result_status == "eb_topk_mvp",
        }
        changed = [name for name, valid in stage6_checks.items() if not valid]
        if changed:
            raise ConfigError(f"Stage 6 frozen protocol was changed: {changed}")
    if stage == "stage7":
        stage7_checks = {
            "character_count": character_count == 20,
            "font split": (len(train_fonts), len(validation_fonts), len(test_fonts))
            == (1, 0, 1),
            "encoders": encoders
            == (
                "raw_topk",
                "edge_topk",
                "skeleton_topk",
                "eb_topk",
                "eb_sse",
            ),
            "budgets": budgets == (32, 64, 128),
            "mappings": mapping_names == ("ideal", "moderate"),
            "mapping_seeds": mapping_seeds == (2101,),
            "max_workers": max_workers == 1,
            "save_spv": _at(raw, "storage.save_spv") is True,
            "overwrite": _at(raw, "storage.overwrite") is False,
            "candidate_utility": candidate_utility
            == "l1_ideal_gaussian_projection",
            "stimulation_amplitude": stimulation_amplitude
            == "unit_selected_electrode",
            "topology_detection": topology_detection == "crossing_number",
            "result_status": result_status == "eb_sse_mvp",
        }
        changed = [name for name, valid in stage7_checks.items() if not valid]
        if changed:
            raise ConfigError(f"Stage 7 frozen protocol was changed: {changed}")
    if stage == "benchmark":
        benchmark_checks = {
            "character_count": character_count == 180,
            "font split": (len(train_fonts), len(validation_fonts), len(test_fonts))
            == (5, 1, 2),
            "encoders": encoders
            == (
                "raw_topk",
                "edge_topk",
                "skeleton_topk",
                "eb_topk",
                "eb_sse",
            ),
            "budgets": budgets == (32, 64, 128),
            "mappings": mapping_names == ("ideal", "moderate"),
            "mapping_seeds": mapping_seeds
            == (2101, 2026, 2027, 2028, 2029, 2030),
            "max_workers": max_workers == 1,
            "candidate_utility": candidate_utility
            == "l1_ideal_gaussian_projection",
            "stimulation_amplitude": stimulation_amplitude
            == "unit_selected_electrode",
            "topology_detection": topology_detection == "crossing_number",
            "result_status": result_status
            == "stage8_benchmark_frozen_no_accuracy",
        }
        changed = [name for name, valid in benchmark_checks.items() if not valid]
        if changed:
            raise ConfigError(f"Stage 8 benchmark constraints were changed: {changed}")
    return project_root


def load_config(path: str | Path) -> ExperimentConfig:
    """Load one YAML file, validate it, then return an immutable view."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigError(f"configuration file does not exist: {source_path}")
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to read YAML configuration: {exc}") from exc
    raw = _mapping(loaded, "root")
    project_root = _validate(raw, source_path)
    return ExperimentConfig(source_path, project_root, _freeze(raw))


def format_summary(config: ExperimentConfig) -> str:
    train = config.get("dataset.split.train_fonts")
    validation = config.get("dataset.split.validation_fonts")
    test = config.get("dataset.split.test_fonts")
    rows, cols = config.grid_shape
    return "\n".join(
        (
            f"[PASS] Config loaded: {config.source_path}",
            f"       project={config.get('project.name')}, stage={config.stage}, device={config.device}",
            (
                f"       dataset={config.get('dataset.character_count')} characters, "
                f"fonts(train/val/test)={len(train)}/{len(validation)}/{len(test)}"
            ),
            f"       grid={rows}x{cols} ({config.candidate_count} candidates), budgets={list(config.budgets)}",
            (
                f"       encoders={list(config.encoders)}, mappings={list(config.mapping_names)}, "
                f"mapping_seeds={list(config.mapping_seeds)}"
            ),
            f"       HOG feature length={config.hog_feature_length}",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one Hanzi-SPV YAML config.")
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.yaml"))
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"[FAIL] Config error: {exc}")
        return 1
    print(format_summary(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
