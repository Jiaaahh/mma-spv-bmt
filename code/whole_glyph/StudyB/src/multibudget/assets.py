"""Freeze blinded confirmation assets for MMA multi-budget Study B.

This module renders inputs and creates mapping geometry only.  It intentionally
contains no classifier, prediction, metric, or accuracy implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from fontTools.ttLib import TTFont
from PIL import ImageFont
import yaml

from src.deterministic_io import array_sha256, sha256_file, write_deterministic_npz
from src.grid import make_regular_grid
from src.mapping import make_ideal_mapping, make_moderate_mapping, mapping_statistics
from src.multibudget.protocol import (
    MultiBudgetProtocol,
    canonical_json_bytes,
    load_protocol,
)
from src.rendering import save_contact_sheet
from src.v2.mma_assets import (
    ConfirmationFont,
    _load_character_sequence,
    _mapping_arrays,
    _render_font,
)


class MultiBudgetAssetError(ValueError):
    """Raised when blinded Study B assets cannot be safely frozen."""


ASSET_SCHEMA = "mma-multibudget-assets-v1"
FONT_SPEC_SCHEMA = "mma-multibudget-font-spec-v1"
EXCLUSION_SCHEMA = "mma-multibudget-font-exclusions-v1"
SCHEMA_VERSION = 1
ASSET_STATUS = "FROZEN_INPUTS_ONLY_NO_PERFORMANCE_READ"
REQUIRED_EXCLUSION_SOURCES = ("training", "c1", "e1")
GLYPH_FILENAME = "glyphs_confirmation.npz"
MANIFEST_SIDECAR = "assets_manifest.sha256"


def _document(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise MultiBudgetAssetError(f"{name} does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise MultiBudgetAssetError(f"cannot read {name}: {path}") from error
    if not isinstance(value, Mapping):
        raise MultiBudgetAssetError(f"{name} root must be a mapping")
    return value


def _safe_id(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result or not all(char.isalnum() or char in "-_" for char in result):
        raise MultiBudgetAssetError(f"{name} must be a filesystem-safe identifier")
    return result


def _resolve_from(document_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MultiBudgetAssetError(f"{name} must be a non-empty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = document_path.parent / candidate
    return candidate.resolve()


def _sha256_text(value: Any, name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise MultiBudgetAssetError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _font_family(path: Path, face_index: int) -> str:
    try:
        family, _ = ImageFont.truetype(str(path), size=32, index=face_index).getname()
    except OSError as error:
        raise MultiBudgetAssetError(f"cannot inspect font: {path}") from error
    result = str(family).strip()
    if not result:
        raise MultiBudgetAssetError(f"font has no readable family name: {path}")
    return result


def _name_value(font: TTFont, name_ids: Sequence[int]) -> str:
    table = font["name"]
    for name_id in name_ids:
        candidates = [record for record in table.names if record.nameID == name_id]
        candidates.sort(key=lambda record: (record.platformID != 3, record.langID != 0x409))
        for record in candidates:
            try:
                value = record.toUnicode().strip()
            except UnicodeDecodeError:
                continue
            if value:
                return value
    return ""


def _fonttools_details(
    path: Path, face_index: int, characters: Sequence[str]
) -> dict[str, Any]:
    try:
        with TTFont(str(path), fontNumber=face_index, lazy=False) as font:
            cmap = font.getBestCmap() or {}
            missing = [character for character in characters if ord(character) not in cmap]
            family = _name_value(font, (16, 1))
            subfamily = _name_value(font, (17, 2))
            postscript = _name_value(font, (6,))
    except Exception as error:
        raise MultiBudgetAssetError(f"fontTools cannot inspect font: {path}") from error
    if not family or not subfamily or not postscript:
        raise MultiBudgetAssetError(f"font lacks required naming metadata: {path}")
    if missing:
        raise MultiBudgetAssetError(
            f"fontTools cmap lacks {len(missing)} required glyphs: " + "".join(missing[:10])
        )
    return {
        "family_id": family,
        "subfamily_id": subfamily,
        "postscript_name": postscript,
        "fonttools_cmap_covered_characters": len(characters),
        "fonttools_cmap_missing_characters": 0,
    }


def _load_exclusions(path: Path) -> dict[str, Any]:
    document = _document(path, "font exclusion list")
    if document.get("schema") != EXCLUSION_SCHEMA or document.get("schema_version") != 1:
        raise MultiBudgetAssetError("font exclusion list schema/version is invalid")
    sources = document.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(REQUIRED_EXCLUSION_SOURCES):
        raise MultiBudgetAssetError("exclusions must contain exactly training, c1, and e1")
    families: set[str] = set()
    hashes: set[str] = set()
    summaries: dict[str, Any] = {}
    for source_name in REQUIRED_EXCLUSION_SOURCES:
        source = sources[source_name]
        if not isinstance(source, Mapping):
            raise MultiBudgetAssetError(f"exclusions.sources.{source_name} must be a mapping")
        raw_families = source.get("family_ids")
        raw_hashes = source.get("file_sha256")
        if not isinstance(raw_families, list) or not raw_families:
            raise MultiBudgetAssetError(
                f"exclusions.sources.{source_name}.family_ids must be non-empty"
            )
        if not isinstance(raw_hashes, list) or not raw_hashes:
            raise MultiBudgetAssetError(
                f"exclusions.sources.{source_name}.file_sha256 must be non-empty"
            )
        source_families = tuple(str(item).strip().casefold() for item in raw_families)
        if any(not item for item in source_families):
            raise MultiBudgetAssetError(f"{source_name} contains an empty family ID")
        source_hashes = tuple(
            _sha256_text(item, f"{source_name}.file_sha256") for item in raw_hashes
        )
        families.update(source_families)
        hashes.update(source_hashes)
        summaries[source_name] = {
            "family_ids_casefold": sorted(set(source_families)),
            "file_sha256": sorted(set(source_hashes)),
        }
    return {
        "path": path.resolve(),
        "sha256": sha256_file(path),
        "sources": summaries,
        "family_ids_casefold": tuple(sorted(families)),
        "file_sha256": tuple(sorted(hashes)),
    }


def _load_font_spec(
    path: Path,
    protocol: MultiBudgetProtocol,
    exclusions: Mapping[str, Any],
    characters: Sequence[str],
) -> tuple[Path, tuple[ConfirmationFont, ...], tuple[dict[str, Any], ...]]:
    document = _document(path, "font specification")
    if document.get("schema") != FONT_SPEC_SCHEMA or document.get("schema_version") != 1:
        raise MultiBudgetAssetError("font specification schema/version is invalid")
    source_dataset = _resolve_from(
        path, document.get("source_dataset"), "font_spec.source_dataset"
    )
    if not source_dataset.is_file():
        raise MultiBudgetAssetError(f"source dataset does not exist: {source_dataset}")
    raw_fonts = document.get("fonts")
    if not isinstance(raw_fonts, list) or len(raw_fonts) != protocol.confirmation_font_family_count:
        raise MultiBudgetAssetError(
            "font specification must contain exactly four confirmation fonts"
        )
    excluded_families = set(exclusions["family_ids_casefold"])
    excluded_hashes = set(exclusions["file_sha256"])
    fonts: list[ConfirmationFont] = []
    details: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_fonts):
        if not isinstance(raw, Mapping):
            raise MultiBudgetAssetError(f"font specification item {index} must be a mapping")
        font_id = _safe_id(raw.get("font_id"), f"fonts[{index}].font_id")
        expected_family = str(raw.get("family_id", "")).strip()
        if not expected_family:
            raise MultiBudgetAssetError(f"fonts[{index}].family_id must not be empty")
        font_path = _resolve_from(path, raw.get("path"), f"fonts[{index}].path")
        if not font_path.is_file():
            raise MultiBudgetAssetError(f"font file does not exist: {font_path}")
        face_index = raw.get("face_index", 0)
        if isinstance(face_index, bool) or not isinstance(face_index, int) or face_index < 0:
            raise MultiBudgetAssetError(f"fonts[{index}].face_index is invalid")
        expected_hash = _sha256_text(
            raw.get("file_sha256"), f"fonts[{index}].file_sha256"
        )
        actual_hash = sha256_file(font_path)
        if actual_hash != expected_hash:
            raise MultiBudgetAssetError(f"font file SHA-256 mismatch: {font_id}")
        pillow_family = _font_family(font_path, face_index)
        names = _fonttools_details(font_path, face_index, characters)
        actual_family = names["family_id"]
        if names["subfamily_id"].casefold() != "regular":
            raise MultiBudgetAssetError(
                f"font subfamily must be Regular for {font_id}: "
                f"{names['subfamily_id']!r}"
            )
        if actual_family.casefold() != expected_family.casefold():
            raise MultiBudgetAssetError(
                f"font family mismatch for {font_id}: {actual_family!r}"
            )
        if pillow_family.casefold() != actual_family.casefold():
            raise MultiBudgetAssetError(
                f"Pillow/fontTools family mismatch for {font_id}: "
                f"{pillow_family!r} != {actual_family!r}"
            )
        if actual_family.casefold() in excluded_families:
            raise MultiBudgetAssetError(f"font family overlaps exclusion list: {actual_family}")
        if actual_hash in excluded_hashes:
            raise MultiBudgetAssetError(f"font file overlaps exclusion list: {font_id}")
        fonts.append(ConfirmationFont(font_id, font_path, face_index))
        details.append(
            {
                "font_id": font_id,
                "analyst_blinded_id": font_id,
                "family_id": actual_family,
                "family_id_casefold": actual_family.casefold(),
                "subfamily_id": names["subfamily_id"],
                "postscript_name": names["postscript_name"],
                "pillow_family_id": pillow_family,
                "source_path": str(font_path),
                "face_index": face_index,
                "file_sha256": actual_hash,
                "file_size_bytes": font_path.stat().st_size,
                "license_scope": "local_system_font_not_cleared_for_redistribution",
                "redistribution_permitted": False,
                "derived_asset_distribution": (
                    "LOCAL_RESEARCH_ONLY_PENDING_SEPARATE_RIGHTS_REVIEW"
                ),
                "fonttools_cmap_covered_characters": names[
                    "fonttools_cmap_covered_characters"
                ],
                "fonttools_cmap_missing_characters": names[
                    "fonttools_cmap_missing_characters"
                ],
                "performance_selection": False,
            }
        )
    ids = [item.font_id for item in fonts]
    families = [item["family_id_casefold"] for item in details]
    hashes = [item["file_sha256"] for item in details]
    if len(set(ids)) != len(ids):
        raise MultiBudgetAssetError("confirmation font IDs must be unique")
    expected_blinded_ids = [f"MB_F{index:02d}" for index in range(1, 5)]
    if ids != expected_blinded_ids:
        raise MultiBudgetAssetError(
            "formal font IDs must be the ordered blinded IDs MB_F01..MB_F04"
        )
    if len(set(families)) != len(families):
        raise MultiBudgetAssetError("confirmation font families must be unique")
    if len(set(hashes)) != len(hashes):
        raise MultiBudgetAssetError("confirmation font files must be unique")
    if hashes != sorted(hashes):
        raise MultiBudgetAssetError(
            "confirmation fonts must follow ascending font-file SHA-256 order"
        )
    return source_dataset, tuple(fonts), tuple(details)


def _font_spec_source_dataset(path: Path) -> Path:
    document = _document(path, "font specification")
    if document.get("schema") != FONT_SPEC_SCHEMA or document.get("schema_version") != 1:
        raise MultiBudgetAssetError("font specification schema/version is invalid")
    return _resolve_from(path, document.get("source_dataset"), "font_spec.source_dataset")


def _historical_mapping_hashes(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise MultiBudgetAssetError("at least one historical mapping manifest is required")
    unique_paths = tuple(dict.fromkeys(path.resolve() for path in paths))
    categories = {
        "file_sha256": set(),
        "centers_sha256": set(),
        "sigma_sha256": set(),
        "geometry_sha256": set(),
    }

    def visit(value: Any, in_mappings: bool = False) -> None:
        if isinstance(value, Mapping):
            now_in_mappings = in_mappings
            for key, child in value.items():
                child_in_mappings = now_in_mappings or key == "mappings"
                if child_in_mappings and key in categories and isinstance(child, str):
                    categories[key].add(_sha256_text(child, f"historical.{key}"))
                visit(child, child_in_mappings)
        elif isinstance(value, list):
            for child in value:
                visit(child, in_mappings)

    records: list[dict[str, str]] = []
    for path in unique_paths:
        document = _document(path, "historical mapping manifest")
        before = sum(len(values) for values in categories.values())
        visit(document)
        after = sum(len(values) for values in categories.values())
        if after == before:
            raise MultiBudgetAssetError(
                f"historical manifest contains no recognizable mapping hashes: {path}"
            )
        records.append({"path": str(path), "sha256": sha256_file(path)})
    return {
        "manifests": records,
        **{key: sorted(values) for key, values in categories.items()},
    }


def _mapping_geometry_sha256(centers: np.ndarray, sigma: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(array_sha256(centers)))
    digest.update(bytes.fromhex(array_sha256(sigma)))
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(manifest)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    sidecar = path.with_name(MANIFEST_SIDECAR)
    with sidecar.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(hashlib.sha256(payload).hexdigest() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _prepare_into_staging(
    *,
    protocol: MultiBudgetProtocol,
    font_spec_path: Path,
    exclusion_path: Path,
    output_dir: Path,
    include_contact_sheets: bool,
    historical_manifest_paths: Sequence[Path],
) -> dict[str, Any]:
    exclusions = _load_exclusions(exclusion_path)
    source_dataset = _font_spec_source_dataset(font_spec_path)
    characters = _load_character_sequence(source_dataset)
    source_dataset, fonts, specified_details = _load_font_spec(
        font_spec_path, protocol, exclusions, characters
    )
    if len(characters) != protocol.class_count:
        raise MultiBudgetAssetError("source dataset does not contain exactly 180 classes")
    historical = _historical_mapping_hashes(historical_manifest_paths)

    all_samples: list[Any] = []
    samples_by_font: dict[str, list[Any]] = {}
    font_details: list[dict[str, Any]] = []
    for font, specified in zip(fonts, specified_details, strict=True):
        samples, rendered = _render_font(font, characters)
        if len(samples) != protocol.class_count:
            raise MultiBudgetAssetError(f"font {font.font_id} does not cover all 180 classes")
        all_samples.extend(samples)
        samples_by_font[font.font_id] = samples
        font_details.append(specified | {
            "required_character_count": rendered["required_character_count"],
            "missing_character_count": rendered["missing_character_count"],
            "unique_raw_glyph_fingerprints": rendered[
                "unique_raw_glyph_fingerprints"
            ],
            "unique_normalized_glyphs": rendered["unique_normalized_glyphs"],
        })
    expected_samples = protocol.class_count * protocol.confirmation_font_family_count
    if len(all_samples) != expected_samples:
        raise MultiBudgetAssetError("glyph sample count is not 4 x 180")

    font_hash_by_id = {item["font_id"]: item["file_sha256"] for item in font_details}
    glyph_arrays = {
        "format_version": np.asarray([1], dtype=np.int64),
        "images": np.stack([item.image for item in all_samples]).astype(np.float32),
        "class_ids": np.asarray([item.class_id for item in all_samples], dtype=np.int64),
        "characters": np.asarray([item.character for item in all_samples], dtype=np.str_),
        "font_ids": np.asarray([item.font_id for item in all_samples], dtype=np.str_),
        "sample_ids": np.asarray([item.sample_id for item in all_samples], dtype=np.str_),
        "splits": np.asarray([item.split for item in all_samples], dtype=np.str_),
        "font_sha256": np.asarray(
            [font_hash_by_id[item.font_id] for item in all_samples], dtype=np.str_
        ),
    }
    glyph_path = output_dir / GLYPH_FILENAME
    glyph_sha256 = write_deterministic_npz(glyph_path, glyph_arrays)

    grid = make_regular_grid(rows=16, cols=16, image_shape=(96, 96))
    ideal = make_ideal_mapping(grid, base_sigma_grid=0.45)
    mapping_details: list[dict[str, Any]] = []
    geometry_hashes: set[str] = set()
    mapping_dir = output_dir / "mappings"
    for seed in protocol.mapping_seeds:
        mapping = make_moderate_mapping(
            grid,
            seed=seed,
            base_sigma_grid=0.45,
            position_jitter_std_grid=0.30,
            sigma_scale_std=0.20,
            sigma_scale_clip=(0.60, 1.60),
        )
        geometry_hash = _mapping_geometry_sha256(mapping.centers_xy, mapping.sigma_px)
        centers_hash = array_sha256(mapping.centers_xy)
        sigma_hash = array_sha256(mapping.sigma_px)
        if geometry_hash in geometry_hashes:
            raise MultiBudgetAssetError("mapping seeds produced duplicate geometry")
        if centers_hash in historical["centers_sha256"]:
            raise MultiBudgetAssetError(f"mapping seed {seed} reuses historical centers")
        if sigma_hash in historical["sigma_sha256"]:
            raise MultiBudgetAssetError(f"mapping seed {seed} reuses historical sigma")
        if geometry_hash in historical["geometry_sha256"]:
            raise MultiBudgetAssetError(f"mapping seed {seed} reuses historical geometry")
        geometry_hashes.add(geometry_hash)
        mapping_path = mapping_dir / f"moderate_seed_{seed}.npz"
        mapping_hash = write_deterministic_npz(
            mapping_path, _mapping_arrays(mapping, "multibudget_confirmation_unseen")
        )
        if mapping_hash in historical["file_sha256"]:
            raise MultiBudgetAssetError(f"mapping seed {seed} reuses a historical file")
        statistics = mapping_statistics(ideal, mapping)
        mapping_details.append(
            {
                "mapping_id": mapping.mapping_id,
                "seed": seed,
                "file": _relative(output_dir, mapping_path),
                "file_sha256": mapping_hash,
                "centers_sha256": centers_hash,
                "sigma_sha256": sigma_hash,
                "geometry_sha256": geometry_hash,
                "geometry_statistics": {
                    "mean_displacement_px": statistics.mean_displacement_px,
                    "max_displacement_px": statistics.max_displacement_px,
                    "mean_sigma_px": statistics.mean_sigma_px,
                    "min_sigma_px": statistics.min_sigma_px,
                    "max_sigma_px": statistics.max_sigma_px,
                },
            }
        )

    contact_sheets: list[dict[str, str]] = []
    if include_contact_sheets:
        for font in fonts:
            sheet_path = output_dir / f"contact_sheet_{font.font_id}.png"
            save_contact_sheet(samples_by_font[font.font_id], sheet_path, columns=10)
            contact_sheets.append(
                {
                    "font_id": font.font_id,
                    "file": _relative(output_dir, sheet_path),
                    "file_sha256": sha256_file(sheet_path),
                }
            )

    manifest: dict[str, Any] = {
        "schema": ASSET_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": ASSET_STATUS,
        "study_id": protocol.study_id,
        "protocol_config": {
            "path": str(protocol.source_path),
            "sha256": sha256_file(protocol.source_path),
        },
        "font_spec": {
            "path": str(font_spec_path.resolve()),
            "sha256": sha256_file(font_spec_path),
        },
        "selection_governance": {
            "font_ids_are_analyst_blinded": True,
            "font_order_rule": "eligible_font_file_sha256_ascending_then_lowest_face_index",
            "performance_selection": False,
            "font_binary_copied_into_project": False,
        },
        "rights_boundary": {
            "font_binaries_copied": False,
            "font_redistribution_permitted": False,
            "glyph_archive_distribution": (
                "LOCAL_RESEARCH_ONLY_PENDING_SEPARATE_RIGHTS_REVIEW"
            ),
            "public_reproduction_requirement": (
                "separately_preregister_four_new_redistributable_fonts"
            ),
        },
        "exclusions": {
            "path": str(exclusion_path.resolve()),
            "sha256": exclusions["sha256"],
            "required_sources": list(REQUIRED_EXCLUSION_SOURCES),
            "sources": exclusions["sources"],
            "family_ids_casefold": list(exclusions["family_ids_casefold"]),
            "file_sha256": list(exclusions["file_sha256"]),
            "all_selected_fonts_disjoint": True,
        },
        "source_character_archive": {
            "path": str(source_dataset),
            "sha256": sha256_file(source_dataset),
            "character_sequence_sha256": hashlib.sha256(
                "".join(characters).encode("utf-8")
            ).hexdigest(),
            "class_count": len(characters),
        },
        "glyph_archive": {
            "file": _relative(output_dir, glyph_path),
            "file_sha256": glyph_sha256,
            "sample_count": len(all_samples),
            "class_count": protocol.class_count,
            "font_count": protocol.confirmation_font_family_count,
            "images_sha256": array_sha256(glyph_arrays["images"]),
        },
        "fonts": font_details,
        "historical_mapping_exclusions": historical,
        "mapping_parameters": {
            "condition": protocol.condition,
            "seeds": list(protocol.mapping_seeds),
            "grid": [16, 16],
            "image_shape": [96, 96],
            "base_sigma_grid": 0.45,
            "position_jitter_std_grid": 0.30,
            "sigma_scale_std": 0.20,
            "sigma_scale_clip": [0.60, 1.60],
            "kernel_normalization": "l1_after_canvas_boundary",
        },
        "mappings": mapping_details,
        "contact_sheets": contact_sheets,
        "prediction_executed": False,
        "accuracy_generated": False,
        "accuracy_read": False,
    }
    _write_manifest(output_dir / "assets_manifest.json", manifest)
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MultiBudgetAssetError(f"cannot read asset manifest: {path}") from error
    if not isinstance(value, dict):
        raise MultiBudgetAssetError("asset manifest root must be an object")
    return value


def _verify_file(root: Path, record: Mapping[str, Any], name: str) -> Path:
    relative = record.get("file")
    if not isinstance(relative, str) or not relative:
        raise MultiBudgetAssetError(f"{name}.file is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise MultiBudgetAssetError(f"{name}.file escapes asset directory") from error
    if not path.is_file() or sha256_file(path) != record.get("file_sha256"):
        raise MultiBudgetAssetError(f"{name} file hash mismatch")
    return path


def validate_multibudget_assets(
    *,
    config_path: str | Path,
    font_spec_path: str | Path,
    exclusion_path: str | Path,
    historical_manifest_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Read-only validate the frozen manifest, inputs, hashes, and NPZ semantics."""

    protocol = load_protocol(config_path)
    manifest_path = protocol.asset_manifest_path
    root = manifest_path.parent.resolve()
    if not manifest_path.is_file():
        raise MultiBudgetAssetError(f"asset manifest does not exist: {manifest_path}")
    sidecar = root / MANIFEST_SIDECAR
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != sha256_file(
        manifest_path
    ):
        raise MultiBudgetAssetError("asset manifest SHA-256 sidecar mismatch")
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema") != ASSET_SCHEMA or manifest.get("schema_version") != 1:
        raise MultiBudgetAssetError("asset manifest schema/version is invalid")
    if manifest.get("status") != ASSET_STATUS:
        raise MultiBudgetAssetError("asset manifest is not frozen-inputs-only")
    for field in ("prediction_executed", "accuracy_generated", "accuracy_read"):
        if manifest.get(field) is not False:
            raise MultiBudgetAssetError(f"asset manifest {field} must be false")
    if manifest.get("study_id") != protocol.study_id:
        raise MultiBudgetAssetError("asset manifest study ID mismatch")
    if manifest.get("selection_governance") != {
        "font_ids_are_analyst_blinded": True,
        "font_order_rule": "eligible_font_file_sha256_ascending_then_lowest_face_index",
        "performance_selection": False,
        "font_binary_copied_into_project": False,
    }:
        raise MultiBudgetAssetError("asset selection governance contract mismatch")
    if manifest.get("rights_boundary") != {
        "font_binaries_copied": False,
        "font_redistribution_permitted": False,
        "glyph_archive_distribution": (
            "LOCAL_RESEARCH_ONLY_PENDING_SEPARATE_RIGHTS_REVIEW"
        ),
        "public_reproduction_requirement": (
            "separately_preregister_four_new_redistributable_fonts"
        ),
    }:
        raise MultiBudgetAssetError("asset rights boundary contract mismatch")
    if manifest.get("protocol_config", {}).get("sha256") != sha256_file(
        protocol.source_path
    ):
        raise MultiBudgetAssetError("protocol config hash mismatch")

    exclusions = _load_exclusions(Path(exclusion_path).resolve())
    source_dataset = _font_spec_source_dataset(Path(font_spec_path).resolve())
    characters = _load_character_sequence(source_dataset)
    source_dataset, fonts, specified = _load_font_spec(
        Path(font_spec_path).resolve(), protocol, exclusions, characters
    )
    if manifest.get("font_spec", {}).get("sha256") != sha256_file(font_spec_path):
        raise MultiBudgetAssetError("font specification hash mismatch")
    if manifest.get("exclusions", {}).get("sha256") != exclusions["sha256"]:
        raise MultiBudgetAssetError("font exclusion list hash mismatch")
    source_record = manifest.get("source_character_archive", {})
    if source_record.get("sha256") != sha256_file(source_dataset):
        raise MultiBudgetAssetError("source character archive hash mismatch")
    if source_record.get("character_sequence_sha256") != hashlib.sha256(
        "".join(characters).encode("utf-8")
    ).hexdigest():
        raise MultiBudgetAssetError("source character sequence hash mismatch")

    manifest_fonts = manifest.get("fonts")
    if not isinstance(manifest_fonts, list) or len(manifest_fonts) != len(fonts):
        raise MultiBudgetAssetError("manifest does not contain exactly four fonts")
    for actual, frozen in zip(specified, manifest_fonts, strict=True):
        for key in (
            "font_id",
            "analyst_blinded_id",
            "family_id",
            "family_id_casefold",
            "subfamily_id",
            "postscript_name",
            "pillow_family_id",
            "source_path",
            "face_index",
            "file_sha256",
            "file_size_bytes",
            "license_scope",
            "redistribution_permitted",
            "derived_asset_distribution",
            "fonttools_cmap_covered_characters",
            "fonttools_cmap_missing_characters",
            "performance_selection",
        ):
            if frozen.get(key) != actual[key]:
                raise MultiBudgetAssetError(f"frozen font field mismatch: {key}")
        if frozen.get("required_character_count") != protocol.class_count:
            raise MultiBudgetAssetError("font character coverage count mismatch")
        if frozen.get("missing_character_count") != 0:
            raise MultiBudgetAssetError("frozen font reports missing characters")
        if frozen.get("unique_raw_glyph_fingerprints") != protocol.class_count:
            raise MultiBudgetAssetError("font raw glyph fingerprints are not unique")
        if frozen.get("unique_normalized_glyphs") != protocol.class_count:
            raise MultiBudgetAssetError("font normalized glyphs are not unique")

    glyph_record = manifest.get("glyph_archive")
    if not isinstance(glyph_record, Mapping):
        raise MultiBudgetAssetError("glyph archive record is invalid")
    glyph_path = _verify_file(root, glyph_record, "glyph_archive")
    with np.load(glyph_path, allow_pickle=False) as archive:
        required = {
            "format_version",
            "images",
            "class_ids",
            "characters",
            "font_ids",
            "sample_ids",
            "splits",
            "font_sha256",
        }
        if set(archive.files) != required:
            raise MultiBudgetAssetError("glyph archive keys differ from frozen schema")
        images = np.asarray(archive["images"], dtype=np.float32)
        class_ids = np.asarray(archive["class_ids"], dtype=np.int64)
        archive_characters = np.asarray(archive["characters"], dtype=np.str_)
        font_ids = np.asarray(archive["font_ids"], dtype=np.str_)
        sample_ids = np.asarray(archive["sample_ids"], dtype=np.str_)
        splits = np.asarray(archive["splits"], dtype=np.str_)
        font_sha256 = np.asarray(archive["font_sha256"], dtype=np.str_)
        one_dimensional = (
            class_ids,
            archive_characters,
            font_ids,
            sample_ids,
            splits,
            font_sha256,
        )
        if images.shape != (720, 96, 96) or any(
            values.shape != (720,) for values in one_dimensional
        ):
            raise MultiBudgetAssetError("glyph archive does not have 4 x 180 samples")
        if not np.isfinite(images).all() or float(images.min()) < 0.0 or float(
            images.max()
        ) > 1.0:
            raise MultiBudgetAssetError("glyph archive images are numerically invalid")
        if len(set(map(str, sample_ids))) != 720:
            raise MultiBudgetAssetError("glyph sample IDs are not unique")
        if set(map(str, splits)) != {"test"}:
            raise MultiBudgetAssetError("glyph archive split must be test-only")
        if glyph_record.get("sample_count") != 720:
            raise MultiBudgetAssetError("glyph manifest sample count mismatch")
        if glyph_record.get("class_count") != 180:
            raise MultiBudgetAssetError("glyph manifest class count mismatch")
        if glyph_record.get("font_count") != 4:
            raise MultiBudgetAssetError("glyph manifest font count mismatch")
        if glyph_record.get("images_sha256") != array_sha256(images):
            raise MultiBudgetAssetError("glyph image array hash mismatch")
        expected_characters = np.asarray(characters, dtype=np.str_)
        hash_by_id = {
            item["font_id"]: item["file_sha256"] for item in specified
        }
        for font in fonts:
            mask = font_ids == font.font_id
            selected = class_ids[mask]
            if not np.array_equal(selected, np.arange(180, dtype=np.int64)):
                raise MultiBudgetAssetError(
                    f"glyph archive lacks exact class coverage for {font.font_id}"
                )
            if not np.array_equal(archive_characters[mask], expected_characters):
                raise MultiBudgetAssetError(
                    f"glyph character order differs for {font.font_id}"
                )
            if set(map(str, font_sha256[mask])) != {hash_by_id[font.font_id]}:
                raise MultiBudgetAssetError(
                    f"glyph font hash differs for {font.font_id}"
                )

    mappings = manifest.get("mappings")
    if not isinstance(mappings, list) or [item.get("seed") for item in mappings] != list(
        protocol.mapping_seeds
    ):
        raise MultiBudgetAssetError("mapping records differ from protocol seeds")
    historical = _historical_mapping_hashes(
        tuple(Path(path).resolve() for path in historical_manifest_paths)
    )
    frozen_historical = manifest.get("historical_mapping_exclusions")
    if frozen_historical != historical:
        raise MultiBudgetAssetError("historical mapping exclusion state mismatch")
    geometry_hashes: set[str] = set()
    for index, record in enumerate(mappings):
        path = _verify_file(root, record, f"mappings[{index}]")
        with np.load(path, allow_pickle=False) as archive:
            centers = np.asarray(archive["centers_xy"], dtype=np.float32)
            sigma = np.asarray(archive["sigma_px"], dtype=np.float32)
            seed = int(np.asarray(archive["seed"], dtype=np.int64)[0])
        if seed != protocol.mapping_seeds[index]:
            raise MultiBudgetAssetError("mapping archive seed mismatch")
        geometry_hash = _mapping_geometry_sha256(centers, sigma)
        if record.get("centers_sha256") != array_sha256(centers):
            raise MultiBudgetAssetError("mapping center hash mismatch")
        if record.get("sigma_sha256") != array_sha256(sigma):
            raise MultiBudgetAssetError("mapping sigma hash mismatch")
        if record.get("geometry_sha256") != geometry_hash:
            raise MultiBudgetAssetError("mapping geometry hash mismatch")
        if record.get("file_sha256") in historical["file_sha256"]:
            raise MultiBudgetAssetError("mapping file overlaps historical manifest")
        if record.get("centers_sha256") in historical["centers_sha256"]:
            raise MultiBudgetAssetError("mapping centers overlap historical manifest")
        if record.get("sigma_sha256") in historical["sigma_sha256"]:
            raise MultiBudgetAssetError("mapping sigma overlaps historical manifest")
        if geometry_hash in historical["geometry_sha256"]:
            raise MultiBudgetAssetError("mapping geometry overlaps historical manifest")
        geometry_hashes.add(geometry_hash)
    if len(geometry_hashes) != len(protocol.mapping_seeds):
        raise MultiBudgetAssetError("mapping geometries are not unique")
    for index, record in enumerate(manifest.get("contact_sheets", [])):
        _verify_file(root, record, f"contact_sheets[{index}]")
    return {
        "status": "PASS_FROZEN_MULTIBUDGET_ASSETS_VALID",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "font_count": len(fonts),
        "class_count": protocol.class_count,
        "sample_count": 720,
        "mapping_seeds": list(protocol.mapping_seeds),
        "prediction_executed": False,
        "accuracy_read": False,
        "validation_mode": "READ_ONLY",
    }


def prepare_multibudget_assets(
    *,
    config_path: str | Path,
    font_spec_path: str | Path,
    exclusion_path: str | Path,
    historical_manifest_paths: Sequence[str | Path],
    include_contact_sheets: bool = False,
) -> dict[str, Any]:
    """Generate the complete asset tree in staging, then atomically freeze it."""

    protocol = load_protocol(config_path)
    final_dir = protocol.asset_manifest_path.parent.resolve()
    if final_dir.exists():
        raise MultiBudgetAssetError(
            f"asset directory already exists; refusing to overwrite: {final_dir}"
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = final_dir.with_name(f".{final_dir.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        manifest = _prepare_into_staging(
            protocol=protocol,
            font_spec_path=Path(font_spec_path).resolve(),
            exclusion_path=Path(exclusion_path).resolve(),
            output_dir=staging,
            include_contact_sheets=include_contact_sheets,
            historical_manifest_paths=tuple(
                Path(path).resolve() for path in historical_manifest_paths
            ),
        )
        if final_dir.exists():
            raise MultiBudgetAssetError(
                f"asset directory appeared during generation: {final_dir}"
            )
        staging.replace(final_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return dict(manifest) | {
        "manifest_path": str(protocol.asset_manifest_path),
        "manifest_sha256": sha256_file(protocol.asset_manifest_path),
    }


__all__ = [
    "ASSET_SCHEMA",
    "ASSET_STATUS",
    "EXCLUSION_SCHEMA",
    "FONT_SPEC_SCHEMA",
    "MultiBudgetAssetError",
    "prepare_multibudget_assets",
    "validate_multibudget_assets",
]
