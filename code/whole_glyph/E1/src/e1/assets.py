"""Prepare the frozen E1 glyph and phosphene-mapping input assets.

This module deliberately stops before feature extraction or prediction.  It
does not import a classifier and it never computes or reads an accuracy.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from fontTools.ttLib import TTFont

from src.contracts import ElectrodeMapping, GlyphSample
from src.deterministic_io import array_sha256, sha256_file, write_deterministic_npz
from src.grid import RegularGrid, make_regular_grid
from src.rendering import render_glyph, save_contact_sheet
from src.spv_forward import GaussianPhospheneForwardModel


class E1AssetError(ValueError):
    """Raised when an E1 input asset is missing, unsafe, or malformed."""


@dataclass(frozen=True, slots=True)
class MappingCondition:
    name: str
    position_jitter_std_grid: float
    sigma_scale_std: float
    sigma_scale_clip: tuple[float, float]


CONDITIONS: tuple[MappingCondition, ...] = (
    MappingCondition("mild", 0.15, 0.10, (0.80, 1.30)),
    MappingCondition("moderate", 0.30, 0.20, (0.60, 1.60)),
    MappingCondition("severe", 0.45, 0.30, (0.40, 1.90)),
    MappingCondition("position_only", 0.30, 0.00, (1.00, 1.00)),
    MappingCondition("scale_only", 0.00, 0.20, (0.60, 1.60)),
)

DEFAULT_FONT_PATHS: tuple[Path, ...] = (
    Path(r"/historical/Windows\Fonts\simhei.ttf"),
    Path(r"/historical/Windows\Fonts\simkai.ttf"),
    Path(r"/historical/Windows\Fonts\simfang.ttf"),
    Path(r"/historical/Windows\Fonts\SIMYOU.TTF"),
)
DEFAULT_MAPPING_SEEDS: tuple[int, ...] = (3101, 3102, 3103, 3104, 3105)


@dataclass(frozen=True, slots=True)
class AnonymousFont:
    font_id: str
    path: Path
    file_sha256: str
    face_index: int = 0


def _atomic_json(path: Path, document: Any) -> str:
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f"{path.name}.write_tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def load_character_sequence(source_dataset: Path) -> tuple[str, ...]:
    """Recover the unique character assigned to every source class 0..179."""

    source_dataset = Path(source_dataset)
    if not source_dataset.is_file():
        raise E1AssetError(f"source glyph archive does not exist: {source_dataset}")
    with np.load(source_dataset, allow_pickle=False) as archive:
        if not {"class_ids", "characters"} <= set(archive.files):
            raise E1AssetError("source glyph archive lacks class_ids/characters")
        class_ids = np.asarray(archive["class_ids"], dtype=np.int64)
        characters = np.asarray(archive["characters"], dtype=np.str_)
    if class_ids.ndim != 1 or characters.shape != class_ids.shape:
        raise E1AssetError("source class_ids and characters must be aligned vectors")
    if not np.array_equal(np.unique(class_ids), np.arange(180, dtype=np.int64)):
        raise E1AssetError("E1 requires exactly the source classes 0..179")
    ordered: list[str] = []
    for class_id in range(180):
        values = np.unique(characters[class_ids == class_id])
        if values.shape != (1,) or len(str(values[0])) != 1:
            raise E1AssetError(f"class {class_id} does not map to one character")
        ordered.append(str(values[0]))
    if len(set(ordered)) != 180:
        raise E1AssetError("source character sequence contains duplicates")
    return tuple(ordered)


def _font_cmap(font_path: Path, face_index: int = 0) -> dict[int, str]:
    try:
        with TTFont(
            str(font_path), fontNumber=face_index, lazy=True, recalcBBoxes=False
        ) as font:
            cmap = font.getBestCmap() or {}
    except Exception as exc:  # fontTools exposes several parser exception classes.
        raise E1AssetError(f"fontTools could not read {font_path}: {exc}") from exc
    return dict(cmap)


def resolve_anonymous_fonts(
    font_paths: Sequence[Path], characters: Sequence[str]
) -> tuple[AnonymousFont, ...]:
    """Validate cmap coverage and assign IDs by sorted font-file SHA-256."""

    if len(font_paths) != 4:
        raise E1AssetError("the frozen E1 protocol requires exactly four fonts")
    if len(characters) != 180 or len(set(characters)) != 180:
        raise E1AssetError("font validation requires 180 unique characters")
    candidates: list[tuple[str, Path]] = []
    for raw_path in font_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise E1AssetError(f"font file does not exist: {path}")
        file_sha = sha256_file(path)
        cmap = _font_cmap(path)
        missing = [character for character in characters if ord(character) not in cmap]
        notdef = [
            character
            for character in characters
            if str(cmap.get(ord(character), "")).lower() == ".notdef"
        ]
        if missing or notdef:
            examples = "".join((missing + notdef)[:10])
            raise E1AssetError(
                f"font {path} lacks {len(missing) + len(notdef)} required cmap "
                f"entries (examples: {examples})"
            )
        candidates.append((file_sha, path))
    if len({file_sha for file_sha, _ in candidates}) != 4:
        raise E1AssetError("font files must have four distinct SHA-256 values")
    candidates.sort(key=lambda item: (item[0], str(item[1]).casefold()))
    return tuple(
        AnonymousFont(
            font_id=f"E1_F{index:02d}",
            path=path,
            file_sha256=file_sha,
        )
        for index, (file_sha, path) in enumerate(candidates, start=1)
    )


def _foreground_bbox_and_margins(image: np.ndarray) -> dict[str, int]:
    coordinates = np.argwhere(image > 0.0)
    if coordinates.size == 0:
        raise E1AssetError("rendered image contains no foreground")
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)
    height, width = image.shape
    margins = {
        "top": int(top),
        "left": int(left),
        "bottom": int(height - 1 - bottom),
        "right": int(width - 1 - right),
    }
    if min(margins.values()) < 8:
        raise E1AssetError(f"normalized glyph violates the 8-pixel margin: {margins}")
    return margins


def _render_font(
    font: AnonymousFont, characters: Sequence[str]
) -> tuple[list[GlyphSample], dict[str, Any]]:
    samples: list[GlyphSample] = []
    image_hashes: list[str] = []
    minimum_margin = 96
    maximum_foreground = 0
    for class_id, character in enumerate(characters):
        image = render_glyph(
            character,
            font.path,
            face_index=font.face_index,
            image_size=(96, 96),
            render_canvas_size=(192, 192),
            font_size=144,
            content_margin=8,
            foreground_value=1.0,
            background_value=0.0,
        )
        if image.dtype != np.float32 or image.shape != (96, 96):
            raise E1AssetError("render_glyph violated the E1 float32 96x96 contract")
        if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise E1AssetError("rendered glyph has invalid numeric values")
        margins = _foreground_bbox_and_margins(image)
        minimum_margin = min(minimum_margin, *margins.values())
        maximum_foreground = max(maximum_foreground, int(np.count_nonzero(image > 0.0)))
        image_hashes.append(array_sha256(image))
        samples.append(
            GlyphSample(
                sample_id=f"{font.font_id}_c{class_id:03d}",
                character=character,
                class_id=class_id,
                font_id=font.font_id,
                split="test",
                image=np.ascontiguousarray(image, dtype=np.float32),
            )
        )
    if len(samples) != 180 or len(set(image_hashes)) != 180:
        raise E1AssetError(
            f"font {font.font_id} did not produce 180 non-empty unique glyphs"
        )
    stacked = np.stack([sample.image for sample in samples]).astype(np.float32)
    return samples, {
        "font_id": font.font_id,
        "source_path": str(font.path),
        "source_filename": font.path.name,
        "face_index": font.face_index,
        "file_sha256": font.file_sha256,
        "file_size_bytes": font.path.stat().st_size,
        "fonttools_cmap_coverage": "180/180",
        "missing_character_count": 0,
        "nonempty_glyph_count": 180,
        "unique_normalized_glyph_count": 180,
        "minimum_observed_margin_px": minimum_margin,
        "maximum_foreground_pixel_count": maximum_foreground,
        "image_stack_sha256": array_sha256(stacked),
        "license_scope": "local_system_font_not_cleared_for_redistribution",
        "redistribution_permitted": False,
    }


def make_shared_latents(seed: int, candidate_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one frozen position/scale latent pair reused by all conditions."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise E1AssetError("mapping seed must be a positive integer")
    if isinstance(candidate_count, bool) or candidate_count <= 0:
        raise E1AssetError("candidate_count must be positive")
    rng = np.random.default_rng(seed)
    position_z = np.ascontiguousarray(
        rng.standard_normal((candidate_count, 2)), dtype=np.float32
    )
    sigma_z = np.ascontiguousarray(rng.standard_normal(candidate_count), dtype=np.float32)
    if not np.isfinite(position_z).all() or not np.isfinite(sigma_z).all():
        raise E1AssetError("mapping latent contains non-finite values")
    return position_z, sigma_z


def mapping_from_shared_latents(
    grid: RegularGrid,
    *,
    condition: MappingCondition,
    seed: int,
    position_z: np.ndarray,
    sigma_z: np.ndarray,
    base_sigma_grid: float = 0.45,
) -> ElectrodeMapping:
    """Construct one E1 mapping without drawing any additional randomness."""

    expected_positions = (grid.candidate_count, 2)
    if position_z.dtype != np.float32 or position_z.shape != expected_positions:
        raise E1AssetError(f"position_z must be float32 with shape {expected_positions}")
    if sigma_z.dtype != np.float32 or sigma_z.shape != (grid.candidate_count,):
        raise E1AssetError(
            f"sigma_z must be float32 with shape {(grid.candidate_count,)}"
        )
    spacing_x, spacing_y = grid.spacing_xy
    centers = np.array(grid.centers_xy, dtype=np.float32, copy=True)
    centers[:, 0] += np.float32(condition.position_jitter_std_grid * spacing_x) * position_z[:, 0]
    centers[:, 1] += np.float32(condition.position_jitter_std_grid * spacing_y) * position_z[:, 1]
    height, width = grid.image_shape
    centers[:, 0] = np.clip(centers[:, 0], 0.0, float(width - 1))
    centers[:, 1] = np.clip(centers[:, 1], 0.0, float(height - 1))

    base_sigma_px = float(base_sigma_grid) * float(np.sqrt(spacing_x * spacing_y))
    scales = np.float32(1.0) + np.float32(condition.sigma_scale_std) * sigma_z
    scales = np.clip(scales, *condition.sigma_scale_clip)
    sigma_px = np.ascontiguousarray(np.float32(base_sigma_px) * scales, dtype=np.float32)
    mapping = ElectrodeMapping(
        mapping_id=f"e1_{condition.name}_seed{seed}",
        seed=seed,
        centers_xy=np.ascontiguousarray(centers, dtype=np.float32),
        sigma_px=sigma_px,
        image_shape=grid.image_shape,
    )
    if condition.position_jitter_std_grid == 0.0 and not np.array_equal(
        mapping.centers_xy, grid.centers_xy
    ):
        raise E1AssetError("zero-position condition changed electrode centers")
    if condition.sigma_scale_std == 0.0 and not np.allclose(
        mapping.sigma_px, base_sigma_px, rtol=0.0, atol=1e-6
    ):
        raise E1AssetError("zero-scale condition changed phosphene sigma")
    return mapping


def _mapping_arrays(
    mapping: ElectrodeMapping,
    condition: MappingCondition,
    position_z: np.ndarray,
    sigma_z: np.ndarray,
    kernel_bank_sha256: str,
) -> dict[str, np.ndarray]:
    return {
        "format_version": np.asarray([1], dtype=np.int64),
        "mapping_id": np.asarray([mapping.mapping_id], dtype=np.str_),
        "condition": np.asarray([condition.name], dtype=np.str_),
        "seed": np.asarray([mapping.seed], dtype=np.int64),
        "image_shape": np.asarray(mapping.image_shape, dtype=np.int64),
        "centers_xy": np.asarray(mapping.centers_xy, dtype=np.float32),
        "sigma_px": np.asarray(mapping.sigma_px, dtype=np.float32),
        "base_sigma_grid": np.asarray([0.45], dtype=np.float64),
        "position_jitter_std_grid": np.asarray(
            [condition.position_jitter_std_grid], dtype=np.float64
        ),
        "sigma_scale_std": np.asarray([condition.sigma_scale_std], dtype=np.float64),
        "sigma_scale_clip": np.asarray(condition.sigma_scale_clip, dtype=np.float64),
        "position_latent_sha256": np.asarray([array_sha256(position_z)], dtype=np.str_),
        "sigma_latent_sha256": np.asarray([array_sha256(sigma_z)], dtype=np.str_),
        "kernel_bank_sha256": np.asarray([kernel_bank_sha256], dtype=np.str_),
        "kernel_normalization": np.asarray(["l1_after_canvas_boundary"], dtype=np.str_),
        "prediction_executed": np.asarray([False], dtype=np.bool_),
        "accuracy_read": np.asarray([False], dtype=np.bool_),
    }


def _require_safe_force_target(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        return
    if not force:
        raise E1AssetError(f"E1 assets already exist; refusing overwrite: {output_dir}")
    manifest_path = output_dir / "manifest.json"
    if not output_dir.is_dir() or not manifest_path.is_file():
        raise E1AssetError("--force requires an existing E1 directory with manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E1AssetError("existing E1 manifest is unreadable; refusing --force") from exc
    if manifest.get("prediction_executed") is not False or manifest.get("accuracy_read") is not False:
        raise E1AssetError(
            "existing E1 assets have prediction/accuracy exposure; refusing --force"
        )


def _publish_staged_directory(staging: Path, output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        staging.replace(output_dir)
        return
    if not force:
        raise E1AssetError("internal error: existing output reached publication")
    backup = output_dir.with_name(f".{output_dir.name}.safe_backup")
    if backup.exists():
        raise E1AssetError(f"stale safety backup blocks publication: {backup}")
    output_dir.replace(backup)
    try:
        staging.replace(output_dir)
    except Exception:
        backup.replace(output_dir)
        raise
    shutil.rmtree(backup)


def _all_asset_file_records(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "file": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]


def prepare_e1_assets(
    *,
    source_dataset: Path,
    output_dir: Path,
    font_paths: Sequence[Path] = DEFAULT_FONT_PATHS,
    mapping_seeds: Sequence[int] = DEFAULT_MAPPING_SEEDS,
    force: bool = False,
) -> dict[str, Any]:
    """Create and atomically publish the input-only E1 asset bundle."""

    source_dataset = Path(source_dataset).resolve()
    output_dir = Path(output_dir).resolve()
    seeds = tuple(int(seed) for seed in mapping_seeds)
    if seeds != DEFAULT_MAPPING_SEEDS:
        raise E1AssetError(f"frozen E1 mapping seeds must be {DEFAULT_MAPPING_SEEDS}")
    _require_safe_force_target(output_dir, force)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        characters = load_character_sequence(source_dataset)
        fonts = resolve_anonymous_fonts(font_paths, characters)

        glyph_samples: list[GlyphSample] = []
        samples_by_font: dict[str, list[GlyphSample]] = {}
        font_records: list[dict[str, Any]] = []
        for font in fonts:
            samples, record = _render_font(font, characters)
            glyph_samples.extend(samples)
            samples_by_font[font.font_id] = samples
            font_records.append(record)
        if len(glyph_samples) != 720:
            raise E1AssetError("glyph archive must contain four fonts x 180 classes")
        if len({record["image_stack_sha256"] for record in font_records}) != 4:
            raise E1AssetError("two E1 font archives are completely identical")

        images = np.stack([sample.image for sample in glyph_samples]).astype(np.float32)
        glyph_arrays = {
            "format_version": np.asarray([1], dtype=np.int64),
            "images": images,
            "class_ids": np.asarray(
                [sample.class_id for sample in glyph_samples], dtype=np.int64
            ),
            "sample_ids": np.asarray(
                [sample.sample_id for sample in glyph_samples], dtype=np.str_
            ),
            "characters": np.asarray(
                [sample.character for sample in glyph_samples], dtype=np.str_
            ),
            "font_ids": np.asarray(
                [sample.font_id for sample in glyph_samples], dtype=np.str_
            ),
            "splits": np.asarray(["test"] * len(glyph_samples), dtype=np.str_),
            "font_sha256": np.asarray(
                [
                    fonts[int(sample.font_id[-2:]) - 1].file_sha256
                    for sample in glyph_samples
                ],
                dtype=np.str_,
            ),
        }
        glyph_path = staging / "glyphs_e1_fonts.npz"
        glyph_file_sha = write_deterministic_npz(glyph_path, glyph_arrays)

        contact_sheet_paths: list[Path] = []
        for font in fonts:
            contact_path = staging / f"contact_sheet_{font.font_id}.png"
            save_contact_sheet(samples_by_font[font.font_id], contact_path, columns=10)
            contact_sheet_paths.append(contact_path)

        grid = make_regular_grid(rows=16, cols=16, image_shape=(96, 96))
        latent_paths: list[Path] = []
        mapping_paths: list[Path] = []
        mapping_records: list[dict[str, Any]] = []
        latent_records: list[dict[str, Any]] = []
        for seed in seeds:
            position_z, sigma_z = make_shared_latents(seed, grid.candidate_count)
            latent_path = staging / "latents" / f"seed{seed}.npz"
            latent_sha = write_deterministic_npz(
                latent_path,
                {
                    "format_version": np.asarray([1], dtype=np.int64),
                    "seed": np.asarray([seed], dtype=np.int64),
                    "position_z": position_z,
                    "sigma_z": sigma_z,
                    "prediction_executed": np.asarray([False], dtype=np.bool_),
                    "accuracy_read": np.asarray([False], dtype=np.bool_),
                },
            )
            latent_paths.append(latent_path)
            latent_records.append(
                {
                    "seed": seed,
                    "file": latent_path.relative_to(staging).as_posix(),
                    "file_sha256": latent_sha,
                    "position_z_sha256": array_sha256(position_z),
                    "sigma_z_sha256": array_sha256(sigma_z),
                    "shared_by_conditions": [item.name for item in CONDITIONS],
                }
            )
            for condition in CONDITIONS:
                mapping = mapping_from_shared_latents(
                    grid,
                    condition=condition,
                    seed=seed,
                    position_z=position_z,
                    sigma_z=sigma_z,
                )
                forward_model = GaussianPhospheneForwardModel(mapping)
                kernel_sums = forward_model.kernel_bank.sum(axis=(1, 2), dtype=np.float64)
                if not np.allclose(kernel_sums, 1.0, rtol=0.0, atol=1e-5):
                    raise E1AssetError("mapping failed phosphene-kernel L1 QC")
                kernel_hash = array_sha256(forward_model.kernel_bank)
                path = staging / "mappings" / f"{mapping.mapping_id}.npz"
                file_sha = write_deterministic_npz(
                    path,
                    _mapping_arrays(
                        mapping, condition, position_z, sigma_z, kernel_hash
                    ),
                )
                mapping_paths.append(path)
                displacement = np.linalg.norm(
                    mapping.centers_xy.astype(np.float64)
                    - grid.centers_xy.astype(np.float64),
                    axis=1,
                )
                mapping_records.append(
                    {
                        "mapping_id": mapping.mapping_id,
                        "condition": condition.name,
                        "seed": seed,
                        "file": path.relative_to(staging).as_posix(),
                        "file_sha256": file_sha,
                        "centers_xy_sha256": array_sha256(mapping.centers_xy),
                        "sigma_px_sha256": array_sha256(mapping.sigma_px),
                        "kernel_bank_sha256": kernel_hash,
                        "mean_displacement_px": float(displacement.mean()),
                        "max_displacement_px": float(displacement.max()),
                        "mean_sigma_px": float(mapping.sigma_px.mean(dtype=np.float64)),
                        "min_sigma_px": float(mapping.sigma_px.min()),
                        "max_sigma_px": float(mapping.sigma_px.max()),
                        "kernel_l1_qc": "PASS_256_OF_256",
                    }
                )

        asset_paths = [glyph_path, *contact_sheet_paths, *latent_paths, *mapping_paths]
        manifest = {
            "schema_version": 1,
            "status": "FROZEN_E1_INPUTS_ONLY_NO_PERFORMANCE_READ",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "source_character_archive": {
                "path": str(source_dataset),
                "sha256": sha256_file(source_dataset),
                "character_count": 180,
                "character_sequence_sha256": hashlib.sha256(
                    "".join(characters).encode("utf-8")
                ).hexdigest(),
            },
            "font_anonymization": "IDs assigned after ascending font-file SHA-256 sort",
            "font_license_notice": (
                "These are local operating-system fonts. Their binaries and the "
                "derived glyph archive are not cleared for redistribution and must "
                "remain local unless separate permission is established."
            ),
            "fonts": font_records,
            "rendering": {
                "image_shape": [96, 96],
                "dtype": "float32",
                "range": [0.0, 1.0],
                "render_canvas_shape": [192, 192],
                "font_size": 144,
                "content_margin_px": 8,
                "within_font_unique": "180/180 for every font",
                "cross_archive_stack_hashes_unique": True,
            },
            "glyph_archive": {
                "file": glyph_path.relative_to(staging).as_posix(),
                "file_sha256": glyph_file_sha,
                "images_sha256": array_sha256(images),
                "sample_count": 720,
                "class_count": 180,
                "font_count": 4,
            },
            "mapping_protocol": {
                "grid": [16, 16],
                "image_shape": [96, 96],
                "candidate_count": 256,
                "base_sigma_grid": 0.45,
                "seeds": list(seeds),
                "shared_latent_pairing": True,
                "conditions": [
                    {
                        "name": item.name,
                        "position_jitter_std_grid": item.position_jitter_std_grid,
                        "sigma_scale_std": item.sigma_scale_std,
                        "sigma_scale_clip": list(item.sigma_scale_clip),
                    }
                    for item in CONDITIONS
                ],
            },
            "latents": latent_records,
            "mappings": mapping_records,
            "asset_files": _all_asset_file_records(staging, asset_paths),
            "prediction_executed": False,
            "accuracy_read": False,
        }
        manifest_path = staging / "manifest.json"
        manifest_sha = _atomic_json(manifest_path, manifest)
        _publish_staged_directory(staging, output_dir, force)
        staging = Path()
        return {
            **manifest,
            "manifest_path": str(output_dir / "manifest.json"),
            "manifest_sha256": manifest_sha,
        }
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)


def load_e1_mapping(path: Path) -> ElectrodeMapping:
    """Load a frozen E1 mapping while enforcing the minimal portable schema."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"centers_xy", "sigma_px", "image_shape", "mapping_id", "seed"}
        if not required <= set(archive.files):
            raise E1AssetError(f"mapping archive lacks required fields: {path}")
        centers = np.asarray(archive["centers_xy"], dtype=np.float32)
        sigma = np.asarray(archive["sigma_px"], dtype=np.float32)
        image_shape_array = np.asarray(archive["image_shape"], dtype=np.int64)
        mapping_id_values = np.asarray(archive["mapping_id"], dtype=np.str_).reshape(-1)
        seed_values = np.asarray(archive["seed"], dtype=np.int64).reshape(-1)
    if image_shape_array.shape != (2,) or mapping_id_values.shape != (1,) or seed_values.shape != (1,):
        raise E1AssetError("mapping scalar/schema fields have invalid shapes")
    return ElectrodeMapping(
        mapping_id=str(mapping_id_values[0]),
        seed=int(seed_values[0]),
        centers_xy=np.ascontiguousarray(centers, dtype=np.float32),
        sigma_px=np.ascontiguousarray(sigma, dtype=np.float32),
        image_shape=(int(image_shape_array[0]), int(image_shape_array[1])),
    )
