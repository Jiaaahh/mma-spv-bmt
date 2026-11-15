"""Prepare independent fonts and unseen mappings for MMA confirmation.

This module only creates frozen input assets.  It deliberately contains no
feature extractor, classifier, prediction, or accuracy calculation.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import ImageFont

from src.contracts import GlyphSample
from src.deterministic_io import array_sha256, sha256_file, write_deterministic_npz
from src.grid import make_regular_grid
from src.mapping import make_ideal_mapping, make_moderate_mapping, mapping_statistics
from src.rendering import render_glyph, save_contact_sheet
from src.spv_forward import GaussianPhospheneForwardModel


class MmaAssetError(ValueError):
    """Raised when an independent confirmation asset is unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class ConfirmationFont:
    font_id: str
    path: Path
    face_index: int = 0

    def __post_init__(self) -> None:
        if not self.font_id or not all(
            character.isalnum() or character in "-_" for character in self.font_id
        ):
            raise MmaAssetError("font_id must be a non-empty filesystem-safe name")
        if not self.path.is_file():
            raise MmaAssetError(f"font file does not exist: {self.path}")
        if isinstance(self.face_index, bool) or self.face_index < 0:
            raise MmaAssetError("font face_index must be a non-negative integer")


def _atomic_json(path: Path, document: Any) -> None:
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _fingerprint_mask(font: ImageFont.FreeTypeFont, character: str) -> str:
    mask = font.getmask(character, mode="L")
    digest = hashlib.sha256()
    digest.update(str(mask.size).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(font.getbbox(character)).encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes(mask))
    return digest.hexdigest()


def _load_character_sequence(source_dataset: Path) -> tuple[str, ...]:
    if not source_dataset.is_file():
        raise MmaAssetError(f"source glyph archive does not exist: {source_dataset}")
    with np.load(source_dataset, allow_pickle=False) as archive:
        required = {"class_ids", "characters"}
        if not required <= set(archive.files):
            raise MmaAssetError("source glyph archive lacks class_ids/characters")
        class_ids = np.asarray(archive["class_ids"], dtype=np.int64)
        characters = np.asarray(archive["characters"], dtype=np.str_)
    if class_ids.shape != characters.shape or class_ids.ndim != 1:
        raise MmaAssetError("source class and character arrays differ in shape")
    unique_classes = np.unique(class_ids)
    if not np.array_equal(unique_classes, np.arange(180, dtype=np.int64)):
        raise MmaAssetError("confirmation protocol requires exactly class IDs 0..179")
    sequence: list[str] = []
    for class_id in unique_classes:
        values = np.unique(characters[class_ids == class_id])
        if values.shape != (1,) or len(str(values[0])) != 1:
            raise MmaAssetError(f"class {class_id} has an ambiguous character")
        sequence.append(str(values[0]))
    if len(set(sequence)) != 180:
        raise MmaAssetError("source character sequence contains duplicates")
    return tuple(sequence)


def _render_font(
    font: ConfirmationFont,
    characters: Sequence[str],
) -> tuple[list[GlyphSample], dict[str, Any]]:
    pillow_font = ImageFont.truetype(
        str(font.path), size=144, index=font.face_index
    )
    missing_fingerprint = _fingerprint_mask(pillow_font, chr(0x10FFFF))
    raw_fingerprints = [_fingerprint_mask(pillow_font, item) for item in characters]
    missing = [
        item
        for item, fingerprint in zip(characters, raw_fingerprints, strict=True)
        if fingerprint == missing_fingerprint
    ]
    if missing:
        raise MmaAssetError(
            f"font {font.font_id} lacks {len(missing)} required glyphs: "
            + "".join(missing[:10])
        )
    if len(set(raw_fingerprints)) != len(characters):
        raise MmaAssetError(
            f"font {font.font_id} produced duplicate raw glyph fingerprints"
        )

    samples: list[GlyphSample] = []
    normalized_hashes: set[str] = set()
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
        image_hash = array_sha256(image)
        if image_hash in normalized_hashes:
            raise MmaAssetError(
                f"font {font.font_id} produced duplicate normalized glyphs"
            )
        normalized_hashes.add(image_hash)
        samples.append(
            GlyphSample(
                sample_id=f"confirmation_{font.font_id}_c{class_id:03d}",
                character=character,
                class_id=class_id,
                font_id=font.font_id,
                split="test",
                image=image,
            )
        )
    details = {
        "font_id": font.font_id,
        "source_path": str(font.path.resolve()),
        "face_index": font.face_index,
        "file_sha256": sha256_file(font.path),
        "file_size_bytes": font.path.stat().st_size,
        "required_character_count": len(characters),
        "missing_character_count": 0,
        "unique_raw_glyph_fingerprints": len(set(raw_fingerprints)),
        "unique_normalized_glyphs": len(normalized_hashes),
    }
    return samples, details


def _mapping_arrays(mapping: Any, role: str) -> dict[str, np.ndarray]:
    model = GaussianPhospheneForwardModel(mapping)
    kernel_sums = model.kernel_bank.sum(axis=(1, 2), dtype=np.float64)
    if not np.allclose(kernel_sums, 1.0, rtol=0.0, atol=1e-5):
        raise MmaAssetError("generated mapping has non-L1 phosphene kernels")
    return {
        "format_version": np.asarray([1], dtype=np.int64),
        "mapping_id": np.asarray([mapping.mapping_id], dtype=np.str_),
        "mapping_role": np.asarray([role], dtype=np.str_),
        "seed": np.asarray([mapping.seed], dtype=np.int64),
        "image_shape": np.asarray(mapping.image_shape, dtype=np.int64),
        "centers_xy": np.asarray(mapping.centers_xy, dtype=np.float32),
        "sigma_px": np.asarray(mapping.sigma_px, dtype=np.float32),
        "base_sigma_grid": np.asarray([0.45], dtype=np.float64),
        "position_jitter_std_grid": np.asarray(
            [0.0 if mapping.seed == 0 else 0.30], dtype=np.float64
        ),
        "sigma_scale_std": np.asarray(
            [0.0 if mapping.seed == 0 else 0.20], dtype=np.float64
        ),
        "sigma_scale_clip": np.asarray([0.60, 1.60], dtype=np.float64),
        "kernel_normalization": np.asarray(
            ["l1_after_canvas_boundary"], dtype=np.str_
        ),
        "kernel_bank_sha256": np.asarray(
            [array_sha256(model.kernel_bank)], dtype=np.str_
        ),
        "used_for_accuracy": np.asarray([False], dtype=np.bool_),
    }


def prepare_confirmation_assets(
    *,
    source_dataset: Path,
    fonts: Sequence[ConfirmationFont],
    mapping_seeds: Sequence[int],
    output_dir: Path,
    replace: bool = False,
) -> dict[str, Any]:
    """Render unseen-font glyphs and generate unseen mappings, without scoring."""

    if len(fonts) < 2 or len({item.font_id for item in fonts}) != len(fonts):
        raise MmaAssetError("at least two uniquely named confirmation fonts are required")
    seeds = tuple(int(seed) for seed in mapping_seeds)
    if len(seeds) < 5 or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise MmaAssetError("at least five unique positive mapping seeds are required")
    output_dir = output_dir.resolve()
    glyph_path = output_dir / "glyphs_confirmation.npz"
    mapping_dir = output_dir / "mappings"
    manifest_path = output_dir / "assets_manifest.json"
    expected = [
        glyph_path,
        manifest_path,
        *(mapping_dir / f"moderate_seed_{seed}.npz" for seed in seeds),
        *(output_dir / f"contact_sheet_{font.font_id}.png" for font in fonts),
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not replace:
        raise MmaAssetError(
            "confirmation assets already exist; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    characters = _load_character_sequence(source_dataset)
    all_samples: list[GlyphSample] = []
    font_details: list[dict[str, Any]] = []
    samples_by_font: dict[str, list[GlyphSample]] = {}
    for font in fonts:
        samples, details = _render_font(font, characters)
        all_samples.extend(samples)
        samples_by_font[font.font_id] = samples
        font_details.append(details)
    if len(all_samples) != len(fonts) * 180:
        raise MmaAssetError("confirmation glyph count differs from fonts x 180")

    grid = make_regular_grid(rows=16, cols=16, image_shape=(96, 96))
    ideal = make_ideal_mapping(grid, base_sigma_grid=0.45)
    mappings = [
        make_moderate_mapping(
            grid,
            seed=seed,
            base_sigma_grid=0.45,
            position_jitter_std_grid=0.30,
            sigma_scale_std=0.20,
            sigma_scale_clip=(0.60, 1.60),
        )
        for seed in seeds
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir.mkdir(parents=True, exist_ok=True)
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
    glyph_sha256 = write_deterministic_npz(glyph_path, glyph_arrays)

    mapping_details: list[dict[str, Any]] = []
    for mapping in mappings:
        path = mapping_dir / f"moderate_seed_{mapping.seed}.npz"
        mapping_sha256 = write_deterministic_npz(
            path, _mapping_arrays(mapping, "independent_confirmation_unseen")
        )
        statistics = mapping_statistics(ideal, mapping)
        mapping_details.append(
            {
                "mapping_id": mapping.mapping_id,
                "seed": mapping.seed,
                "file": path.relative_to(output_dir).as_posix(),
                "file_sha256": mapping_sha256,
                "centers_sha256": array_sha256(mapping.centers_xy),
                "sigma_sha256": array_sha256(mapping.sigma_px),
                "geometry_statistics": {
                    "mean_displacement_px": statistics.mean_displacement_px,
                    "max_displacement_px": statistics.max_displacement_px,
                    "mean_sigma_px": statistics.mean_sigma_px,
                    "min_sigma_px": statistics.min_sigma_px,
                    "max_sigma_px": statistics.max_sigma_px,
                },
            }
        )
    for font in fonts:
        save_contact_sheet(
            samples_by_font[font.font_id],
            output_dir / f"contact_sheet_{font.font_id}.png",
            columns=10,
        )

    manifest = {
        "schema_version": 1,
        "status": "FROZEN_INPUTS_ONLY_NO_PERFORMANCE_READ",
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "source_character_archive": {
            "path": str(source_dataset.resolve()),
            "sha256": sha256_file(source_dataset),
            "character_sequence_sha256": hashlib.sha256(
                "".join(characters).encode("utf-8")
            ).hexdigest(),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "rendering": {
            "image_size": [96, 96],
            "render_canvas_size": [192, 192],
            "font_size": 144,
            "content_margin": 8,
            "normalization": "aspect_fit_center_lanczos",
        },
        "glyph_archive": {
            "file": glyph_path.relative_to(output_dir).as_posix(),
            "file_sha256": glyph_sha256,
            "sample_count": len(all_samples),
            "class_count": 180,
            "font_count": len(fonts),
            "images_sha256": array_sha256(glyph_arrays["images"]),
        },
        "fonts": font_details,
        "mapping_parameters": {
            "grid": [16, 16],
            "image_shape": [96, 96],
            "base_sigma_grid": 0.45,
            "position_jitter_std_grid": 0.30,
            "sigma_scale_std": 0.20,
            "sigma_scale_clip": [0.60, 1.60],
            "kernel_normalization": "l1_after_canvas_boundary",
        },
        "mappings": mapping_details,
        "prediction_executed": False,
        "accuracy_read": False,
    }
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
