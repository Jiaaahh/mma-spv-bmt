"""Pillow-based Chinese glyph rendering and visual quality-control output."""

from __future__ import annotations

from collections import defaultdict
import io
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from src.contracts import GlyphSample
from src.normalization import normalize_glyph


class RenderingError(ValueError):
    """Raised when a font cannot produce a valid normalized glyph."""


def render_glyph(
    character: str,
    font_path: str | Path,
    *,
    face_index: int = 0,
    image_size: tuple[int, int] = (96, 96),
    render_canvas_size: tuple[int, int] = (192, 192),
    font_size: int = 144,
    content_margin: int = 8,
    foreground_value: float = 1.0,
    background_value: float = 0.0,
) -> NDArray[np.float32]:
    """Render one Unicode character and return a normalized float32 mask."""

    if len(character) != 1 or character.isspace():
        raise RenderingError("character must contain exactly one non-space code point")
    source = Path(font_path)
    if not source.is_file():
        raise RenderingError(f"font file does not exist: {source}")
    if isinstance(face_index, bool) or not isinstance(face_index, int) or face_index < 0:
        raise RenderingError("face_index must be a non-negative integer")
    if isinstance(font_size, bool) or not isinstance(font_size, int) or font_size <= 0:
        raise RenderingError("font_size must be a positive integer")
    if (
        not isinstance(render_canvas_size, tuple)
        or len(render_canvas_size) != 2
        or any(value <= 0 for value in render_canvas_size)
    ):
        raise RenderingError("render_canvas_size must contain two positive integers")

    canvas_h, canvas_w = render_canvas_size
    try:
        font = ImageFont.truetype(str(source), size=font_size, index=face_index)
        bbox = font.getbbox(character)
    except OSError as exc:
        raise RenderingError(f"failed to load font {source}: {exc}") from exc
    if bbox is None:
        raise RenderingError(f"font returned no bounding box for {character!r}")
    left, top, right, bottom = bbox
    glyph_w, glyph_h = right - left, bottom - top
    if glyph_w <= 0 or glyph_h <= 0:
        raise RenderingError(f"font returned an empty glyph for {character!r}")

    canvas = Image.new("L", (canvas_w, canvas_h), color=0)
    draw = ImageDraw.Draw(canvas)
    draw_x = (canvas_w - glyph_w) / 2.0 - left
    draw_y = (canvas_h - glyph_h) / 2.0 - top
    draw.text((round(draw_x), round(draw_y)), character, font=font, fill=255)
    raw_pixels = np.asarray(canvas)
    foreground = np.argwhere(raw_pixels > 0)
    if foreground.size == 0:
        raise RenderingError(f"font rendered no visible pixels for {character!r}")
    raw_top, raw_left = foreground.min(axis=0)
    raw_bottom, raw_right = foreground.max(axis=0)
    if raw_top == 0 or raw_left == 0 or raw_bottom == canvas_h - 1 or raw_right == canvas_w - 1:
        raise RenderingError(
            f"glyph {character!r} touches the raw canvas boundary; increase render_canvas_size"
        )
    try:
        return normalize_glyph(
            raw_pixels,
            output_shape=image_size,
            content_margin=content_margin,
            foreground_value=foreground_value,
            background_value=background_value,
        )
    except ValueError as exc:
        raise RenderingError(f"failed to normalize glyph {character!r}: {exc}") from exc


def save_contact_sheet(
    samples: Sequence[GlyphSample],
    output_path: str | Path,
    *,
    columns: int = 10,
) -> None:
    """Save an ASCII-labelled contact sheet for manual glyph inspection."""

    if not samples:
        raise RenderingError("cannot create a contact sheet from zero samples")
    if columns <= 0:
        raise RenderingError("contact sheet columns must be positive")
    image_shape = samples[0].image.shape
    if any(sample.image.shape != image_shape for sample in samples):
        raise RenderingError("all contact-sheet images must have the same shape")

    grouped: dict[tuple[str, str], list[GlyphSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.split, sample.font_id)].append(sample)
    for group in grouped.values():
        group.sort(key=lambda sample: sample.class_id)

    image_h, image_w = image_shape
    header_h = 18
    gap = 6
    label_w = 150
    cell_w = image_w + 2 * gap
    cell_h = image_h + header_h + 2 * gap
    rows_per_font = max((len(group) + columns - 1) // columns for group in grouped.values())
    total_rows = rows_per_font * len(grouped)
    sheet = Image.new(
        "RGB",
        (label_w + columns * cell_w, total_rows * cell_h),
        color=(242, 244, 247),
    )
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.load_default()

    row_cursor = 0
    for (split, font_id), group in grouped.items():
        for chunk_start in range(0, len(group), columns):
            chunk = group[chunk_start : chunk_start + columns]
            y = row_cursor * cell_h
            draw.text(
                (8, y + 8),
                f"{split}: {font_id}",
                fill=(31, 41, 55),
                font=label_font,
            )
            for column, sample in enumerate(chunk):
                x = label_w + column * cell_w
                draw.rectangle(
                    (x + 1, y + 1, x + cell_w - 2, y + cell_h - 2),
                    fill=(255, 255, 255),
                    outline=(203, 213, 225),
                )
                draw.text(
                    (x + gap, y + 3),
                    f"c{sample.class_id:02d} U+{ord(sample.character):04X}",
                    fill=(71, 85, 105),
                    font=label_font,
                )
                glyph_u8 = np.rint(np.clip(sample.image, 0.0, 1.0) * 255.0).astype(
                    np.uint8
                )
                glyph = Image.fromarray(glyph_u8, mode="L").convert("RGB")
                sheet.paste(glyph, (x + gap, y + header_h + gap))
            row_cursor += 1
        row_cursor += rows_per_font - ((len(group) + columns - 1) // columns)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Encode fully in memory before publishing the file.  This avoids leaving
    # a truncated contact sheet if a filesystem write is interrupted and also
    # makes the PNG bytes deterministic for a frozen Pillow version.
    encoded = io.BytesIO()
    sheet.save(encoded, format="PNG", optimize=False, compress_level=6)
    payload = encoded.getvalue()
    try:
        with Image.open(io.BytesIO(payload)) as preview:
            preview.verify()
    except OSError as exc:
        raise RenderingError(f"failed to encode a valid contact sheet: {exc}") from exc
    temporary = destination.with_name(f"{destination.name}.write_tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
