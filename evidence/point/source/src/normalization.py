"""Geometry-preserving normalization for rendered grayscale glyph masks."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class NormalizationError(ValueError):
    """Raised when a glyph cannot be converted into a valid normalized image."""


def foreground_bbox(image: NDArray[np.floating[Any]]) -> tuple[int, int, int, int]:
    """Return (top, left, bottom, right), with bottom/right exclusive."""

    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise NormalizationError("foreground_bbox expects a two-dimensional array")
    coordinates = np.argwhere(image > 0.0)
    if coordinates.size == 0:
        raise NormalizationError("rendered glyph contains no foreground pixels")
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0) + 1
    return int(top), int(left), int(bottom), int(right)


def _to_unit_grayscale(raw_image: Image.Image | np.ndarray[Any, Any]) -> NDArray[np.float32]:
    if isinstance(raw_image, Image.Image):
        array = np.asarray(raw_image.convert("L"))
    elif isinstance(raw_image, np.ndarray):
        array = raw_image
    else:
        raise TypeError("raw_image must be a PIL image or numpy array")
    if array.ndim != 2 or array.size == 0:
        raise NormalizationError("raw glyph must be a non-empty grayscale image")
    if not np.issubdtype(array.dtype, np.number):
        raise NormalizationError("raw glyph must use a numeric dtype")
    values = np.asarray(array, dtype=np.float32)
    if not np.isfinite(values).all():
        raise NormalizationError("raw glyph contains non-finite values")
    minimum, maximum = float(values.min()), float(values.max())
    if minimum < 0:
        raise NormalizationError("raw glyph values must be non-negative")
    if maximum <= 1.0:
        return values
    if maximum <= 255.0:
        return values / np.float32(255.0)
    raise NormalizationError("raw glyph values must lie in [0, 1] or [0, 255]")


def validate_normalized_image(
    image: NDArray[np.float32],
    *,
    expected_shape: tuple[int, int] | None = None,
) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.float32 or image.ndim != 2:
        raise NormalizationError("normalized image must be a two-dimensional float32 array")
    if expected_shape is not None and image.shape != expected_shape:
        raise NormalizationError(
            f"normalized image shape is {image.shape}, expected {expected_shape}"
        )
    if not np.isfinite(image).all():
        raise NormalizationError("normalized image contains non-finite values")
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise NormalizationError("normalized image values must lie in [0, 1]")
    if float(image.max() - image.min()) <= 0.0:
        raise NormalizationError("normalized image contains no visible glyph")


def normalize_glyph(
    raw_image: Image.Image | np.ndarray[Any, Any],
    *,
    output_shape: tuple[int, int] = (96, 96),
    content_margin: int = 8,
    foreground_value: float = 1.0,
    background_value: float = 0.0,
) -> NDArray[np.float32]:
    """Crop, aspect-fit, and center one glyph on a fixed-size canvas."""

    if (
        not isinstance(output_shape, tuple)
        or len(output_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in output_shape)
    ):
        raise NormalizationError("output_shape must contain two positive integers")
    if isinstance(content_margin, bool) or not isinstance(content_margin, int) or content_margin < 0:
        raise NormalizationError("content_margin must be a non-negative integer")
    output_h, output_w = output_shape
    if 2 * content_margin >= min(output_shape):
        raise NormalizationError("content_margin leaves no drawable output area")
    for name, value in (
        ("foreground_value", foreground_value),
        ("background_value", background_value),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise NormalizationError(f"{name} must lie in [0, 1]")
    if float(foreground_value) == float(background_value):
        raise NormalizationError("foreground and background values must differ")

    unit = _to_unit_grayscale(raw_image)
    top, left, bottom, right = foreground_bbox(unit)
    crop = unit[top:bottom, left:right]
    crop_h, crop_w = crop.shape
    available_h = output_h - 2 * content_margin
    available_w = output_w - 2 * content_margin
    scale = min(available_h / crop_h, available_w / crop_w)
    resized_h = max(1, min(available_h, int(round(crop_h * scale))))
    resized_w = max(1, min(available_w, int(round(crop_w * scale))))

    crop_u8 = np.rint(np.clip(crop, 0.0, 1.0) * 255.0).astype(np.uint8)
    resized = Image.fromarray(crop_u8, mode="L").resize(
        (resized_w, resized_h), Image.Resampling.LANCZOS
    )
    resized_unit = np.asarray(resized, dtype=np.float32) / np.float32(255.0)

    mask = np.zeros(output_shape, dtype=np.float32)
    offset_y = (output_h - resized_h) // 2
    offset_x = (output_w - resized_w) // 2
    mask[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized_unit
    normalized = np.float32(background_value) + mask * np.float32(
        foreground_value - background_value
    )
    normalized = np.ascontiguousarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)
    validate_normalized_image(normalized, expected_shape=output_shape)
    return normalized
