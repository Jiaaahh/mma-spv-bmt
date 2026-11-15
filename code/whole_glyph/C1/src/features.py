"""Deterministic HOG feature extraction for SPV images."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.feature import hog

from src.contracts import FloatImage, FloatVector


class FeatureError(ValueError):
    """Raised when an image or extracted feature violates the HOG contract."""


class HogFeatureExtractor:
    """Thin validated wrapper around ``skimage.feature.hog``."""

    name = "hog"

    def __init__(
        self,
        *,
        image_shape: tuple[int, int],
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
    ) -> None:
        if (
            not isinstance(image_shape, tuple)
            or len(image_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in image_shape
            )
        ):
            raise FeatureError("image_shape must contain two positive integers")
        if isinstance(orientations, bool) or not isinstance(orientations, int):
            raise FeatureError("orientations must be a positive integer")
        if orientations <= 0:
            raise FeatureError("orientations must be a positive integer")
        for name, value in (
            ("pixels_per_cell", pixels_per_cell),
            ("cells_per_block", cells_per_block),
        ):
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0
                    for item in value
                )
            ):
                raise FeatureError(f"{name} must contain two positive integers")
        if block_norm != "L2-Hys":
            raise FeatureError("MVP HOG requires block_norm='L2-Hys'")

        self.image_shape = image_shape
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.feature_length = self._calculate_feature_length()

    def _calculate_feature_length(self) -> int:
        height, width = self.image_shape
        cell_h, cell_w = self.pixels_per_cell
        block_h, block_w = self.cells_per_block
        if height % cell_h or width % cell_w:
            raise FeatureError("image_shape must be divisible by pixels_per_cell")
        blocks_h = height // cell_h - block_h + 1
        blocks_w = width // cell_w - block_w + 1
        if blocks_h <= 0 or blocks_w <= 0:
            raise FeatureError("cells_per_block is larger than the HOG cell grid")
        return (
            blocks_h
            * blocks_w
            * block_h
            * block_w
            * self.orientations
        )

    def _validate_image(self, image: FloatImage) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")
        if image.dtype != np.float32 or image.ndim != 2:
            raise TypeError("image must be a two-dimensional float32 array")
        if image.shape != self.image_shape:
            raise FeatureError(
                f"image shape {image.shape} does not match {self.image_shape}"
            )
        if not np.isfinite(image).all() or float(image.min()) < 0.0:
            raise FeatureError("SPV image must contain finite non-negative values")

    def extract(self, image: FloatImage) -> FloatVector:
        self._validate_image(image)
        feature = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=False,
            transform_sqrt=False,
            feature_vector=True,
            channel_axis=None,
        )
        result = np.ascontiguousarray(feature, dtype=np.float32)
        if result.shape != (self.feature_length,):
            raise FeatureError(
                f"HOG returned length {result.size}, expected {self.feature_length}"
            )
        if not np.isfinite(result).all():
            raise FeatureError("HOG produced non-finite values")
        return result

    def extract_batch(
        self, images: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")
        if images.dtype != np.float32 or images.ndim != 3:
            raise TypeError("images must be a float32 array with shape (N, H, W)")
        if images.shape[0] == 0 or images.shape[1:] != self.image_shape:
            raise FeatureError(
                f"images must have shape (N, {self.image_shape[0]}, {self.image_shape[1]})"
            )
        features = np.empty(
            (images.shape[0], self.feature_length), dtype=np.float32
        )
        for sample_index, image in enumerate(images):
            features[sample_index] = self.extract(image)
        return np.ascontiguousarray(features, dtype=np.float32)
