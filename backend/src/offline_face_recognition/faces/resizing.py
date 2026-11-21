"""Aspect-preserving downscaling for face analysis working images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class WorkingImage:
    image_bgr: np.ndarray
    original_width: int
    original_height: int
    scale_x: float
    scale_y: float


def prepare_working_image(image_bgr: np.ndarray, max_dimension: int) -> WorkingImage:
    """Return an aspect-preserving working image, never upscaling the source."""
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.dtype != np.uint8
        or image_bgr.shape[0] <= 0
        or image_bgr.shape[1] <= 0
    ):
        raise ValueError("Working image input must be a non-empty uint8 BGR image")
    if isinstance(max_dimension, bool) or not isinstance(max_dimension, int) or max_dimension <= 0:
        raise ValueError("Maximum detection dimension must be a positive integer")

    original_height, original_width = image_bgr.shape[:2]
    longest = max(original_width, original_height)
    if longest <= max_dimension:
        # Do not hand caller-owned memory to stateful/native inference routines.
        return WorkingImage(image_bgr.copy(), original_width, original_height, 1.0, 1.0)

    scale = max_dimension / float(longest)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    working = cv2.resize(
        image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    # Use actual rounded dimensions, not the ideal scalar, for inverse mapping.
    return WorkingImage(
        working,
        original_width,
        original_height,
        original_width / resized_width,
        original_height / resized_height,
    )
