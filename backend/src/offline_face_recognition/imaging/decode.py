"""Decode supported in-memory images into OpenCV-compatible BGR arrays."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from offline_face_recognition.core.config import Settings

_EXTENSION_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}


class ImageValidationError(ValueError):
    """An uploaded image violates the supported image contract."""


@dataclass(frozen=True)
class ImageLimits:
    """Resource limits used by the decoder."""

    max_upload_bytes: int
    max_width: int
    max_height: int
    max_pixels: int

    @classmethod
    def from_settings(cls, settings: Settings) -> ImageLimits:
        return cls(
            max_upload_bytes=settings.max_upload_bytes,
            max_width=settings.max_image_width,
            max_height=settings.max_image_height,
            max_pixels=settings.max_image_pixels,
        )


def _validate_dimensions(width: int, height: int, limits: ImageLimits) -> None:
    if width <= 0 or height <= 0:
        raise ImageValidationError("Image dimensions must be positive")
    if width > limits.max_width or height > limits.max_height:
        raise ImageValidationError("Image dimensions exceed the configured limit")
    if width * height > limits.max_pixels:
        raise ImageValidationError("Image pixel count exceeds the configured limit")


def decode_image(data: bytes, filename: str | Path, settings: Settings) -> np.ndarray:
    """Validate and decode one image without writing it to persistent storage.

    The returned array is contiguous ``uint8`` BGR, as expected by OpenCV.
    Supported extensions and encoded formats are JPG/JPEG, PNG and WebP.
    """
    limits = ImageLimits.from_settings(settings)
    name = str(filename)
    extension = Path(name.replace("\\", "/")).suffix.casefold()
    expected_format = _EXTENSION_FORMATS.get(extension)
    if expected_format is None:
        raise ImageValidationError("Unsupported image filename extension")
    if not isinstance(data, bytes) or not data:
        raise ImageValidationError("Image data is empty or invalid")
    if len(data) > limits.max_upload_bytes:
        raise ImageValidationError("Image file exceeds the configured byte limit")

    try:
        # Promote Pillow's decompression-bomb warning to a request-local error.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                if probe.format != expected_format:
                    raise ImageValidationError("Filename extension does not match image content")
                if getattr(probe, "n_frames", 1) != 1 or getattr(probe, "is_animated", False):
                    raise ImageValidationError("Multi-frame images are not supported")
                _validate_dimensions(*probe.size, limits)
                probe.verify()

            with Image.open(BytesIO(data)) as image:
                if image.format != expected_format:
                    raise ImageValidationError("Filename extension does not match image content")
                if getattr(image, "n_frames", 1) != 1 or getattr(image, "is_animated", False):
                    raise ImageValidationError("Multi-frame images are not supported")
                image.load()
                oriented = ImageOps.exif_transpose(image)
                _validate_dimensions(*oriented.size, limits)
                rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("Image exceeds Pillow's safe decompression limit") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError("Image is corrupted or cannot be decoded") from exc

    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
