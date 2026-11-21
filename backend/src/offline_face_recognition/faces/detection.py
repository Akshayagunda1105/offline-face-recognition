"""YuNet face detection over decoded OpenCV BGR images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from offline_face_recognition.models.runtime import ModelRuntime


class FaceDetectionError(RuntimeError):
    """YuNet could not produce a structurally valid detection response."""


@dataclass(frozen=True)
class FaceDetection:
    """A validated YuNet result with immutable raw model values and clipped box."""

    raw_row: tuple[float, ...]
    clipped_box: tuple[float, float, float, float]
    score: float
    detection_index: int = 0
    # Box mapped to the original source image; raw_row/clipped_box remain in
    # the detector working-image coordinate frame used by SFace alignment.
    original_box: tuple[float, float, float, float] | None = None

    @property
    def landmarks(self) -> tuple[tuple[float, float], ...]:
        values = self.raw_row[4:14]
        return tuple((values[index], values[index + 1]) for index in range(0, 10, 2))


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[FaceDetection, ...]
    rejected_detection_count: int = 0


class YuNetDetector:
    """Use the already-loaded runtime detector; never instantiate a model."""

    def __init__(self, runtime: ModelRuntime) -> None:
        if runtime.detector is None:
            raise FaceDetectionError("YuNet model is not ready")
        self._runtime = runtime

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.shape[0] <= 0
            or image_bgr.shape[1] <= 0
            or image_bgr.dtype != np.uint8
        ):
            raise FaceDetectionError("Detection input must be a non-empty uint8 BGR image")

        height, width = image_bgr.shape[:2]
        try:
            with self._runtime.detector_lock:
                self._runtime.detector.setInputSize((width, height))
                _status, face_rows = self._runtime.detector.detect(image_bgr)
        except Exception as exc:
            raise FaceDetectionError("YuNet inference failed") from exc

        if face_rows is None:
            return DetectionResult(())
        try:
            rows = np.asarray(face_rows)
        except (TypeError, ValueError) as exc:
            raise FaceDetectionError("YuNet returned an invalid detection array") from exc
        if rows.size == 0:
            return DetectionResult(())
        if rows.ndim != 2 or rows.shape[1] != 15:
            raise FaceDetectionError("YuNet detections must have 15 values per face")

        valid: list[FaceDetection] = []
        rejected = 0
        for detection_index, row in enumerate(rows):
            try:
                values = np.asarray(row, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                rejected += 1
                continue
            if not np.isfinite(values).all():
                rejected += 1
                continue
            x, y, box_width, box_height = (float(value) for value in values[:4])
            score = float(values[14])
            if box_width <= 0.0 or box_height <= 0.0 or not 0.0 <= score <= 1.0:
                rejected += 1
                continue
            left = max(0.0, x)
            top = max(0.0, y)
            right = min(float(width), x + box_width)
            bottom = min(float(height), y + box_height)
            if right <= left or bottom <= top:
                rejected += 1
                continue
            # Preserve YuNet's row values for SFace; clipping is separate metadata.
            raw_row = tuple(float(value) for value in row)
            valid.append(
                FaceDetection(
                    raw_row=raw_row,
                    clipped_box=(left, top, right - left, bottom - top),
                    score=score,
                    detection_index=detection_index,
                )
            )

        return DetectionResult(tuple(valid), rejected_detection_count=rejected)
