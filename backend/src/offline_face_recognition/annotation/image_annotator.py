"""Draw recognition labels on a separate in-memory image copy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

import cv2
import numpy as np


class AnnotationStatus(StrEnum):
    KNOWN = "Known"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class Annotation:
    detection_index: int
    box: tuple[float, float, float, float]
    label: str
    status: AnnotationStatus = AnnotationStatus.KNOWN


class ImageAnnotationError(ValueError):
    """An image or annotation cannot be rendered safely."""


class ImageAnnotator:
    """Annotate boxes and labels without mutating the source image."""

    def annotate(self, image_bgr: np.ndarray, annotations: Sequence[Annotation]) -> np.ndarray:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.dtype != np.uint8
            or image_bgr.shape[0] <= 0
            or image_bgr.shape[1] <= 0
        ):
            raise ImageAnnotationError("Annotation input must be a non-empty uint8 BGR image")

        output = image_bgr.copy()
        height, width = output.shape[:2]
        colors = {
            AnnotationStatus.KNOWN: (0, 255, 0),
            AnnotationStatus.UNKNOWN: (0, 0, 255),
        }
        black = (0, 0, 0)
        for annotation in annotations:
            if annotation.detection_index < 0 or not annotation.label:
                raise ImageAnnotationError("Annotation index and label must be valid")
            x, y, box_width, box_height = annotation.box
            values = np.asarray((x, y, box_width, box_height), dtype=np.float64)
            if not np.isfinite(values).all() or box_width <= 0 or box_height <= 0:
                raise ImageAnnotationError("Annotation box must contain finite positive dimensions")
            left = max(0, min(width - 1, int(np.floor(x))))
            top = max(0, min(height - 1, int(np.floor(y))))
            right = max(left, min(width - 1, int(np.ceil(x + box_width))))
            bottom = max(top, min(height - 1, int(np.ceil(y + box_height))))
            color = colors[annotation.status]
            cv2.rectangle(output, (left, top), (right, bottom), color, 2)

            text = annotation.label.replace("\n", " ").replace("\r", " ")
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_top = max(0, top - text_height - baseline - 6)
            label_bottom = min(height - 1, label_top + text_height + baseline + 6)
            label_right = min(width - 1, left + text_width + 8)
            cv2.rectangle(output, (left, label_top), (label_right, label_bottom), color, -1)
            cv2.putText(
                output,
                text,
                (left + 4, min(label_bottom - baseline - 2, label_top + text_height + 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                black,
                1,
                cv2.LINE_AA,
            )
        return output
