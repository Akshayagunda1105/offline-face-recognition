"""SFace landmark alignment and normalized feature extraction."""

from __future__ import annotations

import numpy as np

from offline_face_recognition.faces.detection import FaceDetection
from offline_face_recognition.models.runtime import ModelRuntime


class EmbeddingGenerationError(RuntimeError):
    """Alignment or feature generation failed for a detected face."""

    def __init__(self, message: str, *, detection_index: int | None = None) -> None:
        self.detection_index = detection_index
        context = f" at detection index {detection_index}" if detection_index is not None else ""
        super().__init__(f"{message}{context}")


class SFaceEmbedder:
    """Align and embed faces using the recognizer already held by ModelRuntime."""

    def __init__(self, runtime: ModelRuntime) -> None:
        if runtime.recognizer is None:
            raise EmbeddingGenerationError("SFace model is not ready")
        self._runtime = runtime

    def align_crop(
        self,
        image_bgr: np.ndarray,
        detection: FaceDetection,
        *,
        detection_index: int | None = None,
    ) -> np.ndarray:
        raw_row = np.asarray(detection.raw_row, dtype=np.float32).reshape(1, 15)
        try:
            with self._runtime.recognizer_lock:
                crop = self._runtime.recognizer.alignCrop(image_bgr, raw_row.copy())
        except Exception as exc:
            raise EmbeddingGenerationError(
                "SFace face alignment failed", detection_index=detection_index
            ) from exc
        if (
            not isinstance(crop, np.ndarray)
            or crop.shape != (112, 112, 3)
            or crop.dtype != np.uint8
            or not np.isfinite(crop).all()
        ):
            raise EmbeddingGenerationError(
                "SFace returned an invalid aligned crop", detection_index=detection_index
            )
        return crop

    def extract_feature(
        self,
        aligned_crop: np.ndarray,
        *,
        detection_index: int | None = None,
    ) -> np.ndarray:
        try:
            with self._runtime.recognizer_lock:
                raw_feature = self._runtime.recognizer.feature(aligned_crop)
        except Exception as exc:
            raise EmbeddingGenerationError(
                "SFace feature extraction failed", detection_index=detection_index
            ) from exc
        feature = np.asarray(raw_feature)
        if (
            feature.shape != (1, 128)
            or not np.issubdtype(feature.dtype, np.number)
            or np.iscomplexobj(feature)
        ):
            raise EmbeddingGenerationError(
                "SFace returned a feature with an invalid shape or type",
                detection_index=detection_index,
            )
        if not np.isfinite(feature).all():
            raise EmbeddingGenerationError(
                "SFace returned non-finite feature values", detection_index=detection_index
            )
        return np.asarray(feature, dtype=np.float32).reshape(128)

    @staticmethod
    def normalize_feature(
        feature: np.ndarray,
        *,
        detection_index: int | None = None,
    ) -> np.ndarray:
        vector = np.asarray(feature)
        if (
            vector.shape not in {(128,), (1, 128)}
            or not np.issubdtype(vector.dtype, np.number)
            or np.iscomplexobj(vector)
        ):
            raise EmbeddingGenerationError(
                "Face embedding must contain exactly 128 numeric values",
                detection_index=detection_index,
            )
        vector = np.asarray(vector, dtype=np.float32).reshape(128)
        if not np.isfinite(vector).all():
            raise EmbeddingGenerationError(
                "Face embedding contains non-finite values", detection_index=detection_index
            )
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.isfinite(norm) or norm <= np.finfo(np.float32).tiny:
            raise EmbeddingGenerationError(
                "Face embedding has a zero or invalid L2 norm", detection_index=detection_index
            )
        normalized = np.ascontiguousarray(vector / np.float32(norm), dtype=np.float32)
        normalized_norm = float(np.linalg.norm(normalized.astype(np.float64)))
        if not np.isfinite(normalized).all() or not np.isclose(normalized_norm, 1.0, atol=1e-6):
            raise EmbeddingGenerationError(
                "Normalized face embedding failed validation", detection_index=detection_index
            )
        return normalized

    def embed(
        self,
        image_bgr: np.ndarray,
        detection: FaceDetection,
        *,
        detection_index: int | None = None,
    ) -> np.ndarray:
        crop = self.align_crop(image_bgr, detection, detection_index=detection_index)
        feature = self.extract_feature(crop, detection_index=detection_index)
        return self.normalize_feature(feature, detection_index=detection_index)
