"""Compose detection and embedding without identity matching or gallery policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offline_face_recognition.faces.detection import DetectionResult, FaceDetection, YuNetDetector
from offline_face_recognition.faces.embedding import SFaceEmbedder
from offline_face_recognition.faces.resizing import WorkingImage, prepare_working_image
from offline_face_recognition.models.runtime import ModelRuntime


class FaceEngineNotReady(RuntimeError):
    """The local face models are not loaded and ready for inference."""


@dataclass(frozen=True)
class FaceEmbedding:
    detection: FaceDetection
    embedding: np.ndarray


@dataclass(frozen=True)
class FaceAnalysisResult:
    detections: DetectionResult
    faces: tuple[FaceEmbedding, ...]


class FaceAnalysisEngine:
    """Run YuNet and SFace through the existing process-local model runtime."""

    def __init__(self, runtime: ModelRuntime) -> None:
        if not runtime.readiness.models_ready:
            raise FaceEngineNotReady("YuNet and SFace models must be loaded before analysis")
        self.detector = YuNetDetector(runtime)
        self.embedder = SFaceEmbedder(runtime)
        self._max_detection_dimension = runtime.settings.max_detection_dimension

    def analyze(self, image_bgr: np.ndarray) -> FaceAnalysisResult:
        prepared = prepare_working_image(image_bgr, self._max_detection_dimension)
        working_image = prepared.image_bgr
        try:
            detection_result = self.detector.detect(working_image)
            analyzed: list[FaceEmbedding] = []
            source_detections: list[FaceDetection] = []
            for detection in detection_result.detections:
                # YuNet raw row and SFace input both use the exact same working image.
                embedding = self.embedder.embed(
                    working_image, detection, detection_index=detection.detection_index
                )
                original_box = self._map_box_to_original(detection, prepared)
                source_detection = FaceDetection(
                    raw_row=detection.raw_row,
                    clipped_box=detection.clipped_box,
                    score=detection.score,
                    detection_index=detection.detection_index,
                    original_box=original_box,
                )
                source_detections.append(source_detection)
                analyzed.append(FaceEmbedding(detection=source_detection, embedding=embedding))
            mapped_result = DetectionResult(
                detections=tuple(source_detections),
                rejected_detection_count=detection_result.rejected_detection_count,
            )
            return FaceAnalysisResult(detections=mapped_result, faces=tuple(analyzed))
        finally:
            # Only clear the internally allocated downscaled copy. The caller owns
            # image_bgr and it must remain unchanged.
            if working_image is not image_bgr:
                working_image.fill(0)

    @staticmethod
    def _map_box_to_original(
        detection: FaceDetection, prepared: WorkingImage
    ) -> tuple[float, float, float, float]:
        x, y, width, height = detection.clipped_box
        left = max(0.0, min(float(prepared.original_width), x * prepared.scale_x))
        top = max(0.0, min(float(prepared.original_height), y * prepared.scale_y))
        right = max(left, min(float(prepared.original_width), (x + width) * prepared.scale_x))
        bottom = max(top, min(float(prepared.original_height), (y + height) * prepared.scale_y))
        return left, top, right - left, bottom - top
