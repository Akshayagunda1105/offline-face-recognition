"""Compose detection and embedding without identity matching or gallery policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offline_face_recognition.faces.detection import DetectionResult, FaceDetection, YuNetDetector
from offline_face_recognition.faces.embedding import SFaceEmbedder
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

    def analyze(self, image_bgr: np.ndarray) -> FaceAnalysisResult:
        detection_result = self.detector.detect(image_bgr)
        analyzed: list[FaceEmbedding] = []
        for detection in detection_result.detections:
            # Fail the whole image operation with context; never drop a failed face.
            embedding = self.embedder.embed(
                image_bgr, detection, detection_index=detection.detection_index
            )
            analyzed.append(FaceEmbedding(detection=detection, embedding=embedding))
        return FaceAnalysisResult(detections=detection_result, faces=tuple(analyzed))
