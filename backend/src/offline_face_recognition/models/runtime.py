"""Load the pinned local models on OpenCV's CPU backend."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Callable

import cv2

from offline_face_recognition.core.config import Settings

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when local model assets cannot be loaded."""


@dataclass(frozen=True)
class ModelReadiness:
    yunet_ready: bool
    sface_ready: bool
    error_code: str | None

    @property
    def models_ready(self) -> bool:
        return self.yunet_ready and self.sface_ready


@dataclass
class ModelRuntime:
    """Own the process-local YuNet and SFace instances."""

    settings: Settings
    detector_factory: Callable[..., Any] | None = None
    recognizer_factory: Callable[..., Any] | None = None

    detector: Any | None = None
    recognizer: Any | None = None
    error_code: str | None = None
    detector_lock: threading.Lock = dataclass_field(default_factory=threading.Lock, init=False, repr=False)
    recognizer_lock: threading.Lock = dataclass_field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def readiness(self) -> ModelReadiness:
        return ModelReadiness(
            yunet_ready=self.detector is not None,
            sface_ready=self.recognizer is not None,
            error_code=self.error_code,
        )

    @property
    def recognition_ready(self) -> bool:
        # A matching threshold and ambiguity margin have not been calibrated.
        return False

    def load(self) -> None:
        """Load only the configured local model files; never fetch assets."""
        self.close()
        try:
            cv2.ocl.setUseOpenCL(False)
            if cv2.ocl.useOpenCL():
                raise ModelLoadError("OpenCL could not be disabled")

            missing = [
                path
                for path in (self.settings.yunet_model_path, self.settings.sface_model_path)
                if not Path(path).is_file()
            ]
            if missing:
                raise FileNotFoundError("A configured local ONNX model file is missing")

            detector_factory = self.detector_factory or cv2.FaceDetectorYN.create
            recognizer_factory = self.recognizer_factory or cv2.FaceRecognizerSF.create

            detector = detector_factory(
                str(self.settings.yunet_model_path),
                "",
                (320, 320),
                self.settings.detection_score_threshold,
                self.settings.detection_nms_threshold,
                self.settings.detection_top_k,
                cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
            recognizer = recognizer_factory(
                str(self.settings.sface_model_path),
                "",
                cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
            if detector is None or recognizer is None:
                raise ModelLoadError("OpenCV returned an empty model instance")

            self.detector = detector
            self.recognizer = recognizer
            self.error_code = None
        except Exception as exc:
            self.close()
            if isinstance(exc, FileNotFoundError):
                self.error_code = "model_files_missing"
            else:
                self.error_code = "model_load_failed"
            logger.error("Local face model initialization failed (%s)", type(exc).__name__)

    def close(self) -> None:
        """Release references to model objects during startup recovery or shutdown."""
        with self.detector_lock:
            self.detector = None
        with self.recognizer_lock:
            self.recognizer = None
