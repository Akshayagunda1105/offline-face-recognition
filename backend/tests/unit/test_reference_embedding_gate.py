import threading
from types import SimpleNamespace

import numpy as np
import pytest

from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.engine import FaceAnalysisEngine


def _yunet_rows(count):
    rows = []
    for index in range(count):
        rows.append([2 + index, 2, 8, 8, *([5 + index, 5] * 5), 0.95])
    return np.asarray(rows, dtype=np.float32).reshape(count, 15)


class FakeDetector:
    def __init__(self, rows):
        self.rows = rows
        self.input_sizes = []

    def setInputSize(self, size):
        self.input_sizes.append(size)

    def detect(self, _image):
        return True, self.rows


class FakeRecognizer:
    def __init__(self):
        self.aligned_rows = []
        self.feature_calls = 0

    def alignCrop(self, image, row):
        self.aligned_rows.append((image.shape, row.copy()))
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def feature(self, _crop):
        self.feature_calls += 1
        return np.ones((1, 128), dtype=np.float32)


def _engine(rows):
    runtime = SimpleNamespace(
        readiness=SimpleNamespace(models_ready=True),
        settings=Settings(project_root=".", yunet_model_path="yunet.onnx", sface_model_path="sface.onnx"),
        detector=FakeDetector(rows),
        recognizer=FakeRecognizer(),
        detector_lock=threading.Lock(),
        recognizer_lock=threading.Lock(),
    )
    return FaceAnalysisEngine(runtime), runtime


@pytest.mark.parametrize("count", [0, 2])
def test_reference_analysis_skips_sface_when_face_count_is_not_exactly_one(count):
    rows = _yunet_rows(count) if count else None
    engine, runtime = _engine(rows)

    result = engine.analyze_reference(np.zeros((20, 20, 3), dtype=np.uint8))

    assert len(result.detections.detections) == count
    assert result.faces == ()
    assert runtime.recognizer.aligned_rows == []
    assert runtime.recognizer.feature_calls == 0


def test_reference_analysis_aligns_and_embeds_single_face_in_detector_working_frame():
    engine, runtime = _engine(_yunet_rows(1))
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    result = engine.analyze_reference(image)

    assert len(result.detections.detections) == 1
    assert len(result.faces) == 1
    assert runtime.recognizer.aligned_rows[0][0] == image.shape
    assert runtime.recognizer.aligned_rows[0][1].shape == (1, 15)
    assert runtime.recognizer.feature_calls == 1
    assert result.faces[0].embedding.shape == (128,)
    assert np.linalg.norm(result.faces[0].embedding) == pytest.approx(1.0)
