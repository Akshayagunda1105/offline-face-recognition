import numpy as np
import pytest

from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.detection import FaceDetection
from offline_face_recognition.faces.engine import FaceAnalysisEngine, FaceEngineNotReady
from offline_face_recognition.faces.embedding import EmbeddingGenerationError, SFaceEmbedder
from offline_face_recognition.models.runtime import ModelRuntime


def _face():
    row = tuple(float(value) for value in [10, 12, 20, 22, *([15, 16] * 5), 0.9])
    return FaceDetection(row, (10.0, 12.0, 20.0, 22.0), 0.9)


def _runtime(recognizer):
    return ModelRuntime(
        Settings(project_root=".", yunet_model_path="yunet.onnx", sface_model_path="sface.onnx"),
        detector=object(),
        recognizer=recognizer,
    )


class FakeRecognizer:
    def __init__(self, feature=None, *, align_error=None, feature_error=None):
        self.feature_value = feature if feature is not None else np.ones((1, 128), np.float32)
        self.align_error = align_error
        self.feature_error = feature_error
        self.received_rows = []

    def alignCrop(self, image, face_row):
        self.received_rows.append(face_row.copy())
        if self.align_error:
            raise self.align_error
        return np.zeros((112, 112, 3), np.uint8)

    def feature(self, _crop):
        if self.feature_error:
            raise self.feature_error
        return self.feature_value


def test_sface_aligns_with_original_yunet_row_and_normalizes_output():
    recognizer = FakeRecognizer(np.arange(1, 129, dtype=np.float32).reshape(1, 128))
    embedder = SFaceEmbedder(_runtime(recognizer))
    image = np.zeros((48, 64, 3), np.uint8)
    detection = _face()

    aligned = embedder.align_crop(image, detection, detection_index=3)
    embedding = embedder.embed(image, detection, detection_index=3)

    np.testing.assert_array_equal(recognizer.received_rows[0], np.asarray(detection.raw_row, np.float32).reshape(1, 15))
    assert aligned.shape == (112, 112, 3)
    assert embedding.shape == (128,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-6)


def test_sface_alignment_uses_unclipped_raw_row():
    original = np.asarray(_face().raw_row, dtype=np.float32)
    original[0] = -5.0
    detection = FaceDetection(tuple(float(value) for value in original), (0.0, 12.0, 15.0, 22.0), 0.9)
    recognizer = FakeRecognizer()

    SFaceEmbedder(_runtime(recognizer)).align_crop(
        np.zeros((48, 64, 3), np.uint8), detection
    )

    np.testing.assert_array_equal(recognizer.received_rows[0], original.reshape(1, 15))


@pytest.mark.parametrize(
    ("feature", "message"),
    [
        (np.ones((1, 127), np.float32), "shape or type"),
        (np.full((1, 128), np.nan, np.float32), "non-finite"),
        (np.zeros((1, 128), np.float32), "zero or invalid"),
    ],
)
def test_invalid_sface_features_raise_contextual_error(feature, message):
    embedder = SFaceEmbedder(_runtime(FakeRecognizer(feature)))

    with pytest.raises(EmbeddingGenerationError, match=message) as error:
        embedder.embed(np.zeros((48, 64, 3), np.uint8), _face(), detection_index=7)

    assert error.value.detection_index == 7
    assert "detection index 7" in str(error.value)


def test_alignment_and_feature_inference_errors_include_detection_context():
    image = np.zeros((48, 64, 3), np.uint8)
    with pytest.raises(EmbeddingGenerationError, match="detection index 2"):
        SFaceEmbedder(_runtime(FakeRecognizer(align_error=RuntimeError()))).embed(
            image, _face(), detection_index=2
        )
    with pytest.raises(EmbeddingGenerationError, match="detection index 4"):
        SFaceEmbedder(_runtime(FakeRecognizer(feature_error=RuntimeError()))).embed(
            image, _face(), detection_index=4
        )


def test_bad_aligned_crop_is_rejected():
    class BadCropRecognizer(FakeRecognizer):
        def alignCrop(self, _image, _row):
            return np.zeros((10, 10, 3), np.uint8)

    with pytest.raises(EmbeddingGenerationError, match="invalid aligned crop"):
        SFaceEmbedder(_runtime(BadCropRecognizer())).embed(
            np.zeros((48, 64, 3), np.uint8), _face(), detection_index=0
        )


def test_engine_fails_with_detection_index_instead_of_skipping_embedding_failure():
    class TwoFaceDetector:
        def setInputSize(self, _size):
            pass

        def detect(self, _image):
            row = np.asarray(_face().raw_row, dtype=np.float32)
            second = row.copy()
            second[0] += 30
            return 2, np.stack([row, second])

    class FailOnSecondFace(FakeRecognizer):
        calls = 0

        def alignCrop(self, image, row):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second-face alignment error")
            return super().alignCrop(image, row)

    runtime = _runtime(FailOnSecondFace())
    runtime.detector = TwoFaceDetector()
    engine = FaceAnalysisEngine(runtime)

    with pytest.raises(EmbeddingGenerationError, match="detection index 1"):
        engine.analyze(np.zeros((64, 96, 3), np.uint8))


def test_engine_rejects_runtime_without_both_loaded_models():
    runtime = _runtime(FakeRecognizer())
    runtime.detector = None

    with pytest.raises(FaceEngineNotReady):
        FaceAnalysisEngine(runtime)


def test_multiple_engine_wrappers_share_runtime_owned_locks():
    runtime = _runtime(FakeRecognizer())
    runtime.detector = object()

    first = FaceAnalysisEngine(runtime)
    second = FaceAnalysisEngine(runtime)

    assert first.detector._runtime.detector_lock is second.detector._runtime.detector_lock
    assert first.embedder._runtime.recognizer_lock is second.embedder._runtime.recognizer_lock
