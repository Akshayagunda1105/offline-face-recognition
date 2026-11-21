from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.detection import DetectionResult, FaceDetection
from offline_face_recognition.faces.embedding import EmbeddingGenerationError
from offline_face_recognition.faces.engine import FaceAnalysisResult, FaceEmbedding
from offline_face_recognition.gallery.builder import GalleryBuildError, GalleryBuilder, ReferenceImage
from offline_face_recognition.gallery.session_store import GallerySessionStore


def _settings(**overrides):
    values = {
        "project_root": ".",
        "yunet_model_path": "yunet.onnx",
        "sface_model_path": "sface.onnx",
    }
    values.update(overrides)
    return Settings(**values)


def _png_bytes(size=(12, 10)):
    buffer = BytesIO()
    Image.new("RGB", size, (80, 120, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def _face(index=0):
    row = tuple(float(value) for value in [1, 1, 5, 5, *([3, 3] * 5), 0.9])
    return FaceDetection(row, (1.0, 1.0, 5.0, 5.0), 0.9, index)


def _analysis(count=1, rejected=0, vector=None):
    detections = tuple(_face(index) for index in range(count))
    embeddings = tuple(
        FaceEmbedding(
            detection=detections[index],
            embedding=vector.copy() if vector is not None else np.eye(1, 128, index, dtype=np.float32)[0],
        )
        for index in range(count)
    )
    return FaceAnalysisResult(DetectionResult(detections, rejected), embeddings)


class FakeEngine:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.calls = []

    def analyze(self, image):
        self.calls.append(image.shape)
        if self.outputs:
            output = self.outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return output
        return _analysis()


def _builder(engine, settings=None):
    active_settings = settings or _settings()
    store = GallerySessionStore(active_settings)
    return GalleryBuilder(active_settings, engine, store), store


def test_valid_gallery_is_created_only_after_one_face_and_normalized_embedding_per_image():
    source_vector = np.eye(1, 128, 0, dtype=np.float32)[0]
    engine = FakeEngine([_analysis(vector=source_vector), _analysis()])
    builder, store = _builder(engine)

    info = builder.build(
        [ReferenceImage("Asha.png", _png_bytes()), ReferenceImage("Ravi.png", _png_bytes())]
    )

    assert info.identity_count == 2
    assert info.identity_labels == ("Asha", "Ravi")
    assert info.expires_at > info.created_at
    assert len(info.session_id) >= 32
    with store._borrow(info.session_id) as entries:
        assert [entry.identity_key for entry in entries] == ["asha", "ravi"]
        assert all(entry.embedding.shape == (128,) for entry in entries)
        assert all(entry.embedding.dtype == np.float32 for entry in entries)
        assert not np.shares_memory(entries[0].embedding, source_vector)
        assert np.linalg.norm(source_vector) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("analysis", "code"),
    [
        (_analysis(count=0), "zero_faces"),
        (_analysis(count=2), "multiple_faces"),
        (_analysis(count=1, rejected=1), "invalid_face_detection"),
        (_analysis(count=1, vector=np.zeros(128, np.float32)), "invalid_embedding"),
    ],
)
def test_invalid_reference_face_analysis_is_rejected(analysis, code):
    builder, store = _builder(FakeEngine([analysis]))

    with pytest.raises(GalleryBuildError) as error:
        builder.build([ReferenceImage("One.png", _png_bytes())])

    assert error.value.code == code
    assert store.active_session_count == 0


def test_decode_and_embedding_failures_abort_gallery_without_publishing_a_session():
    builder, store = _builder(FakeEngine([_analysis(), EmbeddingGenerationError("feature failed", detection_index=0)]))

    with pytest.raises(GalleryBuildError) as error:
        builder.build(
            [ReferenceImage("First.png", _png_bytes()), ReferenceImage("Second.png", _png_bytes())]
        )

    assert error.value.code == "face_processing_failed"
    assert error.value.filename == "Second.png"
    assert store.active_session_count == 0

    builder, store = _builder(FakeEngine())
    with pytest.raises(GalleryBuildError) as error:
        builder.build([ReferenceImage("Broken.png", b"not an image")])
    assert error.value.code == "invalid_reference_image"
    assert store.active_session_count == 0


def test_duplicate_identity_is_rejected_before_face_processing():
    engine = FakeEngine()
    builder, store = _builder(engine)

    with pytest.raises(GalleryBuildError) as error:
        builder.build([ReferenceImage("Akshaya.jpg", _png_bytes()), ReferenceImage("akshaya.png", _png_bytes())])

    assert error.value.code == "duplicate_identity"
    assert engine.calls == []
    assert store.active_session_count == 0


def test_identity_and_aggregate_byte_limits_are_checked_before_processing():
    engine = FakeEngine()
    settings = _settings(max_gallery_upload_bytes=1000)
    builder, store = _builder(engine, settings)

    too_many = [ReferenceImage(f"Person_{index}.png", b"x") for index in range(51)]
    with pytest.raises(GalleryBuildError, match="identity limit"):
        builder.build(too_many)
    assert engine.calls == []

    max_total = len(_png_bytes()) - 1
    limited_settings = _settings(max_gallery_upload_bytes=max_total)
    limited_builder, limited_store = _builder(engine, limited_settings)
    with pytest.raises(GalleryBuildError) as error:
        limited_builder.build([ReferenceImage("A.png", _png_bytes())])
    assert error.value.code == "gallery_too_large"
    assert engine.calls == []
    assert limited_store.active_session_count == 0


def test_gallery_supports_exactly_configured_maximum_of_50_identities():
    engine = FakeEngine()
    settings = _settings(max_gallery_upload_bytes=100 * 1024 * 1024)
    builder, store = _builder(engine, settings)
    refs = [ReferenceImage(f"Person_{index:02d}.png", _png_bytes()) for index in range(50)]

    info = builder.build(refs)

    assert info.identity_count == 50
    assert len(engine.calls) == 50
    store.close()
