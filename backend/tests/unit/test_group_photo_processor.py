from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.detection import DetectionResult, FaceDetection
from offline_face_recognition.faces.engine import FaceAnalysisResult, FaceEmbedding
from offline_face_recognition.gallery.service import GalleryService
from offline_face_recognition.gallery.session_store import (
    GallerySessionStore,
    GallerySessionUnavailableError,
)
from offline_face_recognition.matching.matcher import FaceMatcher, MatchPolicy, MatchStatus
from offline_face_recognition.processing.group_photo import (
    GroupPhotoProcessingError,
    GroupPhotoProcessor,
)


def _settings():
    return Settings(project_root=".", yunet_model_path="yunet.onnx", sface_model_path="sface.onnx")


def _png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (64, 48), (20, 60, 100)).save(buffer, format="PNG")
    return buffer.getvalue()


def _detection(index=0):
    row = tuple(float(value) for value in [10, 10, 20, 20, *([15, 15] * 5), 0.9])
    return FaceDetection(
        row,
        (10.0, 10.0, 20.0, 20.0),
        0.9,
        index,
        original_box=(10.0, 10.0, 20.0, 20.0),
    )


class FakeEngine:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def analyze(self, image):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def _analysis(vectors, *, rejected=0, indexes=None):
    indexes = indexes or list(range(len(vectors)))
    detections = tuple(_detection(index) for index in indexes)
    faces = tuple(
        FaceEmbedding(detection, vector.astype(np.float32).copy())
        for detection, vector in zip(detections, vectors, strict=True)
    )
    return FaceAnalysisResult(DetectionResult(detections, rejected), faces)


def _processor(engine, *, policy=MatchPolicy(0.8, 0.0)):
    settings = _settings()
    store = GallerySessionStore(settings)
    info = store.create({"asha": ("Asha", np.eye(1, 128, 0, dtype=np.float32)[0])})
    service = GalleryService(builder=object(), session_store=store, matcher=FaceMatcher(settings))
    return (
        GroupPhotoProcessor(settings, engine, service, experimental_policy=policy),
        service,
        info.session_id,
    )


def test_processor_returns_ordered_known_and_unknown_results_with_annotated_copy():
    engine = FakeEngine(_analysis([np.eye(1, 128, 0, dtype=np.float32)[0], np.eye(1, 128, 1, dtype=np.float32)[0]], indexes=[2, 5]))
    processor, service, session_id = _processor(engine)
    source_bytes = _png_bytes()

    result = processor.process(session_id, "group.png", source_bytes)

    assert [item.detection_index for item in result.identifications] == [2, 5]
    assert [item.status for item in result.identifications] == [MatchStatus.KNOWN, MatchStatus.UNKNOWN]
    assert [item.identity_label for item in result.identifications] == ["Asha", None]
    assert result.annotated_image_bgr.shape == (48, 64, 3)
    assert np.count_nonzero(result.annotated_image_bgr) > 0
    assert source_bytes == _png_bytes()
    service.close()


def test_processor_annotates_original_coordinate_box_when_engine_resized():
    detection = _detection(4)
    detection = FaceDetection(
        detection.raw_row,
        detection.clipped_box,
        detection.score,
        detection.detection_index,
        original_box=(20.0, 24.0, 40.0, 44.0),
    )
    vector = np.eye(1, 128, 0, dtype=np.float32)[0]
    engine = FakeEngine(FaceAnalysisResult(DetectionResult((detection,)), (FaceEmbedding(detection, vector.copy()),)))
    processor, service, session_id = _processor(engine)

    result = processor.process(session_id, "group.png", _png_bytes())

    assert result.identifications[0].box == (20.0, 24.0, 40.0, 44.0)
    service.close()


def test_processor_rejects_face_without_original_coordinate_box():
    valid_detection = _detection(4)
    detection = FaceDetection(
        valid_detection.raw_row,
        valid_detection.clipped_box,
        valid_detection.score,
        valid_detection.detection_index,
    )
    vector = np.eye(1, 128, 0, dtype=np.float32)[0]
    engine = FakeEngine(
        FaceAnalysisResult(
            DetectionResult((detection,)),
            (FaceEmbedding(detection, vector.copy()),),
        )
    )
    processor, service, session_id = _processor(engine)

    with pytest.raises(GroupPhotoProcessingError) as error:
        processor.process(session_id, "group.png", _png_bytes())

    assert error.value.code == "original_coordinates_missing"
    service.close()


def test_no_faces_returns_unannotated_copy_and_empty_identification_list():
    engine = FakeEngine(FaceAnalysisResult(DetectionResult(()), ()))
    processor, service, session_id = _processor(engine, policy=None)
    source = _png_bytes()

    result = processor.process(session_id, "group.png", source)

    assert result.identifications == ()
    assert result.annotated_image_bgr.shape == (48, 64, 3)
    assert engine.calls == 1
    service.close()


def test_invalid_detection_rows_abort_instead_of_returning_partial_faces():
    engine = FakeEngine(_analysis([np.eye(1, 128, 0, dtype=np.float32)[0]], rejected=1))
    processor, service, session_id = _processor(engine)

    with pytest.raises(GroupPhotoProcessingError) as error:
        processor.process(session_id, "group.png", _png_bytes())

    assert error.value.code == "invalid_face_detections"
    service.close()


def test_invalid_photo_expired_session_and_engine_failure_are_explicit():
    engine = FakeEngine(_analysis([]))
    processor, service, session_id = _processor(engine, policy=None)
    with pytest.raises(GroupPhotoProcessingError) as error:
        processor.process(session_id, "broken.png", b"not an image")
    assert error.value.code == "invalid_group_photo"

    service.delete_gallery(session_id)
    with pytest.raises(GallerySessionUnavailableError):
        processor.process(session_id, "group.png", _png_bytes())
    service.close()


def test_matching_failure_is_not_returned_as_a_partial_result():
    engine = FakeEngine(_analysis([np.zeros(128, dtype=np.float32)]))
    processor, service, session_id = _processor(engine)

    with pytest.raises(GroupPhotoProcessingError) as error:
        processor.process(session_id, "group.png", _png_bytes())

    assert error.value.code == "matching_not_ready_or_failed"
    service.close()
