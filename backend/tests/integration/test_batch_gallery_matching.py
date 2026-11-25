import numpy as np

from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.detection import FaceDetection
from offline_face_recognition.faces.engine import FaceEmbedding
from offline_face_recognition.gallery.service import GalleryService
from offline_face_recognition.gallery.session_store import GallerySessionStore
from offline_face_recognition.matching.matcher import FaceMatcher, MatchPolicy, MatchStatus


def _settings():
    return Settings(
        project_root=".",
        yunet_model_path="yunet.onnx",
        sface_model_path="sface.onnx",
    )


def _reference(index):
    return np.eye(1, 128, index, dtype=np.float32)[0]


def _query(scores):
    vector = np.zeros(128, dtype=np.float32)
    vector[: len(scores)] = np.asarray(scores, dtype=np.float32)
    remainder = 1.0 - float(np.dot(vector, vector))
    vector[len(scores)] = np.sqrt(remainder)
    return vector


def _face(index, embedding):
    raw_row = tuple(float(value) for value in [10, 12, 20, 24, *([15, 18] * 5), 0.95])
    detection = FaceDetection(
        raw_row=raw_row,
        clipped_box=(10.0, 12.0, 20.0, 24.0),
        score=0.95,
        detection_index=index,
        original_box=(10.0, 12.0, 20.0, 24.0),
    )
    return FaceEmbedding(detection, embedding)


def test_gallery_service_assigns_unique_identities_and_uses_valid_alternative():
    settings = _settings()
    store = GallerySessionStore(settings)
    service = GalleryService(object(), store, FaceMatcher(settings))
    gallery = store.create(
        {
            "akki": ("Akki", _reference(0)),
            "bela": ("Bela", _reference(1)),
            "chet": ("Chet", _reference(2)),
        }
    )
    faces = [
        _face(5, _query([0.72, 0.40, 0.10])),
        _face(2, _query([0.65, 0.60, 0.20])),
    ]

    try:
        decisions = service.match_faces(
            gallery.session_id,
            faces,
            policy=MatchPolicy(0.40, 0.10),
        )

        assert [decision.detection_index for decision in decisions] == [5, 2]
        assert [decision.status for decision in decisions] == [MatchStatus.KNOWN, MatchStatus.KNOWN]
        assert [decision.identity_label for decision in decisions] == ["Akki", "Bela"]
        assert len({decision.identity_key for decision in decisions}) == len(decisions)
        assert all(decision.calibrated is False for decision in decisions)
    finally:
        service.close()
