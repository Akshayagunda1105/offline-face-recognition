from pathlib import Path

import numpy as np
import pytest

from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.engine import FaceAnalysisEngine
from offline_face_recognition.gallery.builder import GalleryBuilder, ReferenceImage
from offline_face_recognition.gallery.service import GalleryService
from offline_face_recognition.gallery.session_store import GallerySessionStore
from offline_face_recognition.imaging.decode import decode_image
from offline_face_recognition.matching.matcher import FaceMatcher, MatchPolicy
from offline_face_recognition.models.runtime import ModelRuntime

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DIR = ROOT / "synthetic_dataset" / "reference_gallery"
GROUP_PHOTO = ROOT / "synthetic_dataset" / "group_photos" / "group_01.png"
REFERENCE_PATHS = [REFERENCE_DIR / f"Person_{index:02d}.png" for index in range(1, 6)]


def test_local_synthetic_gallery_and_group_matching_pipeline():
    required = [*REFERENCE_PATHS, GROUP_PHOTO]
    if any(not path.is_file() for path in required):
        pytest.skip("Local synthetic pilot images are unavailable")

    settings = Settings.from_environment(environ={})
    runtime = ModelRuntime(settings)
    runtime.load()
    if not runtime.readiness.models_ready:
        runtime.close()
        pytest.skip(f"Local YuNet/SFace model assets unavailable ({runtime.error_code})")

    store = GallerySessionStore(settings)
    face_engine = FaceAnalysisEngine(runtime)
    service = GalleryService(
        GalleryBuilder(settings, face_engine, store),
        store,
        FaceMatcher(settings),
    )
    try:
        references = [ReferenceImage(path.name, path.read_bytes()) for path in REFERENCE_PATHS]
        gallery = service.create_gallery(references)
        assert gallery.identity_count == 5

        group_bgr = decode_image(GROUP_PHOTO.read_bytes(), GROUP_PHOTO.name, settings)
        group_analysis = face_engine.analyze(group_bgr)
        assert len(group_analysis.faces) == 8

        # This explicit policy exercises the code path only. It is not calibrated
        # and no identity correctness claim is made for the synthetic group image.
        experimental_policy = MatchPolicy(similarity_threshold=0.5, ambiguity_margin=0.05)
        decisions = [
            service.match_face(
                gallery.session_id,
                face.embedding,
                detection_index=face.detection.detection_index,
                policy=experimental_policy,
            )
            for face in group_analysis.faces
        ]

        assert [decision.detection_index for decision in decisions] == list(range(8))
        assert all(not decision.calibrated for decision in decisions)
        assert all(np.isfinite(decision.best_candidate.similarity) for decision in decisions)
        assert store.delete(gallery.session_id)
        assert store.active_session_count == 0
    finally:
        service.close()
        runtime.close()
