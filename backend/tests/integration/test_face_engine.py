from pathlib import Path

import numpy as np
import pytest

from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.engine import FaceAnalysisEngine
from offline_face_recognition.imaging.decode import decode_image
from offline_face_recognition.models.runtime import ModelRuntime

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DIR = ROOT / "synthetic_dataset" / "reference_gallery"
GROUP_PHOTO = ROOT / "synthetic_dataset" / "group_photos" / "group_01.png"
REQUIRED_IMAGES = [REFERENCE_DIR / f"Person_{index:02d}.png" for index in range(1, 6)] + [GROUP_PHOTO]


@pytest.fixture(scope="module")
def face_engine():
    missing = [path for path in REQUIRED_IMAGES if not path.is_file()]
    if missing:
        pytest.skip("Local synthetic pilot images are unavailable")

    settings = Settings.from_environment(environ={})
    runtime = ModelRuntime(settings)
    runtime.load()
    if not runtime.readiness.models_ready:
        runtime.close()
        pytest.skip(f"Local YuNet/SFace model assets are unavailable ({runtime.error_code})")
    try:
        yield FaceAnalysisEngine(runtime), settings
    finally:
        runtime.close()


def test_synthetic_references_and_group_photo_detect_and_embed(face_engine):
    engine, settings = face_engine
    counts = []
    all_embeddings = []
    for path in REQUIRED_IMAGES:
        image = decode_image(path.read_bytes(), path.name, settings)
        result = engine.analyze(image)
        counts.append(len(result.detections.detections))
        assert result.detections.rejected_detection_count == 0
        assert len(result.faces) == len(result.detections.detections)
        for face in result.faces:
            assert face.embedding.shape == (128,)
            assert face.embedding.dtype == np.float32
            assert np.isfinite(face.embedding).all()
            assert np.linalg.norm(face.embedding.astype(np.float64)) == pytest.approx(1.0, abs=1e-6)
            all_embeddings.append(face.embedding)

    assert counts == [1, 1, 1, 1, 1, 8]
    assert len(all_embeddings) == 13
