from pathlib import Path

import pytest
from pydantic import ValidationError

from offline_face_recognition.core.config import Settings, repository_root


def test_default_model_paths_are_repo_relative_not_cwd_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    settings = Settings.from_environment(environ={})

    assert settings.project_root == repository_root()
    assert settings.yunet_model_path == repository_root() / "models/face_detection_yunet_2026may.onnx"
    assert settings.sface_model_path == repository_root() / "models/sface/face_recognition_sface_2021dec.onnx"
    assert settings.yunet_model_path.is_absolute()
    assert settings.sface_model_path.is_absolute()


def test_relative_model_overrides_resolve_from_configured_project_root(tmp_path):
    settings = Settings.from_environment(
        environ={
            "OFFLINE_FACE_PROJECT_ROOT": str(tmp_path),
            "OFFLINE_FACE_YUNET_MODEL": "custom/yunet.onnx",
            "OFFLINE_FACE_SFACE_MODEL": "custom/sface.onnx",
        }
    )

    assert settings.yunet_model_path == (tmp_path / "custom/yunet.onnx").resolve()
    assert settings.sface_model_path == (tmp_path / "custom/sface.onnx").resolve()


@pytest.mark.parametrize(
    "env",
    [
        {"OFFLINE_FACE_DETECTION_SCORE_THRESHOLD": "1.01"},
        {"OFFLINE_FACE_DETECTION_SCORE_THRESHOLD": "-0.01"},
        {"OFFLINE_FACE_DETECTION_NMS_THRESHOLD": "-0.2"},
        {"OFFLINE_FACE_DETECTION_TOP_K": "0"},
        {"OFFLINE_FACE_MAX_UPLOAD_BYTES": "0"},
        {"OFFLINE_FACE_MAX_IMAGE_WIDTH": "0"},
        {"OFFLINE_FACE_MAX_IMAGE_HEIGHT": "0"},
        {"OFFLINE_FACE_MAX_IMAGE_PIXELS": "0"},
        {"OFFLINE_FACE_MAX_GALLERY_IDENTITIES": "51"},
        {"OFFLINE_FACE_MAX_ACTIVE_GALLERY_SESSIONS": "0"},
        {"OFFLINE_FACE_MAX_GALLERY_UPLOAD_BYTES": "0"},
        {"OFFLINE_FACE_GALLERY_SESSION_TTL_SECONDS": "0"},
        {"OFFLINE_FACE_MAX_CONCURRENT_PROCESSING": "0"},
        {"OFFLINE_FACE_MAX_ANNOTATED_RESPONSE_BYTES": "0"},
        {"OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD": "0.5"},
        {"OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN": "0.1"},
        {
            "OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD": "1.1",
            "OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN": "0.1",
        },
        {
            "OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD": "0.5",
            "OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN": "2.1",
        },
        {"OFFLINE_FACE_UNSUPPORTED_SETTING": "value"},
    ],
)
def test_invalid_or_unknown_settings_are_rejected(env):
    with pytest.raises(ValidationError):
        Settings.from_environment(environ=env)


def test_provisional_yunet_settings_are_the_approved_values():
    settings = Settings.from_environment(environ={})

    assert settings.detection_score_threshold == 0.85
    assert settings.detection_nms_threshold == 0.30
    assert settings.detection_top_k == 5000
    assert settings.matching_similarity_threshold is None
    assert settings.matching_ambiguity_margin is None


def test_image_limits_have_configurable_defaults_and_environment_overrides():
    defaults = Settings.from_environment(environ={})
    assert defaults.max_upload_bytes == 20 * 1024 * 1024
    assert defaults.max_image_width == 12_000
    assert defaults.max_image_height == 12_000
    assert defaults.max_image_pixels == 40_000_000

    configured = Settings.from_environment(
        environ={
            "OFFLINE_FACE_MAX_UPLOAD_BYTES": "1024",
            "OFFLINE_FACE_MAX_IMAGE_WIDTH": "640",
            "OFFLINE_FACE_MAX_IMAGE_HEIGHT": "480",
            "OFFLINE_FACE_MAX_IMAGE_PIXELS": "300000",
        }
    )
    assert configured.max_upload_bytes == 1024
    assert configured.max_image_width == 640
    assert configured.max_image_height == 480
    assert configured.max_image_pixels == 300_000


def test_gallery_session_and_threshold_defaults_and_overrides():
    defaults = Settings.from_environment(environ={})
    assert defaults.max_gallery_identities == 50
    assert defaults.max_active_gallery_sessions == 5
    assert defaults.max_gallery_upload_bytes == 100 * 1024 * 1024
    assert defaults.gallery_session_ttl_seconds == 1800
    assert defaults.max_concurrent_processing == 1
    assert defaults.max_annotated_response_bytes == 25 * 1024 * 1024
    assert defaults.matching_similarity_threshold is None
    assert defaults.matching_ambiguity_margin is None

    configured = Settings.from_environment(
        environ={
            "OFFLINE_FACE_MAX_GALLERY_IDENTITIES": "25",
            "OFFLINE_FACE_MAX_ACTIVE_GALLERY_SESSIONS": "3",
            "OFFLINE_FACE_MAX_GALLERY_UPLOAD_BYTES": "1048576",
            "OFFLINE_FACE_GALLERY_SESSION_TTL_SECONDS": "900",
            "OFFLINE_FACE_MAX_CONCURRENT_PROCESSING": "2",
            "OFFLINE_FACE_MAX_ANNOTATED_RESPONSE_BYTES": "1048576",
            "OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD": "0.61",
            "OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN": "0.12",
        }
    )
    assert configured.max_gallery_identities == 25
    assert configured.max_active_gallery_sessions == 3
    assert configured.max_gallery_upload_bytes == 1_048_576
    assert configured.gallery_session_ttl_seconds == 900
    assert configured.max_concurrent_processing == 2
    assert configured.max_annotated_response_bytes == 1_048_576
    assert configured.matching_similarity_threshold == 0.61
    assert configured.matching_ambiguity_margin == 0.12
