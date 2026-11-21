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
    assert not hasattr(settings, "matching_threshold")
    assert not hasattr(settings, "ambiguity_margin")
