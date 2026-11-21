"""Typed, validated settings for the local backend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


def repository_root() -> Path:
    """Return the repository root independently of the process working directory."""
    return Path(__file__).resolve().parents[4]


def _resolve_configured_path(value: str | Path | None, *, root: Path, default: Path) -> Path:
    path = Path(value).expanduser() if value else root / default
    if not path.is_absolute():
        path = root / path
    return path.resolve()


class Settings(BaseModel):
    """Runtime settings. Recognition thresholds are deliberately not defined here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_root: Path
    yunet_model_path: Path
    sface_model_path: Path

    detection_score_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    detection_nms_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    detection_top_k: PositiveInt = 5000

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from the explicitly supported environment variables."""
        env = os.environ if environ is None else environ
        supported_names = {
            "OFFLINE_FACE_PROJECT_ROOT",
            "OFFLINE_FACE_YUNET_MODEL",
            "OFFLINE_FACE_SFACE_MODEL",
            "OFFLINE_FACE_DETECTION_SCORE_THRESHOLD",
            "OFFLINE_FACE_DETECTION_NMS_THRESHOLD",
            "OFFLINE_FACE_DETECTION_TOP_K",
        }
        unknown_names = {
            name
            for name in env
            if name.startswith("OFFLINE_FACE_") and name not in supported_names
        }
        root = _resolve_configured_path(
            env.get("OFFLINE_FACE_PROJECT_ROOT"),
            root=repository_root(),
            default=Path("."),
        )

        settings_values = {
            "project_root": root,
            "yunet_model_path": _resolve_configured_path(
                env.get("OFFLINE_FACE_YUNET_MODEL"),
                root=root,
                default=Path("models/face_detection_yunet_2026may.onnx"),
            ),
            "sface_model_path": _resolve_configured_path(
                env.get("OFFLINE_FACE_SFACE_MODEL"),
                root=root,
                default=Path("models/sface/face_recognition_sface_2021dec.onnx"),
            ),
            "detection_score_threshold": env.get("OFFLINE_FACE_DETECTION_SCORE_THRESHOLD", 0.85),
            "detection_nms_threshold": env.get("OFFLINE_FACE_DETECTION_NMS_THRESHOLD", 0.30),
            "detection_top_k": env.get("OFFLINE_FACE_DETECTION_TOP_K", 5000),
        }
        # Let Pydantic produce the same typed extra-field error used for the
        # rest of the settings contract when an OFFLINE_FACE_* name is mistyped.
        settings_values.update({name: env[name] for name in unknown_names})
        return cls.model_validate(settings_values)
