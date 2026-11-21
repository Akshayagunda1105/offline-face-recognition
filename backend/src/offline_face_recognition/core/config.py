"""Typed, validated settings for the local backend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


def repository_root() -> Path:
    """Return the repository root independently of the process working directory."""
    return Path(__file__).resolve().parents[4]


def _resolve_configured_path(value: str | Path | None, *, root: Path, default: Path) -> Path:
    path = Path(value).expanduser() if value else root / default
    if not path.is_absolute():
        path = root / path
    return path.resolve()


class Settings(BaseModel):
    """Validated runtime settings; matching thresholds stay unset until supplied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_root: Path
    yunet_model_path: Path
    sface_model_path: Path

    detection_score_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    detection_nms_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    detection_top_k: PositiveInt = 5000
    max_upload_bytes: PositiveInt = 20 * 1024 * 1024
    max_image_width: PositiveInt = 12_000
    max_image_height: PositiveInt = 12_000
    max_image_pixels: PositiveInt = 40_000_000
    max_detection_dimension: PositiveInt = 1200
    max_gallery_identities: int = Field(default=50, ge=1, le=50)
    max_active_gallery_sessions: PositiveInt = 5
    max_gallery_upload_bytes: PositiveInt = 100 * 1024 * 1024
    gallery_session_ttl_seconds: PositiveInt = 30 * 60
    max_concurrent_processing: PositiveInt = 1
    max_annotated_response_bytes: PositiveInt = 25 * 1024 * 1024
    matching_similarity_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    matching_ambiguity_margin: float | None = Field(default=None, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_matching_settings_pair(self) -> Settings:
        if (self.matching_similarity_threshold is None) != (
            self.matching_ambiguity_margin is None
        ):
            raise ValueError(
                "Matching similarity threshold and ambiguity margin must be set together"
            )
        return self

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
            "OFFLINE_FACE_MAX_UPLOAD_BYTES",
            "OFFLINE_FACE_MAX_IMAGE_WIDTH",
            "OFFLINE_FACE_MAX_IMAGE_HEIGHT",
            "OFFLINE_FACE_MAX_IMAGE_PIXELS",
            "OFFLINE_FACE_MAX_DETECTION_DIMENSION",
            "OFFLINE_FACE_MAX_GALLERY_IDENTITIES",
            "OFFLINE_FACE_MAX_ACTIVE_GALLERY_SESSIONS",
            "OFFLINE_FACE_MAX_GALLERY_UPLOAD_BYTES",
            "OFFLINE_FACE_GALLERY_SESSION_TTL_SECONDS",
            "OFFLINE_FACE_MAX_CONCURRENT_PROCESSING",
            "OFFLINE_FACE_MAX_ANNOTATED_RESPONSE_BYTES",
            "OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD",
            "OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN",
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
            "max_upload_bytes": env.get("OFFLINE_FACE_MAX_UPLOAD_BYTES", 20 * 1024 * 1024),
            "max_image_width": env.get("OFFLINE_FACE_MAX_IMAGE_WIDTH", 12_000),
            "max_image_height": env.get("OFFLINE_FACE_MAX_IMAGE_HEIGHT", 12_000),
            "max_image_pixels": env.get("OFFLINE_FACE_MAX_IMAGE_PIXELS", 40_000_000),
            "max_detection_dimension": env.get("OFFLINE_FACE_MAX_DETECTION_DIMENSION", 1200),
            "max_gallery_identities": env.get("OFFLINE_FACE_MAX_GALLERY_IDENTITIES", 50),
            "max_active_gallery_sessions": env.get("OFFLINE_FACE_MAX_ACTIVE_GALLERY_SESSIONS", 5),
            "max_gallery_upload_bytes": env.get(
                "OFFLINE_FACE_MAX_GALLERY_UPLOAD_BYTES", 100 * 1024 * 1024
            ),
            "gallery_session_ttl_seconds": env.get("OFFLINE_FACE_GALLERY_SESSION_TTL_SECONDS", 1800),
            "max_concurrent_processing": env.get("OFFLINE_FACE_MAX_CONCURRENT_PROCESSING", 1),
            "max_annotated_response_bytes": env.get(
                "OFFLINE_FACE_MAX_ANNOTATED_RESPONSE_BYTES", 25 * 1024 * 1024
            ),
            "matching_similarity_threshold": env.get("OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD"),
            "matching_ambiguity_margin": env.get("OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN"),
        }
        # Let Pydantic produce the same typed extra-field error used for the
        # rest of the settings contract when an OFFLINE_FACE_* name is mistyped.
        settings_values.update({name: env[name] for name in unknown_names})
        return cls.model_validate(settings_values)
