"""HTTP request and response models for the local API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class GalleryCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    created_at: datetime
    expires_at: datetime
    identity_count: int
    identity_labels: list[str]


class GalleryInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    expires_at: datetime
    identity_count: int
    identity_labels: list[str]


class BoundingBoxResponse(BaseModel):
    x: float
    y: float
    width: float
    height: float


class IdentificationResponse(BaseModel):
    detection_index: int
    bounding_box: BoundingBoxResponse
    status: Literal["Known", "Unknown"]
    identity_label: str | None


class GroupPhotoRecognitionResponse(BaseModel):
    face_count: int
    annotated_image_media_type: Literal["image/jpeg"] = "image/jpeg"
    annotated_image_base64: str
    identifications: list[IdentificationResponse]
