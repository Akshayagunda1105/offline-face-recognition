"""Liveness and independently reported readiness endpoints."""

from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel

from offline_face_recognition.models.runtime import ModelRuntime

router = APIRouter(tags=["health"])


class LivenessResponse(BaseModel):
    status: Literal["alive"] = "alive"


class ModelReadinessResponse(BaseModel):
    models_ready: bool
    yunet_ready: bool
    sface_ready: bool
    error_code: str | None = None


class RecognitionReadinessResponse(BaseModel):
    recognition_ready: Literal[False] = False
    reason: Literal["matching_thresholds_not_calibrated"] = "matching_thresholds_not_calibrated"


def _runtime(request: Request) -> ModelRuntime:
    return request.app.state.model_runtime


@router.get("/health/live", response_model=LivenessResponse)
def liveness() -> LivenessResponse:
    return LivenessResponse()


@router.get("/health/models", response_model=ModelReadinessResponse)
def model_readiness(request: Request, response: Response) -> ModelReadinessResponse:
    readiness = _runtime(request).readiness
    if not readiness.models_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ModelReadinessResponse(
        models_ready=readiness.models_ready,
        yunet_ready=readiness.yunet_ready,
        sface_ready=readiness.sface_ready,
        error_code=readiness.error_code,
    )


@router.get("/health/recognition", response_model=RecognitionReadinessResponse)
def recognition_readiness(response: Response) -> RecognitionReadinessResponse:
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return RecognitionReadinessResponse()
