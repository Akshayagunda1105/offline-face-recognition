"""Liveness and independently reported readiness endpoints."""

from typing import Literal

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, Field

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
    models_ready: bool = Field(description="Both local face models loaded successfully")
    recognition_ready: Literal[False] = Field(
        default=False, description="Calibrated recognition is not enabled by this application"
    )
    experimental_recognition_available: bool = Field(
        description="Models, explicit opt-in, and both matching thresholds are present"
    )
    experimental: bool = Field(description="Experimental-recognition opt-in setting is enabled")
    calibrated: Literal[False] = Field(default=False, description="Experimental decisions are never calibrated")
    reason: Literal[
        "models_not_ready",
        "experimental_opt_in_disabled",
        "matching_thresholds_not_configured",
    ] | None = None


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
def recognition_readiness(request: Request) -> RecognitionReadinessResponse:
    runtime = _runtime(request)
    models_ready = runtime.readiness.models_ready
    experimental = runtime.settings.experimental_recognition_enabled
    if runtime.experimental_recognition_available:
        reason = None
    elif not models_ready:
        reason = "models_not_ready"
    elif not experimental:
        reason = "experimental_opt_in_disabled"
    else:
        reason = "matching_thresholds_not_configured"
    return RecognitionReadinessResponse(
        models_ready=models_ready,
        recognition_ready=False,
        experimental_recognition_available=runtime.experimental_recognition_available,
        experimental=experimental,
        calibrated=False,
        reason=reason,
    )
