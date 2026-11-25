"""Gallery and group-photo HTTP routes."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from typing import AsyncIterator

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.datastructures import UploadFile

from offline_face_recognition.api.schemas import (
    BoundingBoxResponse,
    GalleryCreateResponse,
    GalleryInfoResponse,
    GroupPhotoRecognitionResponse,
    IdentificationResponse,
)
from offline_face_recognition.gallery.builder import (
    GalleryBuildError,
    GalleryValidationFailure,
    ReferenceImage,
)
from offline_face_recognition.gallery.service import GalleryService
from offline_face_recognition.gallery.session_store import (
    GallerySessionCapacityError,
    GallerySessionStoreClosedError,
    GallerySessionUnavailableError,
)
from offline_face_recognition.matching.matcher import MatchStatus
from offline_face_recognition.models.runtime import ModelRuntime
from offline_face_recognition.processing.group_photo import (
    GroupPhotoProcessingError,
    GroupPhotoResult,
    GroupPhotoProcessor,
)

router = APIRouter(prefix="/api", tags=["application"])
_bearer = HTTPBearer(auto_error=False)
_CHUNK_SIZE = 64 * 1024
_JSON_RESPONSE_RESERVE_BYTES = 8 * 1024


def _service(request: Request) -> GalleryService:
    service = request.app.state.gallery_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "models_not_ready", "message": "Local face models are unavailable"},
        )
    return service


def _processor(request: Request) -> GroupPhotoProcessor:
    processor = request.app.state.group_photo_processor
    if processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "models_not_ready", "message": "Local face models are unavailable"},
        )
    return processor


def _session_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "gallery_session_required", "message": "A gallery session is required"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@asynccontextmanager
async def _processing_slot(request: Request) -> AsyncIterator[None]:
    semaphore: asyncio.Semaphore = request.app.state.processing_semaphore
    if semaphore.locked():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "processing_busy", "message": "A local image-processing job is already running"},
            headers={"Retry-After": "1"},
        )
    await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


async def _run_sync_shielded(function, *args, **kwargs):
    """Keep the processing slot until an already-started worker thread exits."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise


async def _read_upload(upload: UploadFile, limit: int, *, limit_code: str) -> bytes:
    if upload.size is not None and upload.size > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": limit_code, "message": "Uploaded file exceeds the configured size limit"},
        )
    chunks = bytearray()
    while True:
        chunk = await upload.read(_CHUNK_SIZE)
        if not chunk:
            break
        if len(chunks) + len(chunk) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={"code": limit_code, "message": "Uploaded file exceeds the configured size limit"},
            )
        chunks.extend(chunk)
    return bytes(chunks)


def _map_gallery_error(exc: Exception) -> HTTPException:
    if isinstance(exc, GallerySessionCapacityError):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "gallery_session_capacity", "message": str(exc)},
        )
    if isinstance(exc, GallerySessionUnavailableError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "gallery_session_unavailable", "message": "Gallery session is unavailable"},
        )
    if isinstance(exc, GallerySessionStoreClosedError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "gallery_store_closed", "message": "Gallery service is unavailable"},
        )
    if isinstance(exc, GalleryBuildError):
        code = exc.code
        status_code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if code == "gallery_too_large"
            else status.HTTP_422_UNPROCESSABLE_CONTENT
        )
        return HTTPException(
            status_code=status_code,
            detail={
                "code": code,
                "message": str(exc),
                **({"filename": exc.filename} if exc.filename is not None else {}),
                **(
                    {
                        "invalid_image_count": exc.invalid_image_count,
                        "errors": [
                            {
                                "filename": failure.filename,
                                "code": failure.code,
                                "message": failure.message,
                            }
                            for failure in exc.failures
                        ],
                    }
                    if exc.failures
                    else {}
                ),
            },
        )
    raise exc


def _multipart_form(request: Request, *, max_files: int):
    return request.form(max_files=max_files, max_fields=2, max_part_size=16 * 1024)


@router.post(
    "/galleries",
    response_model=GalleryCreateResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["reference_images"],
                        "properties": {
                            "reference_images": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            }
                        },
                    }
                }
            },
        }
    },
)
async def create_gallery(request: Request) -> GalleryCreateResponse:
    service = _service(request)
    settings = request.app.state.settings
    references: list[ReferenceImage] = []
    upload_failures: list[GalleryValidationFailure] = []
    aggregate_size = 0
    async with _processing_slot(request):
        async with _multipart_form(request, max_files=settings.max_gallery_identities) as form:
            uploads = form.getlist("reference_images")
            if not uploads or any(not isinstance(upload, UploadFile) for upload in uploads):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={"code": "reference_images_required", "message": "Upload reference_images files"},
                )
            if len(uploads) > settings.max_gallery_identities:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={"code": "too_many_identities", "message": "Gallery exceeds the identity limit"},
                )
            if all(upload.size is not None for upload in uploads):
                declared_total = sum(upload.size or 0 for upload in uploads)
                if declared_total > settings.max_gallery_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={"code": "gallery_too_large", "message": "Gallery exceeds the aggregate upload limit"},
                    )
            try:
                for upload in uploads:
                    filename = upload.filename or ""
                    if upload.size is not None and upload.size > settings.max_upload_bytes:
                        upload_failures.append(
                            GalleryValidationFailure(
                                filename,
                                "reference_image_too_large",
                                "Reference image exceeds the individual upload size limit",
                            )
                        )
                        continue
                    remaining = settings.max_gallery_upload_bytes - aggregate_size
                    if remaining <= 0:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail={"code": "gallery_too_large", "message": "Gallery exceeds the aggregate upload limit"},
                        )
                    limit = min(settings.max_upload_bytes, remaining)
                    try:
                        content = await _read_upload(upload, limit, limit_code="gallery_too_large")
                    except HTTPException as exc:
                        if (
                            limit == settings.max_upload_bytes
                            and isinstance(exc.detail, dict)
                            and exc.detail.get("code") == "gallery_too_large"
                        ):
                            upload_failures.append(
                                GalleryValidationFailure(
                                    filename,
                                    "reference_image_too_large",
                                    "Reference image exceeds the individual upload size limit",
                                )
                            )
                            continue
                        raise
                    aggregate_size += len(content)
                    references.append(ReferenceImage(filename, content))
            finally:
                # FormData also closes these; close explicitly before CPU inference begins.
                await asyncio.gather(*(upload.close() for upload in uploads if isinstance(upload, UploadFile)))

        try:
            info = await _run_sync_shielded(
                service.create_gallery,
                references,
                validation_failures=upload_failures,
            )
        except (GalleryBuildError, GallerySessionCapacityError, GallerySessionStoreClosedError) as exc:
            raise _map_gallery_error(exc) from exc
        finally:
            references.clear()

    return GalleryCreateResponse(
        session_id=info.session_id,
        created_at=info.created_at,
        expires_at=info.expires_at,
        identity_count=info.identity_count,
        identity_labels=list(info.identity_labels),
    )


@router.get("/galleries/current", response_model=GalleryInfoResponse)
def get_current_gallery(request: Request, session_id: str = Depends(_session_id)) -> GalleryInfoResponse:
    try:
        info = _service(request).get_gallery_info(session_id)
    except (GallerySessionUnavailableError, GallerySessionStoreClosedError) as exc:
        raise _map_gallery_error(exc) from exc
    return GalleryInfoResponse(
        created_at=info.created_at,
        expires_at=info.expires_at,
        identity_count=info.identity_count,
        identity_labels=list(info.identity_labels),
    )


@router.delete("/galleries/current", status_code=status.HTTP_204_NO_CONTENT)
def delete_current_gallery(request: Request, session_id: str = Depends(_session_id)) -> Response:
    try:
        deleted = _service(request).delete_gallery(session_id)
    except (GallerySessionUnavailableError, GallerySessionStoreClosedError) as exc:
        raise _map_gallery_error(exc) from exc
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "gallery_session_unavailable", "message": "Gallery session is unavailable"},
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _encode_annotated_jpeg(image_bgr) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode annotated image")
    return encoded.tobytes()


class AnnotatedResponseTooLarge(ValueError):
    """The encoded JSON response exceeds the configured in-memory response cap."""


def _serialize_recognition_result(
    result: GroupPhotoResult,
    max_response_bytes: int,
) -> str:
    encoded = _encode_annotated_jpeg(result.annotated_image_bgr)
    raw_image_limit = max(1, ((max_response_bytes - _JSON_RESPONSE_RESERVE_BYTES) * 3) // 4)
    if len(encoded) > raw_image_limit:
        raise AnnotatedResponseTooLarge
    identification_items = [
        IdentificationResponse(
            detection_index=item.detection_index,
            bounding_box=BoundingBoxResponse(
                x=item.box[0], y=item.box[1], width=item.box[2], height=item.box[3]
            ),
            status=item.status.value,
            identity_label=item.identity_label if item.status is MatchStatus.KNOWN else None,
        )
        for item in result.identifications
    ]
    body = GroupPhotoRecognitionResponse(
        face_count=len(identification_items),
        annotated_image_base64=base64.b64encode(encoded).decode("ascii"),
        identifications=identification_items,
    ).model_dump_json()
    if len(body.encode("utf-8")) > max_response_bytes:
        raise AnnotatedResponseTooLarge
    return body


@router.post(
    "/group-photos/recognize",
    response_model=GroupPhotoRecognitionResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["photo"],
                        "properties": {"photo": {"type": "string", "format": "binary"}},
                    }
                }
            },
        }
    },
)
async def recognize_group_photo(request: Request, session_id: str = Depends(_session_id)) -> Response:
    runtime: ModelRuntime = request.app.state.model_runtime
    if not runtime.readiness.models_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "models_not_ready", "message": "Local face models are unavailable"},
        )
    try:
        _service(request).get_gallery_info(session_id)
    except (GallerySessionUnavailableError, GallerySessionStoreClosedError) as exc:
        raise _map_gallery_error(exc) from exc
    if not runtime.experimental_recognition_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "recognition_not_ready",
                "message": "Experimental recognition requires explicit opt-in and both matching thresholds",
            },
        )

    settings = request.app.state.settings
    processor = _processor(request)
    async with _processing_slot(request):
        async with _multipart_form(request, max_files=1) as form:
            upload = form.get("photo")
            if not isinstance(upload, UploadFile):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={"code": "photo_required", "message": "Upload one group-photo file as photo"},
                )
            filename = upload.filename or ""
            try:
                image_bytes = await _read_upload(
                    upload, settings.max_upload_bytes, limit_code="group_photo_too_large"
                )
            finally:
                await upload.close()

        result = None
        try:
            try:
                result = await _run_sync_shielded(
                    processor.process, session_id, filename, image_bytes
                )
            except GallerySessionUnavailableError as exc:
                raise _map_gallery_error(exc) from exc
            except GroupPhotoProcessingError as exc:
                if exc.code == "invalid_group_photo":
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail={"code": exc.code, "message": str(exc)},
                    ) from exc
                if exc.code == "matching_not_ready_or_failed":
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail={"code": "recognition_not_ready", "message": "Recognition is unavailable"},
                    ) from exc
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"code": exc.code, "message": "Group-photo processing failed"},
                ) from exc

            try:
                body = await _run_sync_shielded(
                    _serialize_recognition_result,
                    result,
                    settings.max_annotated_response_bytes,
                )
            except AnnotatedResponseTooLarge as exc:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "code": "annotated_response_too_large",
                        "message": "Annotated response exceeds the configured size limit",
                    },
                ) from exc
            return Response(content=body, media_type="application/json", status_code=200)
        finally:
            image_bytes = b""
            if result is not None:
                result.annotated_image_bgr.fill(0)
