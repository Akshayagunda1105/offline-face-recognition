"""Group-photo recognition orchestration, separate from HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offline_face_recognition.annotation.image_annotator import Annotation, ImageAnnotator
from offline_face_recognition.annotation.image_annotator import ImageAnnotationError
from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.engine import FaceAnalysisEngine
from offline_face_recognition.gallery.service import GalleryService
from offline_face_recognition.gallery.session_store import GallerySessionUnavailableError
from offline_face_recognition.imaging.decode import ImageValidationError, decode_image
from offline_face_recognition.matching.matcher import MatchPolicy, MatchStatus, MatchingError


class GroupPhotoProcessingError(RuntimeError):
    """The complete group-photo workflow failed and no partial result is returned."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class Identification:
    detection_index: int
    box: tuple[float, float, float, float]
    status: MatchStatus
    identity_label: str | None


@dataclass(frozen=True)
class GroupPhotoResult:
    annotated_image_bgr: np.ndarray
    identifications: tuple[Identification, ...]


class GroupPhotoProcessor:
    """Compose decoding, face engine, gallery matching and image annotation."""

    def __init__(
        self,
        settings: Settings,
        face_engine: FaceAnalysisEngine,
        gallery_service: GalleryService,
        annotator: ImageAnnotator | None = None,
        *,
        experimental_policy: MatchPolicy | None = None,
    ) -> None:
        self._settings = settings
        self._face_engine = face_engine
        self._gallery_service = gallery_service
        self._annotator = annotator or ImageAnnotator()
        # This option is for internal tests/evaluation only. API routes never set it.
        self._experimental_policy = experimental_policy

    def process(self, session_id: str, filename: str, image_bytes: bytes) -> GroupPhotoResult:
        self._gallery_service.get_gallery_info(session_id)

        image_bgr: np.ndarray | None = None
        analysis = None
        try:
            try:
                image_bgr = decode_image(image_bytes, filename, self._settings)
            except ImageValidationError as exc:
                raise GroupPhotoProcessingError(
                    str(exc), code="invalid_group_photo"
                ) from exc

            try:
                analysis = self._face_engine.analyze(image_bgr)
            except Exception as exc:
                raise GroupPhotoProcessingError(
                    "Group-photo face processing failed", code="face_processing_failed"
                ) from exc
            if analysis.detections.rejected_detection_count:
                raise GroupPhotoProcessingError(
                    "Group photo contains invalid face detections", code="invalid_face_detections"
                )
            expected_indices = tuple(
                detection.detection_index for detection in analysis.detections.detections
            )
            face_indices = tuple(face.detection.detection_index for face in analysis.faces)
            if len(analysis.faces) != len(analysis.detections.detections) or face_indices != expected_indices:
                raise GroupPhotoProcessingError(
                    "Detected faces and embeddings are incomplete or misaligned",
                    code="face_embedding_misaligned",
                )
            if any(face.detection.original_box is None for face in analysis.faces):
                raise GroupPhotoProcessingError(
                    "A detected face is missing original-image coordinates",
                    code="original_coordinates_missing",
                )

            decisions = self._gallery_service.match_faces(
                session_id,
                analysis.faces,
                policy=self._experimental_policy,
            )
            if len(decisions) != len(analysis.faces):
                raise GroupPhotoProcessingError(
                    "Matching did not produce one result per detected face", code="matching_incomplete"
                )

            identifications = tuple(
                Identification(
                    detection_index=decision.detection_index,
                    box=face.detection.original_box,
                    status=decision.status,
                    identity_label=(
                        decision.identity_label
                        if decision.status is MatchStatus.KNOWN
                        else None
                    ),
                )
                for face, decision in zip(analysis.faces, decisions, strict=True)
            )
            if any(
                face.detection.detection_index != identification.detection_index
                for face, identification in zip(analysis.faces, identifications, strict=True)
            ):
                raise GroupPhotoProcessingError(
                    "Face results lost YuNet detection ordering", code="matching_misaligned"
                )

            annotations = tuple(
                Annotation(
                    detection_index=result.detection_index,
                    box=result.box,
                    label=result.identity_label or MatchStatus.UNKNOWN.value,
                )
                for result in identifications
            )
            try:
                annotated = self._annotator.annotate(image_bgr, annotations)
            except ImageAnnotationError as exc:
                raise GroupPhotoProcessingError(
                    "Group-photo annotation failed", code="annotation_failed"
                ) from exc
            return GroupPhotoResult(annotated, identifications)
        except MatchingError as exc:
            raise GroupPhotoProcessingError(
                "Recognition thresholds are unavailable or matching failed",
                code="matching_not_ready_or_failed",
            ) from exc
        finally:
            if analysis is not None:
                for face in analysis.faces:
                    if face.embedding.flags.writeable:
                        face.embedding.fill(0.0)
            if image_bgr is not None:
                image_bgr.fill(0)
