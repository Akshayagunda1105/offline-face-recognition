"""Transactional construction of one-reference-per-identity galleries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from offline_face_recognition.core.config import Settings
from offline_face_recognition.core.embeddings import (
    EmbeddingValidationError,
    validate_normalized_embedding,
)
from offline_face_recognition.faces.engine import FaceAnalysisEngine
from offline_face_recognition.gallery.identity import (
    DuplicateIdentityError,
    Identity,
    IdentityRegistry,
    InvalidIdentityError,
)
from offline_face_recognition.gallery.session_store import GallerySessionInfo, GallerySessionStore
from offline_face_recognition.imaging.decode import ImageValidationError, decode_image


@dataclass(frozen=True)
class ReferenceImage:
    filename: str
    content: bytes


class GalleryBuildError(ValueError):
    """A reference gallery could not be built transactionally."""

    def __init__(self, message: str, *, code: str, filename: str | None = None) -> None:
        self.code = code
        self.filename = filename
        context = f" ({filename})" if filename else ""
        super().__init__(f"{message}{context}")


class GalleryBuilder:
    """Decode and embed references, publishing a session only after full success."""

    def __init__(
        self,
        settings: Settings,
        face_engine: FaceAnalysisEngine,
        session_store: GallerySessionStore,
    ) -> None:
        self._settings = settings
        self._face_engine = face_engine
        self._session_store = session_store

    def build(self, references: Sequence[ReferenceImage]) -> GallerySessionInfo:
        items = tuple(references)
        if not items:
            raise GalleryBuildError("At least one reference photograph is required", code="empty_gallery")
        if len(items) > self._settings.max_gallery_identities:
            raise GalleryBuildError("Gallery exceeds the identity limit", code="too_many_identities")
        if any(not isinstance(item, ReferenceImage) for item in items):
            raise GalleryBuildError("References must be ReferenceImage values", code="invalid_reference_input")
        if any(not isinstance(item.content, bytes) for item in items):
            raise GalleryBuildError("Reference contents must be bytes", code="invalid_reference_input")
        total_bytes = sum(len(item.content) for item in items)
        if total_bytes > self._settings.max_gallery_upload_bytes:
            raise GalleryBuildError("Gallery exceeds the aggregate upload limit", code="gallery_too_large")

        registry = IdentityRegistry()
        identities: list[Identity] = []
        for item in items:
            try:
                identities.append(registry.add(item.filename))
            except DuplicateIdentityError as exc:
                raise GalleryBuildError(str(exc), code="duplicate_identity", filename=item.filename) from exc
            except InvalidIdentityError as exc:
                raise GalleryBuildError(str(exc), code="invalid_identity", filename=item.filename) from exc

        entries: dict[str, tuple[str, np.ndarray]] = {}
        temporary_embeddings: list[np.ndarray] = []
        try:
            for item, identity in zip(items, identities, strict=True):
                image_bgr: np.ndarray | None = None
                try:
                    image_bgr = decode_image(item.content, item.filename, self._settings)
                except ImageValidationError as exc:
                    raise GalleryBuildError(
                        str(exc), code="invalid_reference_image", filename=item.filename
                    ) from exc

                try:
                    analysis = self._face_engine.analyze(image_bgr)
                    if analysis.detections.rejected_detection_count:
                        raise GalleryBuildError(
                            "Reference image contains invalid detection rows",
                            code="invalid_face_detection",
                            filename=item.filename,
                        )
                    detected_count = len(analysis.detections.detections)
                    if detected_count == 0:
                        raise GalleryBuildError(
                            "Reference image must contain exactly one face; none were detected",
                            code="zero_faces",
                            filename=item.filename,
                        )
                    if detected_count != 1:
                        raise GalleryBuildError(
                            "Reference image must contain exactly one face",
                            code="multiple_faces",
                            filename=item.filename,
                        )
                    if len(analysis.faces) != 1:
                        raise GalleryBuildError(
                            "Reference face embedding was missing or misaligned",
                            code="embedding_alignment_failure",
                            filename=item.filename,
                        )
                    try:
                        vector = validate_normalized_embedding(analysis.faces[0].embedding)
                    except EmbeddingValidationError as exc:
                        raise GalleryBuildError(
                            str(exc), code="invalid_embedding", filename=item.filename
                        ) from exc
                    temporary_embeddings.append(vector)
                    entries[identity.key] = (identity.label, vector)
                except GalleryBuildError:
                    raise
                except Exception as exc:
                    raise GalleryBuildError(
                        "Reference face processing failed",
                        code="face_processing_failed",
                        filename=item.filename,
                    ) from exc
                finally:
                    if image_bgr is not None:
                        image_bgr.fill(0)

            return self._session_store.create(entries)
        finally:
            for vector in temporary_embeddings:
                if vector.flags.writeable:
                    vector.fill(0.0)
            entries.clear()
