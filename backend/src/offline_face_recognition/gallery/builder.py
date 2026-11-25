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
    Identity,
    InvalidIdentityError,
    extract_identity,
)
from offline_face_recognition.gallery.session_store import GallerySessionInfo, GallerySessionStore
from offline_face_recognition.imaging.decode import ImageValidationError, decode_image


@dataclass(frozen=True)
class ReferenceImage:
    filename: str
    content: bytes


@dataclass(frozen=True)
class GalleryValidationFailure:
    filename: str
    code: str
    message: str


class GalleryBuildError(ValueError):
    """A reference gallery could not be built transactionally."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        filename: str | None = None,
        failures: Sequence[GalleryValidationFailure] = (),
    ) -> None:
        self.code = code
        self.filename = filename
        self.failures = tuple(failures)
        if not self.failures and filename is not None:
            self.failures = (GalleryValidationFailure(filename, code, message),)
        self.invalid_image_count = len({failure.filename for failure in self.failures})
        context = f" ({filename})" if filename else ""
        super().__init__(f"{message}{context}")

    @classmethod
    def from_failures(cls, failures: Sequence[GalleryValidationFailure]) -> GalleryBuildError:
        failures = tuple(failures)
        if not failures:
            raise ValueError("At least one gallery validation failure is required")
        invalid_image_count = len({failure.filename for failure in failures})
        if len(failures) == 1:
            failure = failures[0]
            return cls(
                failure.message,
                code=failure.code,
                filename=failure.filename,
                failures=failures,
            )
        return cls(
            f"{invalid_image_count} reference image(s) failed validation",
            code="gallery_validation_failed",
            failures=failures,
        )


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

    def build(
        self,
        references: Sequence[ReferenceImage],
        *,
        validation_failures: Sequence[GalleryValidationFailure] = (),
    ) -> GallerySessionInfo:
        items = tuple(references)
        failures = list(validation_failures)
        if not items and not failures:
            raise GalleryBuildError("At least one reference photograph is required", code="empty_gallery")
        if len(items) + len(validation_failures) > self._settings.max_gallery_identities:
            raise GalleryBuildError("Gallery exceeds the identity limit", code="too_many_identities")
        if any(not isinstance(item, ReferenceImage) for item in items):
            raise GalleryBuildError("References must be ReferenceImage values", code="invalid_reference_input")
        if any(not isinstance(item.content, bytes) for item in items):
            raise GalleryBuildError("Reference contents must be bytes", code="invalid_reference_input")
        total_bytes = sum(len(item.content) for item in items)
        if total_bytes > self._settings.max_gallery_upload_bytes:
            raise GalleryBuildError("Gallery exceeds the aggregate upload limit", code="gallery_too_large")

        identities: list[Identity | None] = []
        invalid_basic_indices: set[int] = set()
        identity_indices: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            try:
                identity = extract_identity(item.filename)
                identities.append(identity)
                identity_indices.setdefault(identity.key, []).append(index)
            except InvalidIdentityError as exc:
                identities.append(None)
                invalid_basic_indices.add(index)
                failures.append(GalleryValidationFailure(item.filename, "invalid_identity", str(exc)))

        for duplicate_indices in identity_indices.values():
            if len(duplicate_indices) < 2:
                continue
            for index in duplicate_indices:
                invalid_basic_indices.add(index)
                conflicts = [
                    items[other_index].filename
                    for other_index in duplicate_indices
                    if other_index != index
                ]
                failures.append(
                    GalleryValidationFailure(
                        items[index].filename,
                        "duplicate_identity",
                        f"Identity filename duplicates another selected file: {', '.join(conflicts)}",
                    )
                )

        # Decode every item before any model inference. This reports all basic
        # filename/content failures while bounding decoded-image memory to one
        # image at a time. Images that fail this stage are skipped during inference.
        for index, item in enumerate(items):
            image_bgr: np.ndarray | None = None
            try:
                image_bgr = decode_image(item.content, item.filename, self._settings)
            except ImageValidationError as exc:
                invalid_basic_indices.add(index)
                failures.append(
                    GalleryValidationFailure(item.filename, "invalid_reference_image", str(exc))
                )
            except Exception:
                invalid_basic_indices.add(index)
                failures.append(
                    GalleryValidationFailure(
                        item.filename,
                        "invalid_reference_image",
                        "Reference image could not be decoded",
                    )
                )
            finally:
                if image_bgr is not None:
                    image_bgr.fill(0)

        entries: dict[str, tuple[str, np.ndarray]] = {}
        temporary_embeddings: list[np.ndarray] = []
        try:
            for index, (item, identity) in enumerate(zip(items, identities, strict=True)):
                if index in invalid_basic_indices:
                    continue
                assert identity is not None
                image_bgr: np.ndarray | None = None
                try:
                    image_bgr = decode_image(item.content, item.filename, self._settings)
                except ImageValidationError as exc:
                    failures.append(
                        GalleryValidationFailure(item.filename, "invalid_reference_image", str(exc))
                    )
                    continue
                except Exception:
                    failures.append(
                        GalleryValidationFailure(
                            item.filename,
                            "invalid_reference_image",
                            "Reference image could not be decoded",
                        )
                    )
                    continue

                try:
                    analysis = self._face_engine.analyze_reference(image_bgr)
                    if analysis.detections.rejected_detection_count:
                        failures.append(
                            GalleryValidationFailure(
                                item.filename,
                                "invalid_face_detection",
                                "Reference image contains invalid detection rows",
                            )
                        )
                        continue
                    detected_count = len(analysis.detections.detections)
                    if detected_count == 0:
                        failures.append(
                            GalleryValidationFailure(
                                item.filename,
                                "zero_faces",
                                "Reference image must contain exactly one face; none were detected",
                            )
                        )
                        continue
                    if detected_count != 1:
                        failures.append(
                            GalleryValidationFailure(
                                item.filename,
                                "multiple_faces",
                                "Reference image must contain exactly one face",
                            )
                        )
                        continue
                    if len(analysis.faces) != 1:
                        failures.append(
                            GalleryValidationFailure(
                                item.filename,
                                "embedding_alignment_failure",
                                "Reference face embedding was missing or misaligned",
                            )
                        )
                        continue
                    try:
                        vector = validate_normalized_embedding(analysis.faces[0].embedding)
                    except EmbeddingValidationError as exc:
                        failures.append(
                            GalleryValidationFailure(item.filename, "invalid_embedding", str(exc))
                        )
                        continue
                    temporary_embeddings.append(vector)
                    entries[identity.key] = (identity.label, vector)
                except Exception as exc:
                    failures.append(
                        GalleryValidationFailure(
                            item.filename,
                            "face_processing_failed",
                            "Reference face processing failed",
                        )
                    )
                finally:
                    if image_bgr is not None:
                        image_bgr.fill(0)

            if failures:
                raise GalleryBuildError.from_failures(failures)
            return self._session_store.create(entries)
        finally:
            for vector in temporary_embeddings:
                if vector.flags.writeable:
                    vector.fill(0.0)
            entries.clear()
