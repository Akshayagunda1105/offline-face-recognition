"""Coordinate gallery construction, private sessions and independent matching."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from offline_face_recognition.gallery.builder import GalleryBuilder, ReferenceImage
from offline_face_recognition.gallery.session_store import (
    GallerySessionInfo,
    GallerySessionStore,
)
from offline_face_recognition.faces.engine import FaceEmbedding
from offline_face_recognition.matching.matcher import (
    FaceMatcher,
    MatchDecision,
    MatchPolicy,
    RankedCandidates,
)


class GalleryService:
    """Bind a session-scoped gallery to pure matching without exposing vectors."""

    def __init__(
        self,
        builder: GalleryBuilder,
        session_store: GallerySessionStore,
        matcher: FaceMatcher,
    ) -> None:
        self._builder = builder
        self._session_store = session_store
        self._matcher = matcher

    def create_gallery(self, references: Sequence[ReferenceImage]) -> GallerySessionInfo:
        return self._builder.build(references)

    def get_gallery_info(self, session_id: str) -> GallerySessionInfo:
        return self._session_store.get_info(session_id)

    def delete_gallery(self, session_id: str) -> bool:
        return self._session_store.delete(session_id)

    def rank_face(
        self,
        session_id: str,
        embedding: np.ndarray,
        *,
        detection_index: int,
    ) -> RankedCandidates:
        with self._session_store._borrow(session_id) as entries:
            return self._matcher.rank(
                embedding,
                entries,
                detection_index=detection_index,
            )

    def match_face(
        self,
        session_id: str,
        embedding: np.ndarray,
        *,
        detection_index: int,
        policy: MatchPolicy | None = None,
    ) -> MatchDecision:
        with self._session_store._borrow(session_id) as entries:
            return self._matcher.match(
                embedding,
                entries,
                detection_index=detection_index,
                policy=policy,
            )

    def match_faces(
        self,
        session_id: str,
        faces: Sequence[FaceEmbedding],
        *,
        policy: MatchPolicy | None = None,
    ) -> tuple[MatchDecision, ...]:
        """Match an image's faces under one session lease, preserving all indices."""
        with self._session_store._borrow(session_id) as entries:
            decisions = []
            for face in faces:
                decisions.append(
                    self._matcher.match(
                        face.embedding,
                        entries,
                        detection_index=face.detection.detection_index,
                        policy=policy,
                    )
                )
            return tuple(decisions)

    def purge_expired(self) -> int:
        return self._session_store.purge_expired()

    def close(self) -> None:
        self._session_store.close()
