"""Thread-safe, process-local gallery sessions and best-effort cleanup."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np

from offline_face_recognition.core.config import Settings
from offline_face_recognition.core.embeddings import (
    EmbeddingValidationError,
    validate_normalized_embedding,
)

EXPIRY_CLEANUP_INTERVAL_SECONDS = 30


class GallerySessionError(RuntimeError):
    """Base class for gallery session failures."""


class GallerySessionUnavailableError(GallerySessionError):
    """The session does not exist, has expired, or has been deleted."""


class GallerySessionCapacityError(GallerySessionError):
    """The configured number of active sessions has been reached."""


class GallerySessionStoreClosedError(GallerySessionError):
    """The process-local session store is shutting down or closed."""


class GalleryStoreInputError(GallerySessionError, ValueError):
    """Gallery entries cannot be safely stored."""


@dataclass(frozen=True)
class GallerySessionInfo:
    session_id: str
    created_at: datetime
    expires_at: datetime
    identity_count: int
    identity_labels: tuple[str, ...]


@dataclass(frozen=True)
class _StoredIdentity:
    identity_key: str
    label: str
    embedding: np.ndarray = field(repr=False, compare=False)


@dataclass
class _SessionRecord:
    session_id: str
    created_at: datetime
    expires_at: datetime
    expires_at_monotonic: float
    identities: dict[str, _StoredIdentity]
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class GallerySessionStore:
    """Own private gallery vectors behind session-scoped locking.

    The lock order is always store map lock then session lock. A borrower holds
    only the session lock while matching and never reacquires the map lock.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        session_id_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    ) -> None:
        self._settings = settings
        self._monotonic = monotonic
        self._utcnow = utcnow
        self._session_id_factory = session_id_factory
        self._map_lock = threading.RLock()
        self._sessions: dict[str, _SessionRecord] = {}
        self._closed = False

    def create(
        self,
        identities: Mapping[str, tuple[str, np.ndarray]],
    ) -> GallerySessionInfo:
        if not identities:
            raise GalleryStoreInputError("A gallery session must contain at least one identity")
        if len(identities) > self._settings.max_gallery_identities:
            raise GalleryStoreInputError("Gallery exceeds the configured identity limit")

        stored: dict[str, _StoredIdentity] = {}
        try:
            for key, (label, vector) in identities.items():
                if not key or not label:
                    raise GalleryStoreInputError("Gallery identity keys and labels must be non-empty")
                try:
                    private_vector = validate_normalized_embedding(vector, copy=True)
                except EmbeddingValidationError as exc:
                    raise GalleryStoreInputError(f"Invalid embedding for identity {label!r}") from exc
                stored[key] = _StoredIdentity(key, label, private_vector)

            with self._map_lock:
                self._ensure_open_locked()
                self._purge_expired_locked(self._monotonic())
                if len(self._sessions) >= self._settings.max_active_gallery_sessions:
                    raise GallerySessionCapacityError("Maximum active gallery sessions reached")
                session_id = self._new_session_id_locked()
                created_at = self._utcnow()
                ttl = self._settings.gallery_session_ttl_seconds
                expires_at = created_at + timedelta(seconds=ttl)
                record = _SessionRecord(
                    session_id=session_id,
                    created_at=created_at,
                    expires_at=expires_at,
                    expires_at_monotonic=self._monotonic() + ttl,
                    identities=stored,
                )
                self._sessions[session_id] = record
                return self._info(record)
        except Exception:
            self._clear_identities(stored)
            raise

    def get_info(self, session_id: str) -> GallerySessionInfo:
        with self._map_lock:
            self._ensure_open_locked()
            self._purge_expired_locked(self._monotonic())
            record = self._sessions.get(session_id)
            if record is None:
                raise GallerySessionUnavailableError("Gallery session is unavailable")
            with record.lock:
                if record.expires_at_monotonic <= self._monotonic():
                    self._erase_and_remove_locked(record)
                    raise GallerySessionUnavailableError("Gallery session is unavailable")
                return self._info(record)

    @contextmanager
    def _borrow(self, session_id: str) -> Iterator[tuple[_StoredIdentity, ...]]:
        """Yield internal immutable-to-callers vectors while holding the session lock."""
        with self._map_lock:
            self._ensure_open_locked()
            self._purge_expired_locked(self._monotonic())
            record = self._sessions.get(session_id)
            if record is None:
                raise GallerySessionUnavailableError("Gallery session is unavailable")
            record.lock.acquire()
            if record.expires_at_monotonic <= self._monotonic():
                self._erase_and_remove_locked(record)
                record.lock.release()
                raise GallerySessionUnavailableError("Gallery session is unavailable")
        try:
            yield tuple(record.identities.values())
        finally:
            record.lock.release()

    def delete(self, session_id: str) -> bool:
        with self._map_lock:
            self._ensure_open_locked()
            record = self._sessions.get(session_id)
            if record is None:
                return False
            with record.lock:
                self._erase_and_remove_locked(record)
            return True

    def purge_expired(self) -> int:
        with self._map_lock:
            self._ensure_open_locked()
            return self._purge_expired_locked(self._monotonic())

    @property
    def active_session_count(self) -> int:
        with self._map_lock:
            if self._closed:
                return 0
            self._purge_expired_locked(self._monotonic())
            return len(self._sessions)

    def close(self) -> None:
        with self._map_lock:
            if self._closed:
                return
            for record in list(self._sessions.values()):
                with record.lock:
                    self._clear_identities(record.identities)
                    record.identities.clear()
            self._sessions.clear()
            self._closed = True

    def _new_session_id_locked(self) -> str:
        for _ in range(5):
            candidate = self._session_id_factory()
            if candidate and candidate not in self._sessions:
                return candidate
        raise GallerySessionError("Could not allocate a unique gallery session ID")

    def _purge_expired_locked(self, now: float) -> int:
        expired = [
            record
            for record in self._sessions.values()
            if record.expires_at_monotonic <= now
        ]
        for record in expired:
            with record.lock:
                self._erase_and_remove_locked(record)
        return len(expired)

    def _erase_and_remove_locked(self, record: _SessionRecord) -> None:
        self._clear_identities(record.identities)
        record.identities.clear()
        if self._sessions.get(record.session_id) is record:
            del self._sessions[record.session_id]

    @staticmethod
    def _clear_identities(identities: dict[str, _StoredIdentity]) -> None:
        for identity in identities.values():
            try:
                identity.embedding.flags.writeable = True
                identity.embedding.fill(0.0)
            except ValueError:
                # Drop references even if an array implementation prevents overwriting.
                pass

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise GallerySessionStoreClosedError("Gallery session store is closed")

    @staticmethod
    def _info(record: _SessionRecord) -> GallerySessionInfo:
        return GallerySessionInfo(
            session_id=record.session_id,
            created_at=record.created_at,
            expires_at=record.expires_at,
            identity_count=len(record.identities),
            identity_labels=tuple(identity.label for identity in record.identities.values()),
        )
