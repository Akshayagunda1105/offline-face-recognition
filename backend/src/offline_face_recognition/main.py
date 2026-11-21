"""FastAPI application factory and process lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from offline_face_recognition.annotation.image_annotator import ImageAnnotator
from offline_face_recognition.api.galleries import router as gallery_router
from offline_face_recognition.api.health import router as health_router
from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.engine import FaceAnalysisEngine
from offline_face_recognition.gallery.builder import GalleryBuilder
from offline_face_recognition.gallery.service import GalleryService
from offline_face_recognition.gallery.session_store import (
    EXPIRY_CLEANUP_INTERVAL_SECONDS,
    GallerySessionStore,
)
from offline_face_recognition.matching.matcher import FaceMatcher
from offline_face_recognition.models.runtime import ModelRuntime
from offline_face_recognition.processing.group_photo import GroupPhotoProcessor


async def _expire_gallery_sessions(store: GallerySessionStore) -> None:
    while True:
        await asyncio.sleep(EXPIRY_CLEANUP_INTERVAL_SECONDS)
        await asyncio.to_thread(store.purge_expired)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the local API app without loading models until lifespan startup."""
    app_settings = settings or Settings.from_environment()
    runtime = ModelRuntime(app_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model_runtime = runtime
        app.state.settings = app_settings
        runtime.load()
        session_store = GallerySessionStore(app_settings)
        app.state.gallery_session_store = session_store
        app.state.gallery_service = None
        app.state.group_photo_processor = None
        app.state.processing_semaphore = asyncio.Semaphore(app_settings.max_concurrent_processing)
        expiry_task: asyncio.Task[None] | None = None
        try:
            if runtime.readiness.models_ready:
                face_engine = FaceAnalysisEngine(runtime)
                builder = GalleryBuilder(app_settings, face_engine, session_store)
                matcher = FaceMatcher(app_settings)
                app.state.gallery_service = GalleryService(builder, session_store, matcher)
                app.state.group_photo_processor = GroupPhotoProcessor(
                    app_settings,
                    face_engine,
                    app.state.gallery_service,
                    ImageAnnotator(),
                )
            expiry_task = asyncio.create_task(_expire_gallery_sessions(session_store))
            yield
        finally:
            if expiry_task is not None:
                expiry_task.cancel()
                try:
                    await expiry_task
                except asyncio.CancelledError:
                    pass
            session_store.close()
            app.state.gallery_service = None
            app.state.group_photo_processor = None
            runtime.close()

    app = FastAPI(
        title="Offline Face Recognition API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(health_router)
    app.include_router(gallery_router)
    return app


app = create_app()
