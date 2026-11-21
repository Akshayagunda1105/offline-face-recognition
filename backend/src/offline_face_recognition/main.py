"""FastAPI application factory and process lifecycle."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from offline_face_recognition.api.health import router as health_router
from offline_face_recognition.core.config import Settings
from offline_face_recognition.models.runtime import ModelRuntime


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the local API app without loading models until lifespan startup."""
    app_settings = settings or Settings.from_environment()
    runtime = ModelRuntime(app_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model_runtime = runtime
        runtime.load()
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(
        title="Offline Face Recognition API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    return app


app = create_app()
