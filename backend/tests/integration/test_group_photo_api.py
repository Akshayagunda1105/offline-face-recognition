from pathlib import Path

import anyio
import httpx
import pytest

from offline_face_recognition.core.config import Settings
from offline_face_recognition.main import create_app

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_IMAGE = ROOT / "synthetic_dataset" / "reference_gallery" / "Person_01.png"
GROUP_PHOTO = ROOT / "synthetic_dataset" / "group_photos" / "group_01.png"


def test_gallery_api_and_recognition_readiness_gate():
    if not REFERENCE_IMAGE.is_file() or not GROUP_PHOTO.is_file():
        pytest.skip("Local synthetic pilot images are unavailable")
    settings = Settings.from_environment(environ={})
    if not settings.yunet_model_path.is_file() or not settings.sface_model_path.is_file():
        pytest.skip("Pinned local model assets are unavailable")
    app = create_app(settings)

    async def exercise_api():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await app.state.processing_semaphore.acquire()
                try:
                    busy = await client.post("/api/galleries")
                finally:
                    app.state.processing_semaphore.release()
                assert busy.status_code == 503
                assert busy.json()["detail"]["code"] == "processing_busy"

                missing_auth = await client.get("/api/galleries/current")
                assert missing_auth.status_code == 401

                with REFERENCE_IMAGE.open("rb") as image_file:
                    created = await client.post(
                        "/api/galleries",
                        files=[
                            (
                                "reference_images",
                                (REFERENCE_IMAGE.name, image_file, "image/png"),
                            )
                        ],
                    )
                assert created.status_code == 201, created.text
                created_body = created.json()
                assert created_body["identity_count"] == 1
                assert created_body["identity_labels"] == ["Person_01"]
                assert len(created_body["session_id"]) >= 32

                headers = {"Authorization": f"Bearer {created_body['session_id']}"}
                gallery = await client.get("/api/galleries/current", headers=headers)
                assert gallery.status_code == 200
                assert gallery.json()["identity_labels"] == ["Person_01"]
                assert "embedding" not in gallery.text.casefold()

                with GROUP_PHOTO.open("rb") as group_file:
                    recognition = await client.post(
                        "/api/group-photos/recognize",
                        headers=headers,
                        files={"photo": (GROUP_PHOTO.name, group_file, "image/png")},
                    )
                assert recognition.status_code == 503
                assert recognition.json()["detail"]["code"] == "recognition_not_ready"
                assert "annotated_image_base64" not in recognition.text
                assert "similarity" not in recognition.text.casefold()

                deleted = await client.delete("/api/galleries/current", headers=headers)
                assert deleted.status_code == 204
                missing = await client.get("/api/galleries/current", headers=headers)
                assert missing.status_code == 404
                unavailable_session = await client.post(
                    "/api/group-photos/recognize",
                    headers=headers,
                    files={"photo": (GROUP_PHOTO.name, GROUP_PHOTO.read_bytes(), "image/png")},
                )
                assert unavailable_session.status_code == 404
                assert unavailable_session.json()["detail"]["code"] == "gallery_session_unavailable"

            assert app.state.gallery_session_store.active_session_count == 0

    anyio.run(exercise_api)


def test_gallery_api_rejects_corrupt_upload_without_creating_session():
    settings = Settings.from_environment(environ={})
    if not settings.yunet_model_path.is_file() or not settings.sface_model_path.is_file():
        pytest.skip("Pinned local model assets are unavailable")
    app = create_app(settings)

    async def exercise_api():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/galleries",
                    files=[("reference_images", ("Broken.png", b"not an image", "image/png"))],
                )
                assert response.status_code == 422
                assert response.json()["detail"]["code"] == "invalid_reference_image"
                assert app.state.gallery_session_store.active_session_count == 0

    anyio.run(exercise_api)
