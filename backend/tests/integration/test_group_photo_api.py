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


def test_experimental_recognition_endpoint_requires_opt_in_and_threshold_pair():
    base = {
        "OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD": "0.40",
        "OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN": "0.10",
    }
    cases = [
        ({}, "recognition_not_ready"),
        ({"OFFLINE_FACE_EXPERIMENTAL_RECOGNITION_ENABLED": "true"}, "recognition_not_ready"),
        (base, "recognition_not_ready"),
        ({**base, "OFFLINE_FACE_EXPERIMENTAL_RECOGNITION_ENABLED": "false"}, "recognition_not_ready"),
        ({**base, "OFFLINE_FACE_EXPERIMENTAL_RECOGNITION_ENABLED": "true"}, "invalid_group_photo"),
    ]
    async def exercise_case(env, expected_code):
        settings = Settings.from_environment(
            environ={"OFFLINE_FACE_PROJECT_ROOT": str(ROOT), **env}
        )
        settings = settings.model_copy(update={
            "yunet_model_path": ROOT / "missing-yunet-test.onnx",
            "sface_model_path": ROOT / "missing-sface-test.onnx",
        })
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            runtime = app.state.model_runtime
            # Isolate the gate test from real model assets and image data.
            runtime.detector = object()
            runtime.recognizer = object()
            runtime.error_code = None

            class GalleryServiceStub:
                def get_gallery_info(self, _session_id):
                    return object()

            class ProcessorStub:
                def process(self, *_args):
                    from offline_face_recognition.processing.group_photo import GroupPhotoProcessingError
                    raise GroupPhotoProcessingError("invalid fixture", code="invalid_group_photo")

            app.state.gallery_service = GalleryServiceStub()
            app.state.group_photo_processor = ProcessorStub()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                headers = {"Authorization": "Bearer synthetic-test-session"}
                response = await client.post(
                    "/api/group-photos/recognize",
                    headers=headers,
                    files={"photo": ("invalid.png", b"not an image", "image/png")},
                )
                expected_status = 422 if expected_code == "invalid_group_photo" else 503
                assert response.status_code == expected_status
                assert response.json()["detail"]["code"] == expected_code

    async def run_cases():
        for env, expected_code in cases:
            await exercise_case(env, expected_code)

    anyio.run(run_cases)


def test_recognition_endpoint_preserves_models_not_ready_error(tmp_path):
    app = create_app(Settings(
        project_root=tmp_path,
        yunet_model_path=tmp_path / "missing-yunet.onnx",
        sface_model_path=tmp_path / "missing-sface.onnx",
        matching_similarity_threshold=0.40,
        matching_ambiguity_margin=0.10,
        experimental_recognition_enabled=True,
    ))

    async def request():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/group-photos/recognize",
                    headers={"Authorization": "Bearer test-session"},
                    files={"photo": ("group.png", b"bytes", "image/png")},
                )
                assert response.status_code == 503
                assert response.json()["detail"]["code"] == "models_not_ready"

    anyio.run(request)


def test_gallery_validation_api_includes_offending_filename(tmp_path):
    from offline_face_recognition.gallery.builder import GalleryBuildError

    cases = [
        ("multiple_faces", "Family.png", "Reference image must contain exactly one face"),
        ("zero_faces", "EmptyFace.jpg", "Reference image must contain exactly one face; none were detected"),
        ("invalid_reference_image", "Broken.webp", "Image content does not match supported image format"),
        ("duplicate_identity", "alex.png", "Identity label duplicates another filename"),
        ("invalid_identity", ".jpg", "Filename does not contain an identity label"),
    ]

    async def exercise_case(code, filename, message):
        settings = Settings(
            project_root=tmp_path,
            yunet_model_path=tmp_path / "missing-yunet.onnx",
            sface_model_path=tmp_path / "missing-sface.onnx",
        )
        app = create_app(settings)

        class GalleryServiceStub:
            def create_gallery(self, references, *, validation_failures=()):
                assert references[0].filename == filename
                assert not validation_failures
                raise GalleryBuildError(message, code=code, filename=filename)

        async with app.router.lifespan_context(app):
            app.state.gallery_service = GalleryServiceStub()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/galleries",
                    files=[("reference_images", (filename, b"fixture-bytes", "image/png"))],
                )
                assert response.status_code == 422
                assert response.json()["detail"] == {
                    "code": code,
                    "message": f"{message} ({filename})",
                    "filename": filename,
                    "invalid_image_count": 1,
                    "errors": [{"filename": filename, "code": code, "message": message}],
                }

    async def run_cases():
        for code, filename, message in cases:
            await exercise_case(code, filename, message)

    anyio.run(run_cases)


def test_gallery_validation_api_returns_all_per_image_failures(tmp_path):
    from offline_face_recognition.gallery.builder import GalleryBuildError, GalleryValidationFailure

    failures = (
        GalleryValidationFailure("Family.png", "multiple_faces", "More than one face was detected"),
        GalleryValidationFailure("NoFace.jpg", "zero_faces", "No face was detected"),
        GalleryValidationFailure("Broken.webp", "invalid_reference_image", "Image could not be decoded"),
    )
    app = create_app(Settings(
        project_root=tmp_path,
        yunet_model_path=tmp_path / "missing-yunet.onnx",
        sface_model_path=tmp_path / "missing-sface.onnx",
    ))

    class GalleryServiceStub:
        def create_gallery(self, _references, *, validation_failures=()):
            assert not validation_failures
            raise GalleryBuildError.from_failures(failures)

    async def request():
        async with app.router.lifespan_context(app):
            app.state.gallery_service = GalleryServiceStub()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/galleries",
                    files=[
                        ("reference_images", ("Family.png", b"family", "image/png")),
                        ("reference_images", ("NoFace.jpg", b"empty", "image/jpeg")),
                        ("reference_images", ("Broken.webp", b"broken", "image/webp")),
                    ],
                )
                assert response.status_code == 422
                assert response.json()["detail"] == {
                    "code": "gallery_validation_failed",
                    "message": "3 reference image(s) failed validation",
                    "invalid_image_count": 3,
                    "errors": [
                        {"filename": item.filename, "code": item.code, "message": item.message}
                        for item in failures
                    ],
                }

    anyio.run(request)


def test_gallery_api_aggregates_per_image_upload_size_with_validation_errors(tmp_path):
    from offline_face_recognition.gallery.builder import GalleryBuildError, GalleryValidationFailure

    app = create_app(Settings(
        project_root=tmp_path,
        yunet_model_path=tmp_path / "missing-yunet.onnx",
        sface_model_path=tmp_path / "missing-sface.onnx",
        max_upload_bytes=5,
        max_gallery_upload_bytes=100,
    ))

    class GalleryServiceStub:
        def create_gallery(self, references, *, validation_failures=()):
            assert [reference.filename for reference in references] == ["MultiFace.png"]
            assert len(validation_failures) == 1
            assert validation_failures[0].filename == "Large.jpg"
            face_failure = GalleryValidationFailure(
                "MultiFace.png", "multiple_faces", "Reference image must contain exactly one face"
            )
            raise GalleryBuildError.from_failures((*validation_failures, face_failure))

    async def request():
        async with app.router.lifespan_context(app):
            app.state.gallery_service = GalleryServiceStub()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/galleries",
                    files=[
                        ("reference_images", ("Large.jpg", b"0123456789", "image/jpeg")),
                        ("reference_images", ("MultiFace.png", b"ok", "image/png")),
                    ],
                )
                assert response.status_code == 422
                assert response.json()["detail"]["invalid_image_count"] == 2
                assert response.json()["detail"]["errors"] == [
                    {
                        "filename": "Large.jpg",
                        "code": "reference_image_too_large",
                        "message": "Reference image exceeds the individual upload size limit",
                    },
                    {
                        "filename": "MultiFace.png",
                        "code": "multiple_faces",
                        "message": "Reference image must contain exactly one face",
                    },
                ]

    anyio.run(request)
