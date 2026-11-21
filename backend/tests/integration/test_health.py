import anyio
import httpx

from offline_face_recognition.core.config import Settings
from offline_face_recognition.main import create_app


def _get_health_responses(app):
    async def request_health():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return (
                    await client.get("/health/live"),
                    await client.get("/health/models"),
                    await client.get("/health/recognition"),
                )

    return anyio.run(request_health)


def test_health_reports_models_ready_but_recognition_not_ready():
    settings = Settings.from_environment(environ={})
    app = create_app(settings)

    live, models, recognition = _get_health_responses(app)

    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert models.status_code == 200
    assert models.json() == {
        "models_ready": True,
        "yunet_ready": True,
        "sface_ready": True,
        "error_code": None,
    }
    assert recognition.status_code == 503
    assert recognition.json() == {
        "recognition_ready": False,
        "reason": "matching_thresholds_not_calibrated",
    }
    assert app.state.gallery_service is None
    assert app.state.group_photo_processor is None
    assert app.state.model_runtime.recognition_ready is False
    assert app.state.gallery_session_store.active_session_count == 0


def test_health_reports_missing_models_and_keeps_liveness_available(tmp_path):
    settings = Settings(
        project_root=tmp_path,
        yunet_model_path=tmp_path / "missing-yunet.onnx",
        sface_model_path=tmp_path / "missing-sface.onnx",
    )
    app = create_app(settings)

    live, models, recognition = _get_health_responses(app)

    assert live.status_code == 200
    assert models.status_code == 503
    assert models.json() == {
        "models_ready": False,
        "yunet_ready": False,
        "sface_ready": False,
        "error_code": "model_files_missing",
    }
    assert recognition.status_code == 503
    assert recognition.json()["recognition_ready"] is False
    assert app.state.gallery_service is None
    assert app.state.gallery_session_store.active_session_count == 0
