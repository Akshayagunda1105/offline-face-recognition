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
    assert recognition.status_code == 200
    assert recognition.json() == {
        "models_ready": True,
        "recognition_ready": False,
        "experimental_recognition_available": False,
        "experimental": False,
        "calibrated": False,
        "reason": "experimental_opt_in_disabled",
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
    assert recognition.status_code == 200
    assert recognition.json() == {
        "models_ready": False,
        "recognition_ready": False,
        "experimental_recognition_available": False,
        "experimental": False,
        "calibrated": False,
        "reason": "models_not_ready",
    }
    assert app.state.gallery_service is None
    assert app.state.gallery_session_store.active_session_count == 0


def test_health_reports_experimental_availability_without_calibration():
    settings = Settings.from_environment(
        environ={
            "OFFLINE_FACE_MATCHING_SIMILARITY_THRESHOLD": "0.40",
            "OFFLINE_FACE_MATCHING_AMBIGUITY_MARGIN": "0.10",
            "OFFLINE_FACE_EXPERIMENTAL_RECOGNITION_ENABLED": "true",
        }
    )
    app = create_app(settings)
    live, models, recognition = _get_health_responses(app)

    assert models.status_code == 200
    assert recognition.status_code == 200
    assert recognition.json() == {
        "models_ready": True,
        "recognition_ready": False,
        "experimental_recognition_available": True,
        "experimental": True,
        "calibrated": False,
        "reason": None,
    }
    assert app.state.model_runtime.recognition_ready is False


def test_health_distinguishes_opt_in_from_missing_threshold_configuration():
    settings = Settings.from_environment(
        environ={"OFFLINE_FACE_EXPERIMENTAL_RECOGNITION_ENABLED": "true"}
    )
    app = create_app(settings)
    _, models, recognition = _get_health_responses(app)

    assert models.status_code == 200
    assert recognition.status_code == 200
    assert recognition.json() == {
        "models_ready": True,
        "recognition_ready": False,
        "experimental_recognition_available": False,
        "experimental": True,
        "calibrated": False,
        "reason": "matching_thresholds_not_configured",
    }
