from pathlib import Path

import cv2

from offline_face_recognition.core.config import Settings
from offline_face_recognition.models.runtime import ModelRuntime


def test_local_model_loader_requests_opencv_cpu_and_provisional_yunet_settings(tmp_path):
    yunet_path = tmp_path / "yunet.onnx"
    sface_path = tmp_path / "sface.onnx"
    yunet_path.write_bytes(b"local-test-model")
    sface_path.write_bytes(b"local-test-model")
    calls = {}

    def detector_factory(*args):
        calls["yunet"] = args
        return object()

    def recognizer_factory(*args):
        calls["sface"] = args
        return object()

    settings = Settings(
        project_root=tmp_path,
        yunet_model_path=yunet_path,
        sface_model_path=sface_path,
    )
    runtime = ModelRuntime(
        settings,
        detector_factory=detector_factory,
        recognizer_factory=recognizer_factory,
    )
    runtime.load()

    assert runtime.readiness.models_ready
    assert runtime.recognition_ready is False
    assert calls["yunet"] == (
        str(yunet_path),
        "",
        (320, 320),
        0.85,
        0.30,
        5000,
        cv2.dnn.DNN_BACKEND_OPENCV,
        cv2.dnn.DNN_TARGET_CPU,
    )
    assert calls["sface"] == (
        str(sface_path),
        "",
        cv2.dnn.DNN_BACKEND_OPENCV,
        cv2.dnn.DNN_TARGET_CPU,
    )
    assert cv2.ocl.useOpenCL() is False


def test_missing_local_models_report_not_ready_without_network_fallback(tmp_path):
    settings = Settings(
        project_root=tmp_path,
        yunet_model_path=tmp_path / "missing-yunet.onnx",
        sface_model_path=tmp_path / "missing-sface.onnx",
    )
    runtime = ModelRuntime(settings)

    runtime.load()

    assert runtime.readiness.models_ready is False
    assert runtime.readiness.yunet_ready is False
    assert runtime.readiness.sface_ready is False
    assert runtime.readiness.error_code == "model_files_missing"
    assert runtime.recognition_ready is False
    assert runtime.experimental_recognition_available is False


def test_experimental_readiness_requires_models_opt_in_and_both_thresholds(tmp_path):
    yunet_path = tmp_path / "yunet.onnx"
    sface_path = tmp_path / "sface.onnx"
    yunet_path.write_bytes(b"local-test-model")
    sface_path.write_bytes(b"local-test-model")
    kwargs = dict(
        project_root=tmp_path,
        yunet_model_path=yunet_path,
        sface_model_path=sface_path,
    )
    factories = dict(
        detector_factory=lambda *_args: object(),
        recognizer_factory=lambda *_args: object(),
    )

    cases = [
        (Settings(**kwargs), False),
        (Settings(**kwargs, matching_similarity_threshold=0.40, matching_ambiguity_margin=0.10), False),
        (Settings(**kwargs, experimental_recognition_enabled=True), False),
        (Settings(**kwargs, matching_similarity_threshold=0.40, matching_ambiguity_margin=0.10, experimental_recognition_enabled=True), True),
    ]
    for settings, expected in cases:
        runtime = ModelRuntime(settings, **factories)
        runtime.load()
        assert runtime.readiness.models_ready
        assert runtime.experimental_recognition_available is expected
        assert runtime.recognition_ready is False


def test_model_load_failure_clears_partial_models(tmp_path):
    yunet_path = tmp_path / "yunet.onnx"
    sface_path = tmp_path / "sface.onnx"
    yunet_path.write_bytes(b"local-test-model")
    sface_path.write_bytes(b"local-test-model")

    def failing_recognizer(*_args):
        raise RuntimeError("simulated local model load failure")

    settings = Settings(
        project_root=tmp_path,
        yunet_model_path=yunet_path,
        sface_model_path=sface_path,
    )
    runtime = ModelRuntime(
        settings,
        detector_factory=lambda *_args: object(),
        recognizer_factory=failing_recognizer,
    )

    runtime.load()

    assert runtime.detector is None
    assert runtime.recognizer is None
    assert runtime.readiness.models_ready is False
    assert runtime.readiness.error_code == "model_load_failed"
