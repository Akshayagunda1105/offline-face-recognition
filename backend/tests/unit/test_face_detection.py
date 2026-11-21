import numpy as np
import pytest

from offline_face_recognition.core.config import Settings
from offline_face_recognition.faces.detection import FaceDetectionError, YuNetDetector
from offline_face_recognition.models.runtime import ModelRuntime


def _runtime(detector):
    return ModelRuntime(
        Settings(
            project_root=".",
            yunet_model_path="yunet.onnx",
            sface_model_path="sface.onnx",
        ),
        detector=detector,
        recognizer=object(),
    )


def _row(x=10, y=12, width=20, height=22, score=0.9):
    return np.array([x, y, width, height, *([15.0, 16.0] * 5), score], dtype=np.float32)


class FakeDetector:
    def __init__(self, rows):
        self.rows = rows
        self.input_sizes = []

    def setInputSize(self, size):
        self.input_sizes.append(size)

    def detect(self, image):
        return (0 if self.rows is None else len(self.rows), self.rows)


def test_detection_sets_actual_input_size_and_preserves_order_and_raw_rows():
    first, second = _row(), _row(x=40)
    detector = FakeDetector(np.stack([first, second]))
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    result = YuNetDetector(_runtime(detector)).detect(image)

    assert detector.input_sizes == [(64, 48)]
    assert len(result.detections) == 2
    assert result.detections[0].raw_row == tuple(float(v) for v in first)
    assert result.detections[0].clipped_box == (10.0, 12.0, 20.0, 22.0)
    assert result.detections[1].raw_row == tuple(float(v) for v in second)


def test_clipping_is_separate_and_does_not_change_row_used_for_alignment():
    row = _row(x=-5, y=-4, width=20, height=18)
    detector = FakeDetector(row.reshape(1, 15))
    result = YuNetDetector(_runtime(detector)).detect(np.zeros((48, 64, 3), dtype=np.uint8))

    face = result.detections[0]
    assert face.raw_row[0:4] == (-5.0, -4.0, 20.0, 18.0)
    assert face.clipped_box == (0.0, 0.0, 15.0, 14.0)


def test_empty_detection_response_is_a_valid_empty_result():
    result = YuNetDetector(_runtime(FakeDetector(None))).detect(np.zeros((12, 10, 3), dtype=np.uint8))

    assert result.detections == ()
    assert result.rejected_detection_count == 0


def test_invalid_detection_rows_are_counted_and_not_returned():
    invalid_nan = _row()
    invalid_nan[4] = np.nan
    invalid_box = _row(width=0)
    invalid_score = _row(score=1.2)
    detector = FakeDetector(np.stack([invalid_nan, invalid_box, invalid_score, _row()]))

    result = YuNetDetector(_runtime(detector)).detect(np.zeros((48, 64, 3), dtype=np.uint8))

    assert len(result.detections) == 1
    assert result.rejected_detection_count == 3
    assert result.detections[0].detection_index == 3


def test_non_numeric_detection_rows_are_rejected_not_leaked_as_raw_errors():
    detector = FakeDetector(np.full((1, 15), "not-a-number", dtype=object))
    result = YuNetDetector(_runtime(detector)).detect(np.zeros((48, 64, 3), dtype=np.uint8))

    assert result.detections == ()
    assert result.rejected_detection_count == 1


@pytest.mark.parametrize("image", [np.zeros((10, 10), np.uint8), np.zeros((0, 5, 3), np.uint8)])
def test_invalid_detection_inputs_raise_typed_error(image):
    with pytest.raises(FaceDetectionError, match="uint8 BGR"):
        YuNetDetector(_runtime(FakeDetector(None))).detect(image)


def test_malformed_detection_shape_and_inference_error_raise_typed_errors():
    with pytest.raises(FaceDetectionError, match="15 values"):
        YuNetDetector(_runtime(FakeDetector(np.zeros((1, 14), np.float32)))).detect(
            np.zeros((10, 10, 3), np.uint8)
        )

    class FailingDetector(FakeDetector):
        def detect(self, _image):
            raise RuntimeError("synthetic inference failure")

    with pytest.raises(FaceDetectionError, match="inference failed"):
        YuNetDetector(_runtime(FailingDetector(None))).detect(np.zeros((10, 10, 3), np.uint8))
