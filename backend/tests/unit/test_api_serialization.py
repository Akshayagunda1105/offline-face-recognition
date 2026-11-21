import json

import numpy as np
import pytest

from offline_face_recognition.api.galleries import (
    AnnotatedResponseTooLarge,
    _serialize_recognition_result,
)
from offline_face_recognition.matching.matcher import MatchStatus
from offline_face_recognition.processing.group_photo import GroupPhotoResult, Identification


def _result():
    return GroupPhotoResult(
        annotated_image_bgr=np.full((24, 32, 3), 120, dtype=np.uint8),
        identifications=(
            Identification(2, (1.0, 2.0, 10.0, 12.0), MatchStatus.KNOWN, "Asha"),
            Identification(5, (12.0, 2.0, 10.0, 12.0), MatchStatus.UNKNOWN, None),
        ),
    )


def test_serialized_recognition_response_has_labels_but_no_similarity_scores():
    body = _serialize_recognition_result(_result(), 25 * 1024 * 1024)
    parsed = json.loads(body)

    assert parsed["face_count"] == 2
    assert parsed["annotated_image_media_type"] == "image/jpeg"
    assert parsed["identifications"] == [
        {
            "detection_index": 2,
            "bounding_box": {"x": 1.0, "y": 2.0, "width": 10.0, "height": 12.0},
            "status": "Known",
            "identity_label": "Asha",
        },
        {
            "detection_index": 5,
            "bounding_box": {"x": 12.0, "y": 2.0, "width": 10.0, "height": 12.0},
            "status": "Unknown",
            "identity_label": None,
        },
    ]
    assert "similarity" not in body.casefold()
    assert "score" not in body.casefold()


def test_serialized_annotated_response_enforces_configured_size_limit():
    with pytest.raises(AnnotatedResponseTooLarge):
        _serialize_recognition_result(_result(), 1024)
