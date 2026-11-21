import numpy as np
import pytest

from offline_face_recognition.annotation.image_annotator import (
    Annotation,
    ImageAnnotationError,
    ImageAnnotator,
)


def test_annotator_draws_on_a_copy_and_leaves_source_unchanged():
    source = np.zeros((64, 80, 3), dtype=np.uint8)
    before = source.copy()

    annotated = ImageAnnotator().annotate(
        source,
        [Annotation(detection_index=3, box=(10.0, 12.0, 30.0, 28.0), label="Asha")],
    )

    np.testing.assert_array_equal(source, before)
    assert annotated is not source
    assert annotated.shape == source.shape
    assert tuple(int(value) for value in annotated[12, 10]) == (0, 220, 0)
    assert np.count_nonzero(annotated) > 0


def test_annotator_clips_boxes_to_image_edges_and_accepts_unknown_label():
    source = np.zeros((20, 20, 3), dtype=np.uint8)

    annotated = ImageAnnotator().annotate(
        source, [Annotation(detection_index=0, box=(-5.0, -4.0, 10.0, 12.0), label="Unknown")]
    )

    assert annotated.shape == source.shape
    assert np.count_nonzero(annotated) > 0


@pytest.mark.parametrize(
    "annotation",
    [
        Annotation(-1, (0.0, 0.0, 2.0, 2.0), "Unknown"),
        Annotation(0, (0.0, 0.0, float("nan"), 2.0), "Asha"),
        Annotation(0, (0.0, 0.0, 2.0, 2.0), ""),
    ],
)
def test_invalid_annotation_data_is_rejected(annotation):
    with pytest.raises(ImageAnnotationError):
        ImageAnnotator().annotate(np.zeros((20, 20, 3), np.uint8), [annotation])
