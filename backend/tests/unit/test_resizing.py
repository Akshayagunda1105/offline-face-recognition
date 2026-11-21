import numpy as np
import pytest

from offline_face_recognition.faces.resizing import prepare_working_image


def test_downscales_only_above_limit_preserves_aspect_and_does_not_mutate_source():
    source = np.arange(1601 * 800 * 3, dtype=np.uint32).reshape(800, 1601, 3)
    source = (source % 256).astype(np.uint8)
    before = source.copy()

    prepared = prepare_working_image(source, 1200)

    assert prepared.image_bgr.shape == (600, 1200, 3)
    assert prepared.image_bgr is not source
    assert prepared.scale_x == pytest.approx(1601 / 1200)
    assert prepared.scale_y == pytest.approx(800 / 600)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("shape", [(700, 1200), (400, 800), (1200, 800)])
def test_images_at_or_below_limit_are_not_resized_or_upscaled_and_use_a_copy(shape):
    source = np.zeros((*shape, 3), dtype=np.uint8)

    prepared = prepare_working_image(source, 1200)

    assert prepared.image_bgr is not source
    assert prepared.image_bgr.shape == source.shape
    assert (prepared.scale_x, prepared.scale_y) == (1.0, 1.0)
    np.testing.assert_array_equal(prepared.image_bgr, source)


@pytest.mark.parametrize("max_dimension", [0, -1, 1.5, True])
def test_invalid_maximum_dimension_is_rejected(max_dimension):
    with pytest.raises(ValueError, match="positive integer"):
        prepare_working_image(np.zeros((2, 2, 3), np.uint8), max_dimension)
