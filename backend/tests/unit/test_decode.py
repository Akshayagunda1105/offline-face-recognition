from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from offline_face_recognition.core.config import Settings
from offline_face_recognition.imaging.decode import ImageValidationError, decode_image


def _settings(**overrides):
    values = {
        "project_root": ".",
        "yunet_model_path": "models/yunet.onnx",
        "sface_model_path": "models/sface.onnx",
    }
    values.update(overrides)
    return Settings(**values)


def _encoded_image(fmt="PNG", size=(32, 24), *, frames=1, **save_options):
    first = Image.new("RGB", size, (10, 80, 200))
    output = BytesIO()
    if frames > 1:
        second = Image.new("RGB", size, (200, 80, 10))
        first.save(output, format=fmt, save_all=True, append_images=[second], **save_options)
    else:
        first.save(output, format=fmt, **save_options)
    return output.getvalue()


@pytest.mark.parametrize(
    ("fmt", "filename"),
    [("JPEG", "sample.jpg"), ("JPEG", "sample.jpeg"), ("PNG", "sample.png"), ("WEBP", "sample.webp")],
)
def test_supported_formats_decode_to_opencv_bgr(fmt, filename):
    image = decode_image(_encoded_image(fmt), filename, _settings())

    assert image.shape == (24, 32, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    tolerance = 10 if fmt == "WEBP" else 3
    assert tuple(int(channel) for channel in image[0, 0]) == pytest.approx((200, 80, 10), abs=tolerance)


def test_extension_must_match_actual_image_format():
    with pytest.raises(ImageValidationError, match="does not match"):
        decode_image(_encoded_image("PNG"), "portrait.jpg", _settings())


def test_exif_orientation_is_applied_before_returning_dimensions():
    image = Image.new("RGB", (32, 24), "red")
    exif = Image.Exif()
    exif[274] = 6
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif)

    decoded = decode_image(output.getvalue(), "rotated.jpg", _settings())

    assert decoded.shape == (32, 24, 3)


@pytest.mark.parametrize(
    ("data", "filename"),
    [(b"not an image", "broken.png"), (_encoded_image("PNG"), "unsupported.gif")],
)
def test_corrupted_and_unsupported_images_are_rejected(data, filename):
    with pytest.raises(ImageValidationError):
        decode_image(data, filename, _settings())


def test_multi_frame_webp_is_rejected():
    data = _encoded_image("WEBP", frames=2, duration=100, loop=0)

    with pytest.raises(ImageValidationError, match="Multi-frame"):
        decode_image(data, "animated.webp", _settings())


def test_empty_data_and_byte_limit_are_rejected():
    with pytest.raises(ImageValidationError, match="empty"):
        decode_image(b"", "empty.png", _settings())
    data = _encoded_image("PNG")
    with pytest.raises(ImageValidationError, match="byte limit"):
        decode_image(data, "large.png", _settings(max_upload_bytes=len(data) - 1))


def test_pillow_decompression_bomb_warning_is_rejected(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(ImageValidationError, match="decompression limit"):
        decode_image(_encoded_image("PNG", (20, 20)), "large-header.png", _settings())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_image_width": 20}, "dimensions"),
        ({"max_image_height": 20}, "dimensions"),
        ({"max_image_pixels": 700}, "pixel count"),
    ],
)
def test_dimension_and_pixel_limits_are_enforced(overrides, message):
    with pytest.raises(ImageValidationError, match=message):
        decode_image(_encoded_image("PNG"), "sample.png", _settings(**overrides))


def test_variable_image_dimensions_are_supported_within_limits():
    small = decode_image(_encoded_image("PNG", (8, 6)), "small.png", _settings())
    larger = decode_image(_encoded_image("PNG", (320, 200)), "large.png", _settings())

    assert small.shape[:2] == (6, 8)
    assert larger.shape[:2] == (200, 320)
