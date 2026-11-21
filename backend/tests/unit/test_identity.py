import pytest

from offline_face_recognition.gallery.identity import (
    DuplicateIdentityError,
    IdentityRegistry,
    InvalidIdentityError,
    extract_identity,
)


def test_identity_uses_only_sanitized_filename_basename():
    identity = extract_identity(r"C:\private\references\Akshaya.jpg")

    assert identity.label == "Akshaya"
    assert identity.key == "akshaya"


@pytest.mark.parametrize("filename", ["", "   .png", ".jpg", "folder/", "bad\nname.png"])
def test_empty_or_invalid_filename_labels_are_rejected(filename):
    with pytest.raises(InvalidIdentityError):
        extract_identity(filename)


def test_identity_key_uses_unicode_compatibility_normalization_and_casefolding():
    composed = extract_identity("Jos\u00e9.jpg")
    decomposed = extract_identity("Jose\u0301.png")

    assert composed.key == decomposed.key
    assert composed.label == "Jos\u00e9"


def test_duplicate_identity_names_are_detected_case_insensitively():
    registry = IdentityRegistry()
    first = registry.add("Akshaya.jpg")

    assert first.label == "Akshaya"
    with pytest.raises(DuplicateIdentityError):
        registry.add("akshaya.png")
