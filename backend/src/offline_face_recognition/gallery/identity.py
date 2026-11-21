"""Identity labels derived from reference image filenames."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import PurePath


class InvalidIdentityError(ValueError):
    """A filename does not contain a valid display identity."""


class DuplicateIdentityError(ValueError):
    """A gallery contains more than one filename for the same identity key."""


@dataclass(frozen=True)
class Identity:
    key: str
    label: str


def extract_identity(filename: str | PurePath) -> Identity:
    """Extract a readable label and normalized key from a filename only."""
    raw_name = str(filename).replace("\\", "/")
    basename = PurePath(raw_name).name
    parsed_name = PurePath(basename)
    if not parsed_name.suffix:
        raise InvalidIdentityError("Filename does not contain an image filename stem")
    stem = parsed_name.stem
    label = unicodedata.normalize("NFC", stem).strip()
    if not label or label in {".", ".."}:
        raise InvalidIdentityError("Filename does not contain an identity label")
    if any(unicodedata.category(character).startswith("C") for character in label):
        raise InvalidIdentityError("Identity label contains invalid control or formatting characters")
    key = unicodedata.normalize("NFKC", label).casefold()
    if not key:
        raise InvalidIdentityError("Filename does not contain an identity label")
    return Identity(key=key, label=label)


class IdentityRegistry:
    """Detect duplicate identity keys while a reference gallery is assembled."""

    def __init__(self) -> None:
        self._identities: dict[str, Identity] = {}

    def add(self, filename: str | PurePath) -> Identity:
        identity = extract_identity(filename)
        if identity.key in self._identities:
            previous = self._identities[identity.key]
            raise DuplicateIdentityError(
                f"Identity label {identity.label!r} duplicates {previous.label!r}"
            )
        self._identities[identity.key] = identity
        return identity
