"""Shared validation for normalized SFace embeddings."""

from __future__ import annotations

import numpy as np


class EmbeddingValidationError(ValueError):
    """An embedding violates the backend's normalized vector contract."""


def validate_normalized_embedding(value: object, *, copy: bool = False) -> np.ndarray:
    """Validate a 128-value, finite, unit-norm float32 embedding."""
    if not isinstance(value, np.ndarray):
        raise EmbeddingValidationError("Embedding must be a NumPy array")
    if value.shape != (128,) or value.dtype != np.float32:
        raise EmbeddingValidationError("Embedding must have shape (128,) and dtype float32")
    if not np.isfinite(value).all():
        raise EmbeddingValidationError("Embedding contains non-finite values")
    norm = float(np.linalg.norm(value.astype(np.float64)))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-4):
        raise EmbeddingValidationError("Embedding must have unit L2 norm")
    vector = np.ascontiguousarray(value.copy() if copy else value, dtype=np.float32)
    if copy:
        vector.flags.writeable = False
    return vector
