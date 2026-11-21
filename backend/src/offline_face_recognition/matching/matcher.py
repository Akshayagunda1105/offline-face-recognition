"""Cosine ranking and conservative known/unknown decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence

import numpy as np

from offline_face_recognition.core.config import Settings
from offline_face_recognition.core.embeddings import (
    EmbeddingValidationError,
    validate_normalized_embedding,
)


class MatchingError(ValueError):
    """Base error for invalid matching input or unavailable decision policy."""


class MatchingThresholdsNotConfiguredError(MatchingError):
    """Ranking is possible, but no explicit pair of decision thresholds is configured."""


class MatchStatus(StrEnum):
    KNOWN = "Known"
    UNKNOWN = "Unknown"


class RejectionReason(StrEnum):
    BELOW_SIMILARITY_THRESHOLD = "below_similarity_threshold"
    AMBIGUOUS_TOP_CANDIDATES = "ambiguous_top_candidates"


class GalleryCandidate(Protocol):
    identity_key: str
    label: str
    embedding: np.ndarray


@dataclass(frozen=True)
class MatchPolicy:
    similarity_threshold: float
    ambiguity_margin: float
    calibration_status: str = "experimental"

    def __post_init__(self) -> None:
        if not math.isfinite(self.similarity_threshold) or not -1.0 <= self.similarity_threshold <= 1.0:
            raise MatchingError("Similarity threshold must be finite and within [-1, 1]")
        if not math.isfinite(self.ambiguity_margin) or not 0.0 <= self.ambiguity_margin <= 2.0:
            raise MatchingError("Ambiguity margin must be finite and within [0, 2]")
        if self.calibration_status != "experimental":
            raise MatchingError("Milestone thresholds must remain explicitly experimental")


@dataclass(frozen=True)
class CandidateScore:
    identity_key: str
    label: str
    similarity: float


@dataclass(frozen=True)
class RankedCandidates:
    detection_index: int
    candidates: tuple[CandidateScore, ...]

    @property
    def best(self) -> CandidateScore:
        return self.candidates[0]

    @property
    def runner_up(self) -> CandidateScore | None:
        return self.candidates[1] if len(self.candidates) > 1 else None

    @property
    def margin(self) -> float | None:
        runner_up = self.runner_up
        return self.best.similarity - runner_up.similarity if runner_up else None


@dataclass(frozen=True)
class MatchDecision:
    detection_index: int
    status: MatchStatus
    identity_key: str | None
    identity_label: str | None
    best_candidate: CandidateScore
    runner_up: CandidateScore | None
    margin: float | None
    rejection_reason: RejectionReason | None
    similarity_threshold: float
    ambiguity_margin: float
    calibrated: bool = False


class FaceMatcher:
    """Pure ranker/decision-maker; it has no session or model-runtime dependency."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._configured_policy: MatchPolicy | None = None
        if settings is not None and settings.matching_similarity_threshold is not None:
            # Settings validation guarantees both thresholds are configured as a pair.
            self._configured_policy = MatchPolicy(
                similarity_threshold=settings.matching_similarity_threshold,
                ambiguity_margin=settings.matching_ambiguity_margin,  # type: ignore[arg-type]
            )

    def rank(
        self,
        query_embedding: np.ndarray,
        gallery: Sequence[GalleryCandidate],
        *,
        detection_index: int,
    ) -> RankedCandidates:
        if not gallery:
            raise MatchingError("Cannot match against an empty gallery")
        if detection_index < 0:
            raise MatchingError("Detection index must be non-negative")
        try:
            query = validate_normalized_embedding(query_embedding)
        except EmbeddingValidationError as exc:
            raise MatchingError(f"Invalid query embedding: {exc}") from exc

        keys: set[str] = set()
        scored: list[CandidateScore] = []
        for candidate in gallery:
            if not candidate.identity_key or not candidate.label:
                raise MatchingError("Gallery candidate must have a non-empty key and label")
            if candidate.identity_key in keys:
                raise MatchingError("Gallery contains duplicate identity keys")
            keys.add(candidate.identity_key)
            try:
                reference = validate_normalized_embedding(candidate.embedding)
            except EmbeddingValidationError as exc:
                raise MatchingError(
                    f"Invalid gallery embedding for identity {candidate.label!r}: {exc}"
                ) from exc
            similarity = float(np.dot(query.astype(np.float64), reference.astype(np.float64)))
            if not math.isfinite(similarity):
                raise MatchingError("Cosine similarity calculation returned a non-finite value")
            scored.append(
                CandidateScore(
                    identity_key=candidate.identity_key,
                    label=candidate.label,
                    similarity=float(np.clip(similarity, -1.0, 1.0)),
                )
            )
        scored.sort(key=lambda item: (-item.similarity, item.identity_key))
        return RankedCandidates(detection_index=detection_index, candidates=tuple(scored))

    def decide(
        self,
        ranking: RankedCandidates,
        policy: MatchPolicy | None = None,
    ) -> MatchDecision:
        active_policy = policy or self._configured_policy
        if active_policy is None:
            raise MatchingThresholdsNotConfiguredError(
                "Matching thresholds are unset; only candidate ranking is available"
            )
        if not ranking.candidates:
            raise MatchingError("Cannot decide from an empty candidate ranking")

        best = ranking.best
        runner_up = ranking.runner_up
        margin = ranking.margin
        reason: RejectionReason | None = None
        if best.similarity < active_policy.similarity_threshold:
            reason = RejectionReason.BELOW_SIMILARITY_THRESHOLD
        elif runner_up is not None and (
            margin is None
            or math.isclose(margin, 0.0, rel_tol=0.0, abs_tol=1e-12)
            or margin < active_policy.ambiguity_margin
        ):
            # Equal best/runner-up scores remain ambiguous even at zero margin.
            reason = RejectionReason.AMBIGUOUS_TOP_CANDIDATES

        if reason is not None:
            return MatchDecision(
                detection_index=ranking.detection_index,
                status=MatchStatus.UNKNOWN,
                identity_key=None,
                identity_label=None,
                best_candidate=best,
                runner_up=runner_up,
                margin=margin,
                rejection_reason=reason,
                similarity_threshold=active_policy.similarity_threshold,
                ambiguity_margin=active_policy.ambiguity_margin,
                calibrated=False,
            )
        return MatchDecision(
            detection_index=ranking.detection_index,
            status=MatchStatus.KNOWN,
            identity_key=best.identity_key,
            identity_label=best.label,
            best_candidate=best,
            runner_up=runner_up,
            margin=margin,
            rejection_reason=None,
            similarity_threshold=active_policy.similarity_threshold,
            ambiguity_margin=active_policy.ambiguity_margin,
            calibrated=False,
        )

    def match(
        self,
        query_embedding: np.ndarray,
        gallery: Sequence[GalleryCandidate],
        *,
        detection_index: int,
        policy: MatchPolicy | None = None,
    ) -> MatchDecision:
        ranking = self.rank(query_embedding, gallery, detection_index=detection_index)
        return self.decide(ranking, policy)
