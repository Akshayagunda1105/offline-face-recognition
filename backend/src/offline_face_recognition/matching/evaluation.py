"""In-memory evaluation metrics for explicitly experimental match policies."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from offline_face_recognition.matching.matcher import (
    FaceMatcher,
    GalleryCandidate,
    MatchDecision,
    MatchPolicy,
    MatchStatus,
)


@dataclass(frozen=True)
class EvaluationTrial:
    """One already-detected query and its expected identity (None means unknown)."""

    embedding: np.ndarray
    expected_identity_key: str | None
    detection_index: int


@dataclass(frozen=True)
class TrialOutcome:
    trial_index: int
    expected_identity_key: str | None
    decision: MatchDecision
    correct: bool


@dataclass(frozen=True)
class EvaluationMetrics:
    trial_count: int
    known_count: int
    unknown_count: int
    known_correct: int
    known_false_rejects: int
    known_misidentifications: int
    known_top1_correct: int
    unknown_correct_rejections: int
    unknown_false_accepts: int
    known_false_reject_rate: float | None
    known_misidentification_rate: float | None
    known_top1_rate: float | None
    unknown_false_accept_rate: float | None
    unknown_rejection_rate: float | None


@dataclass(frozen=True)
class EvaluationResult:
    policy: MatchPolicy
    metrics: EvaluationMetrics
    outcomes: tuple[TrialOutcome, ...]
    calibrated: bool = False


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_policy(
    matcher: FaceMatcher,
    gallery: Sequence[GalleryCandidate],
    trials: Iterable[EvaluationTrial],
    policy: MatchPolicy,
) -> EvaluationResult:
    """Evaluate one fixed policy; does not select or validate thresholds.

    Keep enrollment and evaluation captures separate before calling this helper.
    The helper accepts labels in memory and writes neither labels nor embeddings.
    """
    gallery_keys = {candidate.identity_key for candidate in gallery}
    outcomes: list[TrialOutcome] = []
    known_count = unknown_count = known_correct = known_false_rejects = 0
    known_misidentifications = known_top1_correct = 0
    unknown_correct_rejections = unknown_false_accepts = 0

    for index, trial in enumerate(trials):
        expected = trial.expected_identity_key
        if expected is not None and expected not in gallery_keys:
            raise ValueError("Known evaluation identity is absent from the enrollment gallery")
        if expected is None and trial.detection_index < 0:
            raise ValueError("Detection index must be non-negative")

        ranking = matcher.rank(
            trial.embedding,
            gallery,
            detection_index=trial.detection_index,
        )
        decision = matcher.decide(ranking, policy)

        if expected is None:
            unknown_count += 1
            correct = decision.status is MatchStatus.UNKNOWN
            if correct:
                unknown_correct_rejections += 1
            else:
                unknown_false_accepts += 1
        else:
            known_count += 1
            if ranking.best.identity_key == expected:
                known_top1_correct += 1
            correct = decision.status is MatchStatus.KNOWN and decision.identity_key == expected
            if correct:
                known_correct += 1
            elif decision.status is MatchStatus.UNKNOWN:
                known_false_rejects += 1
            else:
                known_misidentifications += 1
        outcomes.append(
            TrialOutcome(
                trial_index=index,
                expected_identity_key=expected,
                decision=decision,
                correct=correct,
            )
        )

    metrics = EvaluationMetrics(
        trial_count=len(outcomes),
        known_count=known_count,
        unknown_count=unknown_count,
        known_correct=known_correct,
        known_false_rejects=known_false_rejects,
        known_misidentifications=known_misidentifications,
        known_top1_correct=known_top1_correct,
        unknown_correct_rejections=unknown_correct_rejections,
        unknown_false_accepts=unknown_false_accepts,
        known_false_reject_rate=_rate(known_false_rejects, known_count),
        known_misidentification_rate=_rate(known_misidentifications, known_count),
        known_top1_rate=_rate(known_top1_correct, known_count),
        unknown_false_accept_rate=_rate(unknown_false_accepts, unknown_count),
        unknown_rejection_rate=_rate(unknown_correct_rejections, unknown_count),
    )
    return EvaluationResult(policy=policy, metrics=metrics, outcomes=tuple(outcomes))


def evaluate_policy_grid(
    matcher: FaceMatcher,
    gallery: Sequence[GalleryCandidate],
    trials: Sequence[EvaluationTrial],
    similarity_thresholds: Sequence[float],
    ambiguity_margins: Sequence[float],
) -> tuple[EvaluationResult, ...]:
    """Return a policy grid for human review; never picks a winner automatically."""
    results = []
    for threshold in similarity_thresholds:
        for margin in ambiguity_margins:
            policy = MatchPolicy(threshold, margin)
            results.append(evaluate_policy(matcher, gallery, trials, policy))
    return tuple(results)
