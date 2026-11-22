from dataclasses import dataclass

import numpy as np
import pytest

from offline_face_recognition.core.config import Settings
from offline_face_recognition.matching.matcher import (
    FaceMatcher,
    MatchPolicy,
    MatchStatus,
    MatchingError,
    MatchingThresholdsNotConfiguredError,
    RankedCandidates,
    CandidateScore,
    RejectionReason,
)


@dataclass(frozen=True)
class Candidate:
    identity_key: str
    label: str
    embedding: np.ndarray


def _unit(index):
    return np.eye(1, 128, index, dtype=np.float32)[0]


def _candidate(name, index):
    return Candidate(name.casefold(), name, _unit(index))


def test_cosine_ranking_returns_best_runner_up_and_detection_index():
    matcher = FaceMatcher()
    ranked = matcher.rank(
        _unit(0),
        [_candidate("Bela", 1), _candidate("Asha", 0), _candidate("Chet", 2)],
        detection_index=6,
    )

    assert ranked.detection_index == 6
    assert [(item.label, item.similarity) for item in ranked.candidates] == [
        ("Asha", 1.0), ("Bela", 0.0), ("Chet", 0.0)
    ]
    assert ranked.best.label == "Asha"
    assert ranked.runner_up.similarity == 0.0
    assert ranked.margin == 1.0


def test_configured_experimental_threshold_accepts_only_when_both_rules_pass():
    settings = Settings(
        project_root=".",
        yunet_model_path="yunet.onnx",
        sface_model_path="sface.onnx",
        matching_similarity_threshold=0.8,
        matching_ambiguity_margin=0.5,
    )
    matcher = FaceMatcher(settings)
    result = matcher.match(_unit(0), [_candidate("Asha", 0), _candidate("Bela", 1)], detection_index=4)

    assert result.status is MatchStatus.KNOWN
    assert result.identity_label == "Asha"
    assert result.detection_index == 4
    assert result.calibrated is False
    assert result.best_candidate.similarity == pytest.approx(1.0)
    assert result.runner_up.similarity == pytest.approx(0.0)


def test_low_similarity_is_rejected_as_unknown():
    matcher = FaceMatcher()
    result = matcher.match(
        _unit(2),
        [_candidate("Asha", 0), _candidate("Bela", 1)],
        detection_index=0,
        policy=MatchPolicy(0.2, 0.0),
    )

    assert result.status is MatchStatus.UNKNOWN
    assert result.identity_label is None
    assert result.rejection_reason is RejectionReason.BELOW_SIMILARITY_THRESHOLD


def test_equal_top_scores_are_always_ambiguous_even_with_zero_margin():
    matcher = FaceMatcher()
    query = np.zeros(128, np.float32)
    query[0:2] = np.float32(1.0 / np.sqrt(2.0))
    result = matcher.match(
        query,
        [_candidate("Asha", 0), _candidate("Bela", 1)],
        detection_index=3,
        policy=MatchPolicy(0.5, 0.0),
    )

    assert result.status is MatchStatus.UNKNOWN
    assert result.rejection_reason is RejectionReason.AMBIGUOUS_TOP_CANDIDATES
    assert result.margin == pytest.approx(0.0, abs=1e-12)


def test_margin_boundary_is_inclusive_and_single_identity_gallery_is_rejected():
    matcher = FaceMatcher()
    ranking = RankedCandidates(
        detection_index=2,
        candidates=(
            CandidateScore("asha", "Asha", 0.75),
            CandidateScore("bela", "Bela", 0.50),
        ),
    )
    result = matcher.decide(ranking, MatchPolicy(0.5, 0.25))
    single = matcher.match(
        _unit(0), [_candidate("Asha", 0)], detection_index=9, policy=MatchPolicy(1.0, 2.0)
    )

    assert result.status is MatchStatus.KNOWN
    assert result.margin == 0.25
    assert single.status is MatchStatus.UNKNOWN
    assert single.runner_up is None
    assert single.margin is None
    assert single.detection_index == 9
    assert single.rejection_reason is RejectionReason.INSUFFICIENT_GALLERY_IDENTITIES


@pytest.mark.parametrize(
    ("best", "runner_up", "expected_status", "expected_reason"),
    [
        (0.55, 0.40, MatchStatus.KNOWN, None),  # both conditions pass
        (0.70, 0.65, MatchStatus.UNKNOWN, RejectionReason.AMBIGUOUS_TOP_CANDIDATES),
        (0.39, 0.20, MatchStatus.UNKNOWN, RejectionReason.BELOW_SIMILARITY_THRESHOLD),
        (0.39, 0.35, MatchStatus.UNKNOWN, RejectionReason.BELOW_SIMILARITY_THRESHOLD),
    ],
)
def test_both_similarity_and_margin_conditions_are_required(
    best, runner_up, expected_status, expected_reason
):
    ranking = RankedCandidates(
        detection_index=5,
        candidates=(
            CandidateScore("top-key", "Top label", best),
            CandidateScore("runner-key", "Runner label", runner_up),
        ),
    )

    decision = FaceMatcher().decide(ranking, MatchPolicy(0.40, 0.10))

    assert decision.status is expected_status
    assert decision.rejection_reason is expected_reason
    assert decision.detection_index == 5


def test_similarity_and_margin_exact_boundaries_are_inclusive():
    ranking = RankedCandidates(
        detection_index=0,
        candidates=(
            CandidateScore("asha", "Asha", 0.40),
            CandidateScore("bela", "Bela", 0.30),
        ),
    )

    decision = FaceMatcher().decide(ranking, MatchPolicy(0.40, 0.10))

    assert decision.status is MatchStatus.KNOWN
    assert decision.identity_key == "asha"
    assert decision.margin == pytest.approx(0.10)


def test_high_similarity_and_large_margin_unknown_like_probe_is_not_rejected_by_this_policy():
    # Similarity alone cannot reveal that a query is actually an unknown person.
    # Such a probe passing both policy checks demonstrates the open-set limitation.
    ranking = RankedCandidates(
        detection_index=1,
        candidates=(
            CandidateScore("nearest", "Nearest gallery identity", 0.91),
            CandidateScore("runner-up", "Runner-up identity", 0.20),
        ),
    )

    decision = FaceMatcher().decide(ranking, MatchPolicy(0.40, 0.10))

    assert decision.status is MatchStatus.KNOWN
    assert decision.identity_label == "Nearest gallery identity"


def test_identity_label_remains_attached_to_its_score_after_ranking():
    query = np.zeros(128, dtype=np.float32)
    query[0] = np.float32(0.6)
    query[1] = np.float32(0.8)
    gallery = [
        Candidate("second", "Second identity", _unit(1)),
        Candidate("first", "First identity", _unit(0)),
    ]

    ranking = FaceMatcher().rank(query, gallery, detection_index=2)

    assert [(item.label, item.similarity) for item in ranking.candidates] == [
        ("Second identity", pytest.approx(0.8)),
        ("First identity", pytest.approx(0.6)),
    ]


@pytest.mark.parametrize(
    "bad_score", [float("nan"), float("inf"), -1.01, 1.01, "not-a-score", True]
)
def test_decision_rejects_invalid_candidate_scores(bad_score):
    ranking = RankedCandidates(
        detection_index=0,
        candidates=(CandidateScore("asha", "Asha", bad_score),),
    )

    with pytest.raises(MatchingError, match="invalid cosine similarity"):
        FaceMatcher().decide(ranking, MatchPolicy(0.40, 0.10))


def test_decision_rejects_unsorted_or_duplicate_candidate_rankings():
    matcher = FaceMatcher()
    policy = MatchPolicy(0.40, 0.10)
    unsorted = RankedCandidates(
        0,
        (CandidateScore("asha", "Asha", 0.4), CandidateScore("bela", "Bela", 0.8)),
    )
    duplicate = RankedCandidates(
        0,
        (CandidateScore("asha", "Asha", 0.8), CandidateScore("asha", "Asha 2", 0.2)),
    )

    with pytest.raises(MatchingError, match="descending order"):
        matcher.decide(unsorted, policy)
    with pytest.raises(MatchingError, match="duplicate identity"):
        matcher.decide(duplicate, policy)


def test_configured_threshold_changes_match_decision_and_defaults_remain_unset():
    permissive = Settings(
        project_root=".",
        yunet_model_path="yunet.onnx",
        sface_model_path="sface.onnx",
        matching_similarity_threshold=0.40,
        matching_ambiguity_margin=0.10,
    )
    strict = Settings(
        project_root=".",
        yunet_model_path="yunet.onnx",
        sface_model_path="sface.onnx",
        matching_similarity_threshold=0.60,
        matching_ambiguity_margin=0.10,
    )
    query = np.zeros(128, dtype=np.float32)
    query[0] = np.float32(0.5)
    query[1] = np.float32(0.2)
    query[2] = np.float32(np.sqrt(0.71))
    gallery = [_candidate("Asha", 0), _candidate("Bela", 1)]

    assert FaceMatcher(permissive).match(query, gallery, detection_index=0).status is MatchStatus.KNOWN
    assert FaceMatcher(strict).match(query, gallery, detection_index=0).status is MatchStatus.UNKNOWN

    defaults = Settings.from_environment(environ={})
    assert defaults.matching_similarity_threshold is None
    assert defaults.matching_ambiguity_margin is None
    with pytest.raises(MatchingThresholdsNotConfiguredError):
        FaceMatcher(defaults).match(query, gallery, detection_index=0)


def test_ranking_is_available_without_thresholds_but_decision_is_not():
    matcher = FaceMatcher()
    ranking = matcher.rank(_unit(0), [_candidate("Asha", 0)], detection_index=0)

    with pytest.raises(MatchingThresholdsNotConfiguredError):
        matcher.decide(ranking)


@pytest.mark.parametrize(
    "query",
    [np.zeros(127, np.float32), np.full(128, np.nan, np.float32), np.ones(128, np.float32)],
)
def test_invalid_or_non_normalized_query_embeddings_are_rejected(query):
    with pytest.raises(MatchingError, match="Invalid query embedding"):
        FaceMatcher().rank(query, [_candidate("Asha", 0)], detection_index=0)


def test_invalid_gallery_embedding_and_duplicate_key_are_rejected():
    matcher = FaceMatcher()
    with pytest.raises(MatchingError, match="Invalid gallery embedding"):
        matcher.rank(_unit(0), [Candidate("asha", "Asha", np.zeros(128, np.float32))], detection_index=0)
    with pytest.raises(MatchingError, match="duplicate identity"):
        matcher.rank(_unit(0), [_candidate("Asha", 0), _candidate("ASHA", 1)], detection_index=0)


def test_match_policy_rejects_out_of_range_values():
    with pytest.raises(MatchingError):
        MatchPolicy(1.1, 0.1)
    with pytest.raises(MatchingError):
        MatchPolicy(0.5, 2.1)
