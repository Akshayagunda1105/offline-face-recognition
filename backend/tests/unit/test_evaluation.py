from dataclasses import dataclass

import numpy as np
import pytest

from offline_face_recognition.matching.evaluation import (
    EvaluationTrial,
    evaluate_policy,
    evaluate_policy_grid,
)
from offline_face_recognition.matching.matcher import FaceMatcher, MatchPolicy


@dataclass(frozen=True)
class Candidate:
    identity_key: str
    label: str
    embedding: np.ndarray


def _unit(index):
    return np.eye(1, 128, index, dtype=np.float32)[0]


def _gallery():
    return [Candidate("asha", "Asha", _unit(0)), Candidate("bela", "Bela", _unit(1))]


def test_evaluation_separates_false_accepts_rejects_and_wrong_known_results():
    trials = [
        EvaluationTrial(_unit(0), "asha", 0),
        EvaluationTrial(_unit(1), "bela", 1),
        EvaluationTrial(_unit(2), None, 2),
        EvaluationTrial(_unit(0), "bela", 3),
    ]

    result = evaluate_policy(FaceMatcher(), _gallery(), trials, MatchPolicy(0.8, 0.0))

    assert result.calibrated is False
    assert result.metrics.trial_count == 4
    assert result.metrics.known_count == 3
    assert result.metrics.unknown_count == 1
    assert result.metrics.known_correct == 2
    assert result.metrics.known_false_rejects == 0
    assert result.metrics.known_misidentifications == 1
    assert result.metrics.known_top1_correct == 2
    assert result.metrics.unknown_correct_rejections == 1
    assert result.metrics.unknown_false_accepts == 0
    assert result.metrics.known_false_reject_rate == 0.0
    assert result.metrics.known_misidentification_rate == pytest.approx(1 / 3)
    assert result.metrics.unknown_false_accept_rate == 0.0


def test_policy_grid_reports_all_pairs_without_selecting_or_calibrating_one():
    trials = [EvaluationTrial(_unit(0), "asha", 0), EvaluationTrial(_unit(2), None, 1)]

    results = evaluate_policy_grid(
        FaceMatcher(), _gallery(), trials, similarity_thresholds=[0.2, 0.9], ambiguity_margins=[0.0, 0.1]
    )

    assert len(results) == 4
    assert [result.policy.similarity_threshold for result in results] == [0.2, 0.2, 0.9, 0.9]
    assert all(result.calibrated is False for result in results)


def test_evaluation_rejects_known_labels_missing_from_enrollment_gallery():
    with pytest.raises(ValueError, match="absent from the enrollment"):
        evaluate_policy(
            FaceMatcher(),
            _gallery(),
            [EvaluationTrial(_unit(0), "unknown-person", 0)],
            MatchPolicy(0.5, 0.0),
        )


def test_metrics_with_no_unknown_trials_use_none_for_undefined_rates():
    result = evaluate_policy(
        FaceMatcher(),
        _gallery(),
        [EvaluationTrial(_unit(0), "asha", 0)],
        MatchPolicy(0.8, 0.0),
    )

    assert result.metrics.unknown_false_accept_rate is None
    assert result.metrics.unknown_rejection_rate is None
