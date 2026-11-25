"""Cosine ranking and conservative known/unknown decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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
    INSUFFICIENT_GALLERY_IDENTITIES = "insufficient_gallery_identities"
    IDENTITY_ALREADY_ASSIGNED = "identity_already_assigned"
    AMBIGUOUS_GROUP_ASSIGNMENT = "ambiguous_group_assignment"


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
        # Rankings are normally produced by rank(), but decide() is also a
        # public boundary used by evaluation and tests. Validate externally
        # supplied rankings before trusting their order or score values.
        ranking = self._validate_ranking(ranking)

        best = ranking.best
        runner_up = ranking.runner_up
        margin = ranking.margin
        reason: RejectionReason | None = None
        if best.similarity < active_policy.similarity_threshold:
            reason = RejectionReason.BELOW_SIMILARITY_THRESHOLD
        elif runner_up is None:
            # A one-identity gallery has no second score, so the required
            # top-two margin cannot be established. Do not fall back to score-only.
            reason = RejectionReason.INSUFFICIENT_GALLERY_IDENTITIES
        elif (
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

    def match_many(
        self,
        rankings: Sequence[RankedCandidates],
        policy: MatchPolicy | None = None,
    ) -> tuple[MatchDecision, ...]:
        """Resolve one group photo as a deterministic one-to-one assignment.

        The objective is lexicographic: maximize the number of Known identities,
        then maximize their summed cosine similarity. For each proposed assignment,
        the existing threshold and margin policy is reevaluated against identities
        not assigned to another face. Invalid assignments branch either to an
        alternative identity or to assigning their strongest remaining competitor
        elsewhere. This searches for the best complete policy-valid assignment.

        Equivalent optimal assignments are treated as ambiguous. Every face whose
        assignment changes across an equally scoring optimum is left Unknown rather
        than being selected by input/detection order.
        """
        if not rankings:
            return ()
        active_policy = policy or self._configured_policy
        if active_policy is None:
            raise MatchingThresholdsNotConfiguredError(
                "Matching thresholds are unset; only candidate ranking is available"
            )

        validated = tuple(self._validate_ranking(item) for item in rankings)
        detection_indices = [item.detection_index for item in validated]
        if len(set(detection_indices)) != len(detection_indices):
            raise MatchingError("Batch rankings contain duplicate detection indices")

        identity_labels: dict[str, str] = {}
        identity_keys: set[str] | None = None
        score_rows: list[dict[str, CandidateScore]] = []
        for ranking in validated:
            row: dict[str, CandidateScore] = {}
            for candidate in ranking.candidates:
                previous_label = identity_labels.setdefault(
                    candidate.identity_key, candidate.label
                )
                if previous_label != candidate.label:
                    raise MatchingError(
                        "Batch rankings contain inconsistent identity labels"
                    )
                row[candidate.identity_key] = candidate
            keys = set(row)
            if identity_keys is None:
                identity_keys = keys
            elif keys != identity_keys:
                raise MatchingError("Batch rankings do not use the same gallery identities")
            score_rows.append(row)

        keys = tuple(sorted(identity_keys or ()))
        if not keys:
            raise MatchingError("Cannot match against an empty gallery")
        key_index = {key: index for index, key in enumerate(keys)}
        score_matrix = [
            [row[key].similarity for key in keys]
            for row in score_rows
        ]
        count_faces = len(validated)
        count_identities = len(keys)
        # Preserve an independently valid top claim: if multiple faces' ordinary
        # single-face decisions accept the same identity, its highest-scoring
        # claimant owns that identity. Exact claim ties make that identity
        # unavailable; they are never settled by row or detection order.
        claim_scores: dict[str, list[tuple[int, float]]] = {}
        for face, ranking in enumerate(validated):
            initial = self.decide(ranking, active_policy)
            if initial.status is MatchStatus.KNOWN and initial.identity_key is not None:
                claim_scores.setdefault(initial.identity_key, []).append(
                    (face, initial.best_candidate.similarity)
                )
        claim_owner: dict[str, int] = {}
        tied_claims: set[str] = set()
        tied_claim_faces: set[int] = set()
        for identity, claims in claim_scores.items():
            highest = max(score for _, score in claims)
            winners = [
                face
                for face, score in claims
                if math.isclose(score, highest, rel_tol=0.0, abs_tol=1e-12)
            ]
            if len(winners) == 1:
                claim_owner[identity] = winners[0]
            else:
                tied_claims.add(identity)
                tied_claim_faces.update(winners)
        # A valid decision needs at least one runner-up identity left unassigned.
        # At most count_faces - 1 other identities can be consumed by other faces.
        minimum_unassigned = max(1, count_identities - count_faces)
        policy_eligible = [[False] * count_identities for _ in range(count_faces)]
        for face in range(count_faces):
            for identity in range(count_identities):
                score = score_matrix[face][identity]
                if score < active_policy.similarity_threshold:
                    continue
                identity_key = keys[identity]
                if identity_key in tied_claims or (
                    identity_key in claim_owner and claim_owner[identity_key] != face
                ):
                    continue
                lower_candidates = 0
                for other_identity in range(count_identities):
                    if other_identity == identity:
                        continue
                    gap = score - score_matrix[face][other_identity]
                    if (
                        gap >= active_policy.ambiguity_margin
                        and not math.isclose(gap, 0.0, rel_tol=0.0, abs_tol=1e-12)
                    ):
                        lower_candidates += 1
                # Even if every other face receives an identity, this many
                # qualifying alternatives must remain available as the runner-up.
                policy_eligible[face][identity] = lower_candidates >= minimum_unassigned
        cardinality_bonus = 2 * count_faces + 2
        blocked_weight = -(cardinality_bonus * (count_faces + count_identities + 2))

        def raw_assignment(
            forced: dict[int, int],
            forbidden: frozenset[tuple[int, int]],
            excluded_faces: frozenset[int],
        ) -> dict[int, int] | None:
            forced_identities = list(forced.values())
            if len(set(forced_identities)) != len(forced_identities):
                return None
            if any(face in excluded_faces for face in forced):
                return None
            if any(
                not policy_eligible[face][identity]
                or (face, identity) in forbidden
                for face, identity in forced.items()
            ):
                return None

            assignment = dict(forced)
            remaining_faces = [
                face
                for face in range(count_faces)
                if face not in forced and face not in excluded_faces
            ]
            remaining_identities = [
                identity
                for identity in range(count_identities)
                if identity not in forced_identities
            ]
            if not remaining_faces:
                return assignment

            weights: list[list[float]] = []
            for face in remaining_faces:
                row = []
                for identity in remaining_identities:
                    if (
                        policy_eligible[face][identity]
                        and (face, identity) not in forbidden
                    ):
                        row.append(cardinality_bonus + score_matrix[face][identity])
                    else:
                        row.append(blocked_weight)
                # One dummy column per remaining face represents an Unknown result.
                row.extend([0.0] * len(remaining_faces))
                weights.append(row)

            columns = _maximum_weight_assignment(weights)
            for row_index, column in enumerate(columns):
                if column < len(remaining_identities):
                    face = remaining_faces[row_index]
                    identity = remaining_identities[column]
                    if (
                        policy_eligible[face][identity]
                        and (face, identity) not in forbidden
                    ):
                        assignment[face] = identity
            return assignment

        def objective(assignment: dict[int, int]) -> tuple[int, float]:
            return (
                len(assignment),
                math.fsum(score_matrix[face][identity] for face, identity in assignment.items()),
            )

        def better(left: tuple[int, float], right: tuple[int, float]) -> bool:
            return left[0] > right[0] or (
                left[0] == right[0] and left[1] > right[1] + 1e-12
            )

        def equivalent(left: tuple[int, float], right: tuple[int, float]) -> bool:
            return left[0] == right[0] and math.isclose(
                left[1], right[1], rel_tol=0.0, abs_tol=1e-12
            )

        def solve(
            initial_forbidden: frozenset[tuple[int, int]] = frozenset(),
            excluded_faces: frozenset[int] = frozenset(),
        ) -> tuple[dict[int, int], tuple[int, float]]:
            best_assignment: dict[int, int] = {}
            best_objective = (-1, -math.inf)
            visited: set[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[int, ...]]] = set()

            def search(
                forced: dict[int, int],
                forbidden: frozenset[tuple[int, int]],
                excluded: frozenset[int],
            ) -> None:
                nonlocal best_assignment, best_objective
                state = (
                    tuple(sorted(forced.items())),
                    tuple(sorted(forbidden)),
                    tuple(sorted(excluded)),
                )
                if state in visited:
                    return
                visited.add(state)

                assignment = raw_assignment(forced, forbidden, excluded)
                if assignment is None:
                    return
                current_objective = objective(assignment)
                if current_objective[0] < best_objective[0] or (
                    current_objective[0] == best_objective[0]
                    and current_objective[1] < best_objective[1] - 1e-12
                ):
                    return

                used = set(assignment.values())
                failed: tuple[int, int, int | None] | None = None
                # Face iteration only chooses a branch order; all valid branches are
                # explored, and tied optima are rejected below, so it cannot break ties.
                for face in sorted(assignment):
                    assigned_identity = assignment[face]
                    alternatives = [
                        identity
                        for identity in range(count_identities)
                        if identity == assigned_identity or identity not in used
                    ]
                    alternatives.sort(
                        key=lambda identity: (
                            -score_matrix[face][identity],
                            keys[identity],
                        )
                    )
                    if alternatives[0] != assigned_identity:
                        failed = (face, assigned_identity, alternatives[0])
                        break
                    if len(alternatives) < 2:
                        failed = (face, assigned_identity, None)
                        break
                    margin = (
                        score_matrix[face][assigned_identity]
                        - score_matrix[face][alternatives[1]]
                    )
                    if (
                        math.isclose(margin, 0.0, rel_tol=0.0, abs_tol=1e-12)
                        or margin < active_policy.ambiguity_margin
                    ):
                        failed = (face, assigned_identity, alternatives[1])
                        break

                if failed is None:
                    if better(current_objective, best_objective):
                        best_assignment = assignment
                        best_objective = current_objective
                    return

                face, assigned_identity, blocker = failed
                # Any valid solution must either choose another identity for this
                # face, or consume its strongest unassigned blocker on another face.
                search(
                    forced,
                    forbidden | {(face, assigned_identity)},
                    excluded,
                )
                if blocker is not None:
                    for other_face in range(count_faces):
                        if (
                            other_face == face
                            or other_face in excluded
                            or (other_face in forced and forced[other_face] != blocker)
                            or (other_face, blocker) in forbidden
                            or not policy_eligible[other_face][blocker]
                        ):
                            continue
                        next_forced = dict(forced)
                        next_forced[other_face] = blocker
                        search(next_forced, forbidden, excluded)
                else:
                    # There is no runner-up only when every identity is already
                    # assigned. Free one by leaving any currently assigned face Unknown.
                    for other_face in assignment:
                        search(forced, forbidden, excluded | {other_face})

            search({}, initial_forbidden, excluded_faces)
            return best_assignment, best_objective

        assignments, best_objective = solve()
        ambiguous_faces: set[int] = set()
        # If another globally optimal valid assignment exists, mark every face
        # whose identity changes across those optima Unknown. No row-order tiebreak.
        for face, identity in tuple(assignments.items()):
            alternative, alternative_objective = solve(
                frozenset({(face, identity)})
            )
            if equivalent(best_objective, alternative_objective):
                ambiguous_faces.update(
                    changed_face
                    for changed_face in set(assignments) | set(alternative)
                    if assignments.get(changed_face) != alternative.get(changed_face)
                )

        while ambiguous_faces:
            assignments, best_objective = solve(excluded_faces=frozenset(ambiguous_faces))
            newly_ambiguous: set[int] = set()
            for face, identity in tuple(assignments.items()):
                alternative, alternative_objective = solve(
                    frozenset({(face, identity)}),
                    frozenset(ambiguous_faces),
                )
                if equivalent(best_objective, alternative_objective):
                    newly_ambiguous.update(
                        changed_face
                        for changed_face in set(assignments) | set(alternative)
                        if assignments.get(changed_face) != alternative.get(changed_face)
                    )
            newly_ambiguous.difference_update(ambiguous_faces)
            if not newly_ambiguous:
                break
            ambiguous_faces.update(newly_ambiguous)

        used_keys = {keys[identity] for identity in assignments.values()}
        decisions: list[MatchDecision] = []
        for face, ranking in enumerate(validated):
            available = [
                candidate
                for candidate in ranking.candidates
                if candidate.identity_key not in used_keys
                or assignments.get(face) == key_index[candidate.identity_key]
            ]
            available.sort(key=lambda candidate: (-candidate.similarity, candidate.identity_key))
            decision_ranking = RankedCandidates(ranking.detection_index, tuple(available))
            if face in assignments:
                decision = self.decide(decision_ranking, active_policy)
                if (
                    decision.status is not MatchStatus.KNOWN
                    or decision.identity_key != keys[assignments[face]]
                ):
                    raise MatchingError(
                        "Global assignment and policy validation disagree "
                        f"(assigned={keys[assignments[face]]!r}, "
                        f"decision={decision.status.value}:{decision.identity_key!r}, "
                        f"margin={decision.margin!r})"
                    )
                decisions.append(decision)
                continue

            # If every gallery identity was assigned elsewhere, use the original
            # ranking only to populate internal candidate diagnostics safely.
            if not decision_ranking.candidates:
                decision_ranking = ranking
            decision = self.decide(decision_ranking, active_policy)
            if face in ambiguous_faces or face in tied_claim_faces:
                decisions.append(
                    replace(
                        decision,
                        status=MatchStatus.UNKNOWN,
                        identity_key=None,
                        identity_label=None,
                        rejection_reason=RejectionReason.AMBIGUOUS_GROUP_ASSIGNMENT,
                        calibrated=False,
                    )
                )
            elif decision.status is MatchStatus.KNOWN:
                decisions.append(
                    replace(
                        decision,
                        status=MatchStatus.UNKNOWN,
                        identity_key=None,
                        identity_label=None,
                        rejection_reason=RejectionReason.IDENTITY_ALREADY_ASSIGNED,
                        calibrated=False,
                    )
                )
            else:
                decisions.append(decision)
        return tuple(decisions)

    def _validate_ranking(self, ranking: RankedCandidates) -> RankedCandidates:
        """Validate and normalize an externally supplied ranking."""
        if not ranking.candidates:
            raise MatchingError("Cannot decide from an empty candidate ranking")
        if ranking.detection_index < 0:
            raise MatchingError("Detection index must be non-negative")
        seen_keys: set[str] = set()
        previous_similarity = math.inf
        validated_candidates: list[CandidateScore] = []
        for candidate in ranking.candidates:
            if not candidate.identity_key or not candidate.label:
                raise MatchingError("Ranked candidate must have a non-empty key and label")
            if candidate.identity_key in seen_keys:
                raise MatchingError("Ranking contains duplicate identity keys")
            seen_keys.add(candidate.identity_key)
            if isinstance(candidate.similarity, bool):
                raise MatchingError("Ranking contains an invalid cosine similarity score")
            try:
                similarity = float(candidate.similarity)
            except (TypeError, ValueError, OverflowError) as exc:
                raise MatchingError("Ranking contains an invalid cosine similarity score") from exc
            if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
                raise MatchingError("Ranking contains an invalid cosine similarity score")
            if similarity > previous_similarity:
                raise MatchingError("Candidate scores must be sorted in descending order")
            previous_similarity = similarity
            validated_candidates.append(
                CandidateScore(candidate.identity_key, candidate.label, similarity)
            )
        return RankedCandidates(ranking.detection_index, tuple(validated_candidates))


def _maximum_weight_assignment(weights: Sequence[Sequence[float]]) -> list[int]:
    """Return a maximum-weight row-to-column assignment (Hungarian algorithm)."""
    row_count = len(weights)
    if row_count == 0:
        return []
    column_count = len(weights[0])
    if column_count < row_count or any(len(row) != column_count for row in weights):
        raise MatchingError("Assignment matrix must be rectangular with at least as many columns as rows")

    # Hungarian potentials solve a minimum-cost assignment; negate weights.
    costs = [[-float(value) for value in row] for row in weights]
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        current_column = 0
        minimum = [math.inf] * (column_count + 1)
        used_columns = [False] * (column_count + 1)
        while True:
            used_columns[current_column] = True
            current_row = matched_row[current_column]
            delta = math.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used_columns[column]:
                    continue
                reduced_cost = costs[current_row - 1][column - 1] - u[current_row] - v[column]
                if reduced_cost < minimum[column]:
                    minimum[column] = reduced_cost
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            if not math.isfinite(delta):
                raise MatchingError("Global assignment could not find a feasible augmenting path")
            for column in range(column_count + 1):
                if used_columns[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while True:
            previous_column = predecessor[current_column]
            matched_row[current_column] = matched_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            assignment[matched_row[column] - 1] = column - 1
    return assignment
