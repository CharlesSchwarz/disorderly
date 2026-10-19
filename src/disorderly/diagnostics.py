"""Diagnostics for Monte Carlo policies.

:func:`check_proposal_symmetry` tests empirically whether a policy's
proposals satisfy detailed balance, i.e. whether each move from ``a`` to
``b`` and its reverse are proposed with the probabilities implied by the
move's ``log_proposal_ratio`` (equal probabilities when it is ``0``). The
engine cannot check this from the policy's code, so it relies on the
policy's ``detailed_balance`` declaration; this is how to verify one.

Depends on :mod:`disorderly.monte_carlo` (to build candidates exactly as the
engine does) and :mod:`disorderly.settings`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from pymatgen.core import Structure

from disorderly.monte_carlo import DEFAULT_MOVE_APPLIERS, _check_disallowed_moves, _propose_once
from disorderly.settings import MonteCarloPolicy, MoveProposal

__all__ = ["ProposalPairCheck", "ProposalSymmetryReport", "check_proposal_symmetry"]


@dataclass(frozen=True)
class ProposalPairCheck:
    """The comparison of one move ``a -> b`` with its reverse ``b -> a``.

    Attributes:
        move: A move proposed from ``a`` that produced ``b``.
        forward_count: Proposals of ``b`` among ``forward_trials`` draws from ``a``.
        forward_trials: Draws from ``a`` counted in the forward probability.
        reverse_count: Proposals of ``a`` among ``reverse_trials`` draws from ``b``.
        reverse_trials: Draws from ``b`` counted in the reverse probability.
        declared_log_ratio: ``move.log_proposal_ratio``, the policy's
            ``ln[g(b->a) / g(a->b)]``.
        estimated_log_ratio: The same ratio estimated from the counts (with
            0.5 added to each count, so it stays finite when the reverse
            move was never proposed).
        z_score: ``(estimated - declared) / standard error``. Large
            magnitudes mean the declared ratio is inconsistent with the
            policy's actual proposals.
    """

    move: MoveProposal
    forward_count: int
    forward_trials: int
    reverse_count: int
    reverse_trials: int
    declared_log_ratio: float
    estimated_log_ratio: float
    z_score: float


@dataclass
class ProposalSymmetryReport:
    """Result of :func:`check_proposal_symmetry`.

    Attributes:
        pairs: The move pairs that were tested.
        num_targets: Distinct configurations proposed from the starting
            configuration (only the most frequent are tested).
        z_threshold: ``|z_score|`` above which a pair counts as a violation.
    """

    pairs: list[ProposalPairCheck] = field(default_factory=list)
    num_targets: int = 0
    z_threshold: float = 4.0

    @property
    def violations(self) -> list[ProposalPairCheck]:
        """Pairs whose proposals are inconsistent with detailed balance."""
        return [p for p in self.pairs if abs(p.z_score) > self.z_threshold]

    @property
    def conclusive(self) -> bool:
        """Whether any pair was proposed often enough to be tested."""
        return bool(self.pairs)

    @property
    def satisfied(self) -> bool:
        """Whether at least one pair was tested and none violated detailed balance."""
        return self.conclusive and not self.violations


def _configuration_key(configuration: Structure, decimals: int = 3) -> tuple:
    """Identifies an ordered configuration on a fixed lattice by which species
    sits at which (rounded, wrapped) fractional position, regardless of site
    order."""
    coords = np.round(np.mod(configuration.frac_coords, 1.0), decimals) % 1.0
    return tuple(sorted((site.species_string, *map(float, xyz)) for site, xyz in zip(configuration, coords)))


def _sample_proposals(
    configuration: Structure,
    policy: MonteCarloPolicy,
    rng: np.random.Generator,
    appliers: dict,
    num_proposals: int,
    disallowed_moves: str,
) -> tuple[Counter, dict[tuple, tuple[Structure, MoveProposal]], int]:
    counts: Counter = Counter()
    examples: dict[tuple, tuple[Structure, MoveProposal]] = {}
    trials = 0
    for _ in range(num_proposals):
        candidate, move = _propose_once(configuration, policy, rng, appliers)
        if candidate is None:
            # Under "reject" a disallowed draw is a step that stays put, so it
            # counts towards the trials; "retry" redraws it, so it does not.
            trials += disallowed_moves == "reject"
            continue
        trials += 1
        key = _configuration_key(candidate)
        counts[key] += 1
        examples.setdefault(key, (candidate, move))
    return counts, examples, trials


def check_proposal_symmetry(
    policy: MonteCarloPolicy,
    configuration: Structure,
    num_proposals: int = 2000,
    max_pairs: int = 10,
    min_count: int = 10,
    z_threshold: float = 4.0,
    seed: int | None = None,
    disallowed_moves: Literal["reject", "retry"] = "reject",
    move_appliers: dict[str, Callable[[Structure, MoveProposal], Structure]] | None = None,
) -> ProposalSymmetryReport:
    """Test empirically whether ``policy``'s proposals satisfy detailed balance.

    Draws ``num_proposals`` proposals from ``configuration`` (``a``). For
    each of the (up to ``max_pairs``) most frequently proposed candidates
    ``b`` proposed at least ``min_count`` times, draws ``num_proposals``
    proposals from ``b`` and counts how often ``a`` is proposed back. The
    estimated ``ln[g(b->a) / g(a->b)]`` is compared with each move's declared
    ``log_proposal_ratio``. No energies are evaluated, so the check is cheap
    compared with a run, but each proposal still builds a candidate
    structure.

    Use a small cell: from a large one, each particular move is proposed so
    rarely that few pairs reach ``min_count`` (see
    :attr:`ProposalSymmetryReport.conclusive`). The check covers the
    proposals only. Energies, ergodicity, and configurations other than
    ``configuration`` and its neighbors are not tested.

    Args:
        policy: Policy to test.
        configuration: Ordered configuration to start from.
        num_proposals: Proposals drawn from each configuration.
        max_pairs: Maximum number of move pairs to test.
        min_count: Minimum forward count for a pair to be tested.
        z_threshold: ``|z_score|`` above which a pair is a violation.
        seed: Random seed.
        disallowed_moves: The engine mode to test, as for
            :func:`disorderly.monte_carlo.run_monte_carlo`. With
            ``"retry"``, redrawing disallowed proposals is part of the
            tested proposal distribution.
        move_appliers: Extra move appliers, as for ``run_monte_carlo``.

    Returns:
        A :class:`ProposalSymmetryReport`; check ``report.satisfied`` and
        ``report.violations``.
    """
    _check_disallowed_moves(disallowed_moves)
    rng = np.random.default_rng(seed)
    appliers = {**DEFAULT_MOVE_APPLIERS, **(move_appliers or {})}
    start_key = _configuration_key(configuration)

    forward, examples, forward_trials = _sample_proposals(
        configuration, policy, rng, appliers, num_proposals, disallowed_moves
    )
    report = ProposalSymmetryReport(num_targets=len(forward), z_threshold=z_threshold)
    for key, forward_count in forward.most_common(max_pairs):
        if forward_count < min_count or key == start_key:
            continue
        candidate, move = examples[key]
        reverse, _, reverse_trials = _sample_proposals(candidate, policy, rng, appliers, num_proposals, disallowed_moves)
        reverse_count = reverse.get(start_key, 0)
        estimated = np.log((reverse_count + 0.5) / max(reverse_trials, 1)) - np.log(
            (forward_count + 0.5) / max(forward_trials, 1)
        )
        standard_error = np.sqrt(1 / (forward_count + 0.5) + 1 / (reverse_count + 0.5))
        report.pairs.append(
            ProposalPairCheck(
                move=move,
                forward_count=forward_count,
                forward_trials=forward_trials,
                reverse_count=reverse_count,
                reverse_trials=reverse_trials,
                declared_log_ratio=move.log_proposal_ratio,
                estimated_log_ratio=float(estimated),
                z_score=float((estimated - move.log_proposal_ratio) / standard_error),
            )
        )
    return report
