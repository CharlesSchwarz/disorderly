import warnings

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from disorderly.diagnostics import check_proposal_symmetry
from disorderly.monte_carlo import run_monte_carlo
from disorderly.settings import DetailedBalanceWarning, MonteCarloPolicy, MoveProposal, default_policy

# A "star" of four sites in a 10 A cubic cell: a center with three leaves
# 2 A away. Leaves are >= 2.8 A from each other, so within a 2.5 A hop radius
# the center has three neighbors and each leaf one.
STAR_SITES = np.array([[0.5, 0.5, 0.5], [0.7, 0.5, 0.5], [0.3, 0.5, 0.5], [0.5, 0.7, 0.5]])
HOP_RADIUS = 2.5


@pytest.fixture
def one_li_at_center() -> Structure:
    return Structure(Lattice.cubic(10.0), ["Li+"], [STAR_SITES[0]])


def _occupied(configuration, sites):
    dists = configuration.lattice.get_all_distances(sites, configuration.frac_coords)
    return dists.min(axis=1) < 0.1


def _local_hop(configuration, rng, context):
    """Garnet-reference-style hop: a random Li to a random *vacant neighbor*.
    The number of vacant neighbors differs between sites, so the proposal is
    asymmetric; ``context["correct"]`` reports the Hastings correction."""
    sites = context["sites"]
    lattice = configuration.lattice
    occupied = _occupied(configuration, sites)
    site_index = int(rng.integers(len(configuration)))
    origin = int(np.argmin(lattice.get_all_distances(configuration[site_index].frac_coords, sites)[0]))

    def vacant_neighbors(center, occupied):
        dists = lattice.get_all_distances(sites[center], sites)[0]
        return [j for j in range(len(sites)) if 0.1 < dists[j] <= HOP_RADIUS and not occupied[j]]

    forward = vacant_neighbors(origin, occupied)
    if not forward:
        return None
    target = forward[int(rng.integers(len(forward)))]
    after = occupied.copy()
    after[origin], after[target] = False, True
    reverse = vacant_neighbors(target, after)
    log_ratio = np.log(len(forward) / len(reverse)) if context.get("correct") else 0.0
    return MoveProposal("move", site_index, target_frac_coords=tuple(sites[target]), log_proposal_ratio=log_ratio)


def _global_hop(configuration, rng, context):
    """LLZO-demo-style hop: a random Li to any random vacant site (symmetric)."""
    sites = context["sites"]
    vacant = np.flatnonzero(~_occupied(configuration, sites))
    site_index = int(rng.integers(len(configuration)))
    target = int(vacant[rng.integers(len(vacant))])
    return MoveProposal("move", site_index, target_frac_coords=tuple(sites[target]))


def _no_leaf_to_leaf(configuration, candidate, move, context):
    """Forbid hops between two leaves (only hops via the center are allowed)."""
    lattice = configuration.lattice
    before = configuration[move.site_index].frac_coords
    return lattice.get_distance_and_image(before, STAR_SITES[0])[0] < 0.1 or (
        lattice.get_distance_and_image(move.target_frac_coords, STAR_SITES[0])[0] < 0.1
    )


def _star_policy(choose, correct=False, is_move_allowed=None):
    return MonteCarloPolicy(
        choose_potential_move=choose,
        is_move_allowed=is_move_allowed,
        evaluate_props=lambda configuration, context: {"energy": 0.0},
        context={"sites": STAR_SITES, "correct": correct},
        detailed_balance=True,
    )


# --- check_proposal_symmetry ------------------------------------------------


def test_symmetric_global_hop_passes(one_li_at_center):
    report = check_proposal_symmetry(_star_policy(_global_hop), one_li_at_center, num_proposals=600, seed=0)
    assert report.conclusive
    assert report.satisfied
    assert len(report.pairs) == 3


def test_uncorrected_local_hop_is_flagged(one_li_at_center):
    report = check_proposal_symmetry(_star_policy(_local_hop), one_li_at_center, num_proposals=600, seed=0)
    assert not report.satisfied
    assert len(report.violations) == 3
    for pair in report.violations:
        # Center -> leaf is proposed 1/3 of the time; leaf -> center always.
        assert pair.estimated_log_ratio == pytest.approx(np.log(3), abs=0.3)


def test_hastings_corrected_local_hop_passes(one_li_at_center):
    policy = _star_policy(_local_hop, correct=True)
    report = check_proposal_symmetry(policy, one_li_at_center, num_proposals=600, seed=0)
    assert report.satisfied
    assert all(pair.declared_log_ratio == pytest.approx(np.log(3)) for pair in report.pairs)


def test_redrawing_disallowed_moves_is_flagged_only_in_retry_mode():
    li_on_leaf = Structure(Lattice.cubic(10.0), ["Li+"], [STAR_SITES[1]])
    policy = _star_policy(_global_hop, is_move_allowed=_no_leaf_to_leaf)
    reject = check_proposal_symmetry(policy, li_on_leaf, num_proposals=900, seed=0, disallowed_moves="reject")
    retry = check_proposal_symmetry(policy, li_on_leaf, num_proposals=900, seed=0, disallowed_moves="retry")
    assert reject.satisfied
    assert not retry.satisfied


def test_default_swap_policy_passes():
    lattice = Lattice.cubic(4.0)
    coords = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    structure = Structure(lattice, ["Na+", "Na+", "Na+", "K+"], coords)
    report = check_proposal_symmetry(default_policy(), structure, num_proposals=600, seed=0)
    assert report.satisfied
    assert len(report.pairs) == 3


def test_inconclusive_when_moves_are_too_rare(one_li_at_center):
    report = check_proposal_symmetry(_star_policy(_global_hop), one_li_at_center, num_proposals=20, min_count=50, seed=0)
    assert not report.conclusive
    assert not report.satisfied


def test_rejects_unknown_disallowed_moves_mode(one_li_at_center):
    with pytest.raises(ValueError, match="disallowed_moves"):
        check_proposal_symmetry(_star_policy(_global_hop), one_li_at_center, disallowed_moves="ignore")


# --- the engine samples the right distribution ------------------------------


def _fraction_at_center(policy, structure, num_steps=3000, **kwargs):
    trajectory = run_monte_carlo(
        structure, policy, temperature=1000, num_steps=num_steps, seed=1, record_structure_every=1, **kwargs
    )
    lattice = structure.lattice
    at_center = [lattice.get_distance_and_image(s.structure[0].frac_coords, STAR_SITES[0])[0] < 0.1
                 for s in trajectory.steps[1:]]
    return np.mean(at_center)


def test_flat_energy_uniform_with_hastings_correction(one_li_at_center):
    # With equal energies, every site should be occupied 1/4 of the time.
    assert _fraction_at_center(_star_policy(_local_hop, correct=True), one_li_at_center) == pytest.approx(0.25, abs=0.04)


def test_flat_energy_biased_without_hastings_correction(one_li_at_center):
    # The uncorrected local hop is a random walk weighted by each site's
    # number of neighbors: the center (3 of 6) is occupied half the time.
    assert _fraction_at_center(_star_policy(_local_hop), one_li_at_center) == pytest.approx(0.5, abs=0.04)


def test_flat_energy_uniform_when_disallowed_moves_rejected(one_li_at_center):
    policy = _star_policy(_global_hop, is_move_allowed=_no_leaf_to_leaf)
    assert _fraction_at_center(policy, one_li_at_center) == pytest.approx(0.25, abs=0.04)


def test_flat_energy_biased_when_disallowed_moves_retried(one_li_at_center):
    # Redrawing: the center hops to each leaf 1/3 of the time but each leaf
    # always hops back, so the center is occupied half the time.
    policy = _star_policy(_global_hop, is_move_allowed=_no_leaf_to_leaf)
    with pytest.warns(DetailedBalanceWarning, match="redrawn"):
        fraction = _fraction_at_center(policy, one_li_at_center, disallowed_moves="retry")
    assert fraction == pytest.approx(0.5, abs=0.04)


def test_declared_policy_runs_without_warning(one_li_at_center):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetailedBalanceWarning)
        run_monte_carlo(one_li_at_center, _star_policy(_global_hop), temperature=1000, num_steps=5, seed=0)
