import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from disorderly.energy import disordered_ewald_energy
from disorderly.monte_carlo import propose_next_configuration, run_monte_carlo
from disorderly.settings import (
    MonteCarloPolicy,
    MoveProposal,
    default_evaluate_props,
    default_is_move_allowed,
    default_policy,
    preserves_composition,
)

ROCK_SALT_COORDS = [
    [0, 0, 0],
    [0, 0.5, 0.5],
    [0.5, 0, 0.5],
    [0.5, 0.5, 0],
    [0.5, 0.5, 0.5],
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5],
]


@pytest.fixture
def na_k_cl() -> Structure:
    lattice = Lattice.cubic(5.64)
    species = ["Na+", "Na+", "K+", "K+", "Cl-", "Cl-", "Cl-", "Cl-"]
    structure = Structure(lattice, species, ROCK_SALT_COORDS)
    structure.add_oxidation_state_by_element({"Na": 1, "K": 1, "Cl": -1})
    return structure


def test_default_policy_fills_in_defaults():
    policy = default_policy()
    assert policy.is_move_allowed is default_is_move_allowed
    assert policy.evaluate_props is default_evaluate_props
    assert policy.context == {}


def test_default_policy_custom_context_preserved():
    policy = default_policy(context={"foo": "bar"})
    assert policy.context == {"foo": "bar"}


def test_default_choose_potential_move_proposes_swap_of_differing_species(na_k_cl):
    rng = np.random.default_rng(0)
    policy = default_policy()
    move = policy.choose_potential_move(na_k_cl, rng, policy.context)
    assert move is not None
    assert move.move_type == "swap"
    assert na_k_cl[move.site_index].species != na_k_cl[move.target_site_index].species


def test_default_choose_potential_move_none_when_all_sites_identical():
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na+", "Na+"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structure.add_oxidation_state_by_element({"Na": 1})
    rng = np.random.default_rng(0)
    policy = default_policy()
    assert policy.choose_potential_move(structure, rng, policy.context) is None


def test_preserves_composition_true_for_swap(na_k_cl):
    from disorderly.monte_carlo import swap_ion

    swapped = swap_ion(na_k_cl, 0, 2)
    assert preserves_composition(na_k_cl, swapped)


def test_preserves_composition_false_when_species_added():
    lattice = Lattice.cubic(4.0)
    s1 = Structure(lattice, ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s2 = Structure(lattice, ["Na+", "K+"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert not preserves_composition(s1, s2)


def test_default_evaluate_props_matches_direct_ewald_call(na_k_cl):
    props = default_evaluate_props(na_k_cl, {})
    assert props == {"energy": disordered_ewald_energy(na_k_cl)}


def test_custom_policy_is_used_by_engine(na_k_cl):
    """A custom policy overriding all three hooks should drive the engine's
    behavior end to end, demonstrating the injection point works."""
    calls = {"choose": 0, "allowed": 0, "evaluate": 0}

    def choose(configuration, rng, context):
        calls["choose"] += 1
        return MoveProposal("swap", 0, 2)

    def allowed(configuration, proposed, move, context):
        calls["allowed"] += 1
        return True

    def evaluate(configuration, context):
        calls["evaluate"] += 1
        return {"energy": 0.0, "custom_property": 42}

    policy = MonteCarloPolicy(
        choose_potential_move=choose, is_move_allowed=allowed, evaluate_props=evaluate, detailed_balance=True
    )
    rng = np.random.default_rng(0)
    candidate, move, props = propose_next_configuration(na_k_cl, policy, rng)
    assert calls == {"choose": 1, "allowed": 1, "evaluate": 1}
    assert props["custom_property"] == 42

    trajectory = run_monte_carlo(na_k_cl, policy, temperature=300, num_steps=5, seed=0)
    assert all(s.properties.get("custom_property") == 42 for s in trajectory.steps[1:])


@pytest.fixture
def lto_parent() -> Structure:
    from pathlib import Path

    return Structure.from_file(Path(__file__).parent / "test_structures" / "LTO_COD_1001098.cif")


def test_sublattice_swap_policy_only_swaps_disordered_sites(lto_parent):
    """Swaps must stay on the Li/Ti 16d sites even though
    random_ordered_configuration lists sites in a different order than the
    parent, and must never touch O or the fully ordered 8a Li."""
    from disorderly.sampling import random_ordered_configuration
    from disorderly.settings import sublattice_site_indices, sublattice_swap_policy
    from disorderly.structures import disordered_site_indices

    configuration = random_ordered_configuration(lto_parent, np.random.default_rng(0))
    policy = sublattice_swap_policy(lto_parent)
    disordered = disordered_site_indices(lto_parent)
    (on_16d,) = sublattice_site_indices(configuration, [lto_parent.frac_coords[disordered]])
    assert len(on_16d) == len(disordered)

    rng = np.random.default_rng(1)
    for _ in range(200):
        move = policy.choose_potential_move(configuration, rng, policy.context)
        assert move is not None
        assert move.site_index in on_16d and move.target_site_index in on_16d
        assert configuration[move.site_index].species != configuration[move.target_site_index].species
        swapped = {configuration[i].specie.symbol for i in (move.site_index, move.target_site_index)}
        assert swapped == {"Li", "Ti"}


def test_sublattice_swap_policy_run_leaves_other_sites_unchanged(lto_parent):
    from disorderly.sampling import random_ordered_configuration
    from disorderly.settings import sublattice_swap_policy

    configuration = random_ordered_configuration(lto_parent, np.random.default_rng(0))
    policy = sublattice_swap_policy(lto_parent, evaluate_props=lambda s, ctx: {"energy": 0.0})
    trajectory = run_monte_carlo(
        configuration, policy, temperature=1000, num_steps=50, seed=0, record_structure_every=50
    )
    final = trajectory.steps[-1].structure
    for before, after in zip(configuration, final):
        if "O2-" in (before.species_string, after.species_string):
            assert before.species == after.species
    assert final.composition == configuration.composition


def test_sublattice_swap_policy_explicit_groups_restrict_swaps(na_k_cl):
    from disorderly.settings import sublattice_swap_policy

    # Sites 0, 1 (Na) and 2 (K) form one group; site 3 (K) is excluded.
    policy = sublattice_swap_policy(na_k_cl, sublattices=[[0, 1, 2]], evaluate_props=lambda s, ctx: {"energy": 0.0})
    rng = np.random.default_rng(0)
    pairs = set()
    for _ in range(100):
        move = policy.choose_potential_move(na_k_cl, rng, policy.context)
        if move is not None:
            pairs.add(frozenset((move.site_index, move.target_site_index)))
    assert pairs == {frozenset((0, 2)), frozenset((1, 2))}


@pytest.mark.parametrize(
    "sublattices, message",
    [
        ([[0]], "at least two sites"),
        ([[0, 8]], "out of range"),
        ([[0, 1], [1, 2]], "more than one sublattice"),
    ],
)
def test_sublattice_swap_policy_rejects_bad_groups(na_k_cl, sublattices, message):
    from disorderly.settings import sublattice_swap_policy

    with pytest.raises(ValueError, match=message):
        sublattice_swap_policy(na_k_cl, sublattices=sublattices)


def test_sublattice_swap_policy_requires_groups_for_ordered_parent(na_k_cl):
    from disorderly.settings import sublattice_swap_policy

    with pytest.raises(ValueError, match="no disordered sites"):
        sublattice_swap_policy(na_k_cl)
