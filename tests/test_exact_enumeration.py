"""Small-scale version of ``demos/lifeo2_exact_enumeration``: Monte Carlo with
the sublattice swap policy must reproduce exact Boltzmann statistics of a
system small enough to enumerate (rock-salt LiFeO2, 8 cation sites, 70
orderings), and symmetry classes must agree with configuration_degeneracy."""

import itertools

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from disorder_mc import (
    KB_EV_PER_K,
    configuration_degeneracy,
    disordered_ewald_energy,
    run_monte_carlo,
    site_permutations,
    sublattice_swap_policy,
)


@pytest.fixture(scope="module")
def enumerated():
    parent = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.158), [{"Li+": 0.5, "Fe3+": 0.5}, "O2-"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    parent.make_supercell([1, 1, 2])
    cations = [i for i, site in enumerate(parent) if not site.is_ordered]
    orderings = []
    for li_sites in itertools.combinations(range(len(cations)), len(cations) // 2):
        structure = parent.copy()
        for a, i in enumerate(cations):
            structure.replace(i, "Li+" if a in li_sites else "Fe3+")
        orderings.append(structure)
    energies = np.array([disordered_ewald_energy(s) for s in orderings])
    return parent, cations, orderings, energies


def _occupation(structure, cations):
    return tuple(structure[i].specie.symbol == "Li" for i in cations)


def test_symmetry_classes_match_configuration_degeneracy(enumerated):
    parent, cations, orderings, energies = enumerated
    perms = site_permutations(parent)
    classes: dict[frozenset, list[int]] = {}
    for k, structure in enumerate(orderings):
        labels = [site.species_string for site in structure]
        orbit = frozenset(tuple(labels[p[j]] for j in range(len(labels))) for p in perms)
        classes.setdefault(orbit, []).append(k)

    assert sum(len(members) for members in classes.values()) == len(orderings) == 70
    for orbit, members in classes.items():
        assert len(members) == len(orbit)
        assert configuration_degeneracy(parent, orderings[members[0]], perms=perms) == len(members)
        np.testing.assert_allclose(energies[members], energies[members[0]], atol=1e-8)


def test_monte_carlo_reproduces_exact_boltzmann_statistics(enumerated):
    parent, cations, orderings, energies = enumerated
    energy_of = {_occupation(s, cations): e for s, e in zip(orderings, energies)}

    def evaluate(structure, context):
        return {"energy": energy_of[_occupation(structure, cations)]}

    # Several levels are populated, but the distribution is far from the
    # infinite-temperature (accept-everything) limit.
    temperature = 0.1 * np.ptp(energies) / KB_EV_PER_K
    weights = np.exp(-(energies - energies.min()) / (KB_EV_PER_K * temperature))
    exact_mean = float((weights * energies).sum() / weights.sum())
    exact_var = float((weights * energies**2).sum() / weights.sum() - exact_mean**2)
    tolerance = 0.25 * np.sqrt(exact_var)
    assert energies.mean() - exact_mean > 5 * tolerance

    policy = sublattice_swap_policy(parent, evaluate_props=evaluate)
    trajectory = run_monte_carlo(orderings[0], policy, temperature, num_steps=3000, num_structures_to_save=0, seed=0)
    sampled = np.array(trajectory.energies[300:])

    assert abs(sampled.mean() - exact_mean) < tolerance
    assert sampled.var() == pytest.approx(exact_var, rel=0.25)
