import sys

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from disorderly.enumeration import enumerate_ordered_configurations
from disorderly.ordering import ordering_key, structure_from_ordering_key
from disorderly.structures import count_configurations, disordered_site_groups
from disorderly.symmetry import (
    _occupations_on_parent,
    configuration_degeneracy,
    count_distinct_configurations,
    site_permutations,
)


def _cubic_supercell(species, scaling):
    structure = Structure(Lattice.cubic(3.0), [species], [[0, 0, 0]])
    structure.make_supercell(scaling)
    return structure


@pytest.fixture
def binary_parent():
    return _cubic_supercell({"Li+": 0.5, "Na+": 0.5}, [2, 2, 2])


@pytest.fixture
def rocksalt_parent():
    """Conventional rock-salt cell: disordered Li/Fe cations, fixed O anions."""
    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(4.2),
        [{"Li+": 0.5, "Fe3+": 0.5}, "O2-"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


def _canonical_orbit_label(parent, ordered, perms):
    labels = _occupations_on_parent(parent, ordered)
    return min(tuple(labels[p[j]] for j in range(len(labels))) for p in perms)


# The remaining tests need icet; the missing-dependency test above them does not.


def test_missing_icet_raises_helpful_error(binary_parent, monkeypatch):
    monkeypatch.setitem(sys.modules, "icet", None)
    monkeypatch.setitem(sys.modules, "icet.tools", None)
    with pytest.raises(ModuleNotFoundError, match="icet"):
        enumerate_ordered_configurations(binary_parent)


icet = pytest.importorskip("icet")


@pytest.mark.parametrize(
    "parent",
    [
        _cubic_supercell({"Li+": 0.5, "Na+": 0.5}, [2, 2, 1]),
        _cubic_supercell({"Li+": 0.5, "Na+": 0.5}, [2, 2, 2]),
        _cubic_supercell({"Li+": 0.25}, [2, 2, 2]),  # vacancies
        _cubic_supercell({"Li+": 0.25, "Na+": 0.25, "K+": 0.5}, [2, 2, 1]),
    ],
)
def test_size_one_gives_one_configuration_per_symmetry_orbit(parent):
    perms = site_permutations(parent)
    configurations = list(enumerate_ordered_configurations(parent))

    assert len(configurations) == count_distinct_configurations(parent, perms=perms)
    orbit_labels = {_canonical_orbit_label(parent, c, perms) for c in configurations}
    assert len(orbit_labels) == len(configurations)
    degeneracies = [configuration_degeneracy(parent, c, perms=perms) for c in configurations]
    assert sum(degeneracies) == count_configurations(parent)


def test_composition_and_fixed_sites_preserved(rocksalt_parent):
    configurations = list(enumerate_ordered_configurations(rocksalt_parent))
    assert len(configurations) == count_distinct_configurations(rocksalt_parent)
    for configuration in configurations:
        assert configuration.is_ordered
        assert configuration.composition == rocksalt_parent.composition
        # Oxidation states are kept, so Ewald energies can be evaluated.
        assert {str(sp) for sp in configuration.species} == {"Li+", "Fe3+", "O2-"}


def test_composition_is_fixed_per_site_group():
    # Li appears on two groups with different occupancies; each group must keep its own count.
    parent = Structure(
        Lattice.tetragonal(3.0, 5.0),
        [{"Li+": 0.5, "Na+": 0.5}] * 2 + [{"Li+": 0.25, "K+": 0.75}] * 4,
        [[0, 0, 0], [0.5, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]],
    )
    groups = disordered_site_groups(parent, warn=False)
    configurations = list(enumerate_ordered_configurations(parent))
    assert len(configurations) == count_distinct_configurations(parent)
    for configuration in configurations:
        labels = _occupations_on_parent(parent, configuration)
        for group in groups:
            group_species = sorted(labels[i] for i in group.site_indices)
            expected = sorted(str(sp) for sp, n in group.counts.items() for _ in range(n))
            assert group_species == expected


def test_vacancies_are_absent_sites():
    parent = _cubic_supercell({"Li+": 0.25}, [2, 2, 2])
    for configuration in enumerate_ordered_configurations(parent):
        assert len(configuration) == 2
        assert configuration.composition.reduced_formula == "Li"


def test_size_one_configurations_share_parent_lattice_and_ordering_keys(binary_parent):
    for configuration in enumerate_ordered_configurations(binary_parent):
        assert np.allclose(configuration.lattice.matrix, binary_parent.lattice.matrix)
        code = ordering_key(binary_parent, configuration)
        rebuilt = structure_from_ordering_key(binary_parent, code)
        assert ordering_key(binary_parent, rebuilt) == code


def test_supercell_sizes_on_primitive_fcc(rocksalt_parent):
    primitive = rocksalt_parent.get_primitive_structure()
    assert len(primitive) == 2
    # Known fcc result: at 50/50, two inequivalent orderings with two cations per cell (L1_0, L1_1).
    size_two = list(enumerate_ordered_configurations(primitive, sizes=2))
    assert len(size_two) == 2
    for configuration in size_two:
        assert len(configuration) == 4
        assert configuration.composition == 2 * primitive.composition
    # Incommensurate sizes (1 and 3) are skipped; each ordering appears once across sizes.
    size_four = list(enumerate_ordered_configurations(primitive, sizes=4))
    across = list(enumerate_ordered_configurations(primitive, sizes=range(1, 5)))
    assert len(across) == len(size_two) + len(size_four)


def test_no_commensurate_size_raises(rocksalt_parent):
    primitive = rocksalt_parent.get_primitive_structure()
    with pytest.raises(ValueError, match="supercell_multiplier"):
        enumerate_ordered_configurations(primitive, sizes=[1, 3])


@pytest.mark.parametrize("sizes", [0, -1, [], [1, 0], 1.0, [True]])
def test_invalid_sizes_raise(binary_parent, sizes):
    with pytest.raises(ValueError):
        enumerate_ordered_configurations(binary_parent, sizes=sizes)


def test_parent_not_mutated(binary_parent):
    before = binary_parent.copy()
    list(enumerate_ordered_configurations(binary_parent))
    assert binary_parent == before
    assert binary_parent.site_properties == before.site_properties


def test_ordered_parent_yields_itself():
    parent = Structure(Lattice.cubic(3.0), ["Li+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    configurations = list(enumerate_ordered_configurations(parent))
    assert len(configurations) == 1
    assert configurations[0].composition == parent.composition
