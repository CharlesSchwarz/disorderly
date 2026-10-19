import itertools
from collections import Counter
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from disorder_mc.structures import count_configurations
from disorder_mc.symmetry import (
    configuration_degeneracy,
    count_distinct_configurations,
    describe_wyckoff_sites,
    get_spacegroup,
    get_wyckoff_labels,
    get_wyckoff_sites,
    label_wyckoff_sites,
    site_permutations,
    symmetrically_equivalent,
)

TEST_STRUCTURES = Path(__file__).parent / "test_structures"


@pytest.fixture
def rock_salt() -> Structure:
    """Conventional NaCl rock-salt cell, space group Fm-3m (#225)."""
    lattice = Lattice.cubic(5.64)
    species = ["Na+"] * 4 + ["Cl-"] * 4
    coords = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
        [0.5, 0.5, 0.5],
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
    ]
    return Structure(lattice, species, coords)


@pytest.fixture
def two_site_parent() -> Structure:
    """A 2-site disordered parent on a low-symmetry orthorhombic lattice.

    The only nontrivial symmetry operation is the translation that swaps
    the two sites, giving a symmetry group of order 2. This makes the
    expected orbit sizes computable by hand.
    """
    lattice = Lattice.orthorhombic(4.0, 5.0, 6.0)
    return Structure(
        lattice,
        [{"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}],
        [[0, 0, 0], [0.5, 0, 0]],
    )


def test_get_spacegroup_rock_salt(rock_salt):
    info = get_spacegroup(rock_salt)
    assert info.symbol == "Fm-3m"
    assert info.number == 225
    assert info.crystal_system == "cubic"


def test_get_wyckoff_sites_rock_salt(rock_salt):
    wyckoffs = get_wyckoff_sites(rock_salt)
    assert len(wyckoffs) == len(rock_salt)
    # Na and Cl sublattices are each symmetry-equivalent within themselves.
    assert len(set(wyckoffs[:4])) == 1
    assert len(set(wyckoffs[4:])) == 1
    assert wyckoffs[0] != wyckoffs[4]


def test_symmetrically_equivalent_true_for_translated_duplicate():
    lattice = Lattice.cubic(5.64)
    s1 = Structure(lattice, ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s2 = Structure(lattice, ["Na+", "Cl-"], [[0.5, 0.5, 0.5], [0, 0, 0]])
    assert symmetrically_equivalent(s1, s2)


def test_symmetrically_equivalent_false_for_different_species():
    lattice = Lattice.cubic(5.64)
    s1 = Structure(lattice, ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s2 = Structure(lattice, ["K+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert not symmetrically_equivalent(s1, s2)


def test_site_permutations_two_site_parent(two_site_parent):
    perms = site_permutations(two_site_parent, symprec=1e-3)
    assert set(perms) == {(0, 1), (1, 0)}


def test_configuration_degeneracy_heterogeneous_ordering(two_site_parent):
    lattice = two_site_parent.lattice
    ordered_ab = Structure(lattice, ["A", "B"], [[0, 0, 0], [0.5, 0, 0]])
    assert configuration_degeneracy(two_site_parent, ordered_ab) == 2


def test_configuration_degeneracy_homogeneous_ordering(two_site_parent):
    lattice = two_site_parent.lattice
    ordered_aa = Structure(lattice, ["A", "A"], [[0, 0, 0], [0.5, 0, 0]])
    assert configuration_degeneracy(two_site_parent, ordered_aa) == 1


def test_configuration_degeneracy_reuses_precomputed_perms(two_site_parent):
    perms = site_permutations(two_site_parent)
    lattice = two_site_parent.lattice
    ordered_ab = Structure(lattice, ["A", "B"], [[0, 0, 0], [0.5, 0, 0]])
    ordered_ba = Structure(lattice, ["B", "A"], [[0, 0, 0], [0.5, 0, 0]])
    # Both orderings belong to the same orbit -> same degeneracy.
    assert configuration_degeneracy(
        two_site_parent, ordered_ab, perms=perms
    ) == configuration_degeneracy(two_site_parent, ordered_ba, perms=perms)


def test_configuration_degeneracy_rejects_extra_sites(two_site_parent):
    lattice = two_site_parent.lattice
    too_many_sites = Structure(
        lattice, ["A", "B", "A"], [[0, 0, 0], [0.5, 0, 0], [0.25, 0.25, 0.25]]
    )
    with pytest.raises(ValueError):
        configuration_degeneracy(two_site_parent, too_many_sites)


def test_get_wyckoff_labels_rock_salt(rock_salt):
    assert get_wyckoff_labels(rock_salt) == ["4a"] * 4 + ["4b"] * 4


@pytest.mark.parametrize("cell", ["primitive", "supercell"])
def test_get_wyckoff_labels_use_conventional_multiplicity(rock_salt, cell):
    if cell == "primitive":
        structure = SpacegroupAnalyzer(rock_salt).get_primitive_standard_structure()
    else:
        structure = rock_salt.copy()
        structure.make_supercell([2, 1, 1])
    labels = get_wyckoff_labels(structure)
    assert set(labels) == {"4a", "4b"}
    for site, label in zip(structure, labels):
        assert label == ("4a" if site.specie.symbol == "Na" else "4b")


@pytest.mark.parametrize("primitive", [False, True])
def test_get_wyckoff_labels_rhombohedral_uses_hexagonal_axes(primitive):
    structure = Structure.from_spacegroup(
        "R-3m",
        Lattice.hexagonal(2.82, 14.05),
        ["Li+", "Co3+", "O2-"],
        [[0, 0, 0.5], [0, 0, 0], [0, 0, 0.24]],
    )
    if primitive:
        structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
    by_species = {site.specie.symbol: label for site, label in zip(structure, get_wyckoff_labels(structure))}
    assert by_species == {"Li": "3b", "Co": "3a", "O": "6c"}


def test_get_wyckoff_labels_triclinic():
    structure = Structure(
        Lattice.from_parameters(4, 5, 6, 80, 85, 95),
        ["Na", "Cl", "Cl"],
        [[0, 0, 0], [0.3, 0.2, 0.1], [0.7, 0.8, 0.9]],
    )
    assert get_spacegroup(structure).symbol == "P-1"
    assert get_wyckoff_labels(structure) == ["1a", "2i", "2i"]


def test_get_wyckoff_labels_disordered_llzo():
    llzo = Structure.from_file(TEST_STRUCTURES / "cubic_LLZO.cif")
    labels = get_wyckoff_labels(llzo)
    li_labels = Counter(label for site, label in zip(llzo, labels) if not site.is_ordered)
    assert li_labels == {"24d": 24, "48g": 48}


def test_label_wyckoff_sites(rock_salt):
    labelled = label_wyckoff_sites(rock_salt)
    assert labelled.site_properties["wyckoff"] == ["4a"] * 4 + ["4b"] * 4
    assert [site.label for site in labelled] == ["Na_4a"] * 4 + ["Cl_4b"] * 4
    assert "wyckoff" not in rock_salt.site_properties


def test_label_wyckoff_sites_mixed_site():
    structure = Structure(Lattice.cubic(4.0), [{"Li+": 0.5, "Ti4+": 0.3}, "O2-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert [site.label for site in label_wyckoff_sites(structure)] == ["Li-Ti_1a", "O_1b"]


def _table_rows(table: str) -> dict[str, list[str]]:
    """Rows of a describe_wyckoff_sites table keyed by label (header lines skipped)."""
    return {line.split()[0]: line for line in table.splitlines()[2:]}


def test_describe_wyckoff_sites_disordered_llzo():
    llzo = Structure.from_file(TEST_STRUCTURES / "cubic_LLZO.cif")
    rows = _table_rows(describe_wyckoff_sites(llzo))
    assert set(rows) == {"Li_24d", "Li_48g", "La_24c", "Zr_16a", "O_96h"}
    assert rows["Li_24d"].split()[2:4] == ["24", "yes"]
    assert rows["Li_24d"].endswith("13 Li+, 11 vacancy")
    assert rows["Li_48g"].endswith("43 Li+, 5 vacancy")
    assert rows["O_96h"].split()[2:] == ["96", "no", "-"]


def test_describe_wyckoff_sites_ordered(rock_salt):
    table = describe_wyckoff_sites(rock_salt)
    assert table.splitlines()[0] == "Wyckoff sites for Na4 Cl4"
    rows = _table_rows(table)
    assert rows["Na_4a"].split()[1:] == ["Na+", "4", "no", "-"]
    assert rows["Cl_4b"].split()[1:] == ["Cl-", "4", "no", "-"]


def test_describe_wyckoff_sites_pool_spanning_orbits():
    # Two inequivalent Li sites with the same occupancy form one pool for ordering: 1 Li and 1 vacancy
    # over both sites, not over each.
    structure = Structure(
        Lattice.tetragonal(3, 5), [{"Li+": 0.5}, {"Li+": 0.5}, "O2-"], [[0, 0, 0], [0.5, 0.5, 0.3], [0, 0, 0.5]]
    )
    rows = _table_rows(describe_wyckoff_sites(structure))
    assert rows["Li_1a"].endswith("1 Li+, 1 vacancy (pooled with Li_1b)")
    assert rows["Li_1b"].endswith("1 Li+, 1 vacancy (pooled with Li_1a)")


def test_count_distinct_configurations_two_site_parent(two_site_parent):
    # AB and BA are one class.
    assert count_distinct_configurations(two_site_parent) == 1


def test_count_distinct_configurations_matches_brute_force():
    """8 rock-salt cation sites, 4 Li + 4 Fe: compare Burnside with explicit
    orbit enumeration (same cell as tests/test_exact_enumeration.py)."""
    parent = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(4.158), [{"Li+": 0.5, "Fe3+": 0.5}, "O2-"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    parent.make_supercell([1, 1, 2])
    perms = site_permutations(parent)
    cations = [i for i, site in enumerate(parent) if not site.is_ordered]

    orbits = set()
    for li_sites in itertools.combinations(cations, len(cations) // 2):
        labels = ["Li" if i in li_sites else ("Fe" if i in cations else "O") for i in range(len(parent))]
        orbits.add(frozenset(tuple(labels[p[j]] for j in range(len(labels))) for p in perms))

    assert sum(len(orbit) for orbit in orbits) == count_configurations(parent) == 70
    assert count_distinct_configurations(parent, perms=perms) == len(orbits)
