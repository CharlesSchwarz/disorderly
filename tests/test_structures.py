import math
from pathlib import Path
import warnings

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import get_el_sp

from disorderly.structures import (
    count_configurations,
    disordered_site_groups,
    disordered_site_indices,
    is_disordered,
    replace_species_and_occupancy,
    set_site_occupancies,
    supercell_multiplier,
)

TEST_STRUCTURES = Path(__file__).parent / "test_structures"


@pytest.fixture
def mixed_site_structure() -> Structure:
    """A cubic 2-site structure with one mixed-occupancy site."""
    lattice = Lattice.cubic(4.0)
    return Structure(
        lattice,
        [{"Li+": 0.5, "Fe3+": 0.0625}, "O2-"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture
def ordered_structure() -> Structure:
    lattice = Lattice.cubic(4.0)
    return Structure(lattice, ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_is_disordered(mixed_site_structure, ordered_structure):
    assert is_disordered(mixed_site_structure)
    assert not is_disordered(ordered_structure)


def test_disordered_site_indices(mixed_site_structure, ordered_structure):
    assert disordered_site_indices(mixed_site_structure) == [0]
    assert disordered_site_indices(ordered_structure) == []


def test_replace_preserves_co_occupying_species(mixed_site_structure):
    new_struct = replace_species_and_occupancy(mixed_site_structure, "Li+", "Na+")
    species = dict(new_struct[0].species.items())
    from pymatgen.core.periodic_table import get_el_sp

    assert species[get_el_sp("Na+")] == pytest.approx(0.5)
    assert species[get_el_sp("Fe3+")] == pytest.approx(0.0625)
    assert get_el_sp("Li+") not in species


def test_replace_does_not_mutate_input(mixed_site_structure):
    original_species = dict(mixed_site_structure[0].species.items())
    replace_species_and_occupancy(mixed_site_structure, "Li+", "Na+")
    assert dict(mixed_site_structure[0].species.items()) == original_species


def test_replace_default_occupancy_preserves_value(mixed_site_structure):
    new_struct = replace_species_and_occupancy(mixed_site_structure, "Li+", "Na+")
    from pymatgen.core.periodic_table import get_el_sp

    assert new_struct[0].species[get_el_sp("Na+")] == pytest.approx(0.5)


def test_replace_explicit_new_occupancy(mixed_site_structure):
    new_struct = replace_species_and_occupancy(
        mixed_site_structure, "Li+", "Na+", new_occupancy=0.3
    )
    from pymatgen.core.periodic_table import get_el_sp

    assert new_struct[0].species[get_el_sp("Na+")] == pytest.approx(0.3)


def test_replace_target_occupancy_filter(mixed_site_structure):
    # Matching occupancy: replacement happens.
    new_struct = replace_species_and_occupancy(
        mixed_site_structure, "Li+", "Na+", target_occupancy=0.5
    )
    from pymatgen.core.periodic_table import get_el_sp

    assert get_el_sp("Na+") in dict(new_struct[0].species.items())

    # Non-matching occupancy: nothing found, raises.
    with pytest.raises(ValueError):
        replace_species_and_occupancy(mixed_site_structure, "Li+", "Na+", target_occupancy=0.9)


def test_replace_missing_species_raises(mixed_site_structure):
    with pytest.raises(ValueError):
        replace_species_and_occupancy(mixed_site_structure, "K+", "Na+")


def test_replace_overflowing_occupancy_raises(mixed_site_structure):
    with pytest.raises(ValueError):
        replace_species_and_occupancy(mixed_site_structure, "Fe3+", "Na+", new_occupancy=0.99)


def test_replace_ordered_site(ordered_structure):
    new_struct = replace_species_and_occupancy(ordered_structure, "Na+", "K+")
    from pymatgen.core.periodic_table import get_el_sp

    assert new_struct[0].species == {get_el_sp("K+"): 1.0}
    # Unrelated site untouched.
    assert ordered_structure[1].species_string == new_struct[1].species_string


@pytest.fixture(scope="module")
def llzo() -> Structure:
    """Cubic LLZO: 13 Li on 24 tetrahedral (24d) and 43 Li on 48 octahedral (48g) sites."""
    return Structure.from_file(TEST_STRUCTURES / "cubic_LLZO.cif")


@pytest.fixture(scope="module")
def lto() -> Structure:
    """Spinel LTO conventional cell: 16 sites of {Li+: 1/6, Ti4+: 5/6}."""
    return Structure.from_file(TEST_STRUCTURES / "LTO_COD_1001098.cif")


def test_disordered_site_groups_llzo_counts(llzo):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        groups = disordered_site_groups(llzo)
    li = get_el_sp("Li+")
    assert [group.n_sites for group in groups] == [24, 48]
    assert [group.counts for group in groups] == [{li: 13, None: 11}, {li: 43, None: 5}]
    assert all(group.commensurate for group in groups)
    assert groups[0].ideal_counts[li] == pytest.approx(13)
    assert sum(group.n_sites for group in groups) == len(disordered_site_indices(llzo))


def test_disordered_site_groups_warns_with_multiplier(lto):
    with pytest.warns(UserWarning, match="Multiply the cell by 3"):
        (group,) = disordered_site_groups(lto)
    assert not group.commensurate
    assert sum(group.counts.values()) == group.n_sites == 16
    assert supercell_multiplier(lto) == 3


def test_disordered_site_groups_supercell_fixes_counts(lto):
    supercell = lto.copy()
    supercell.make_supercell([3, 1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        (group,) = disordered_site_groups(supercell)
    assert group.counts == {get_el_sp("Li+"): 8, get_el_sp("Ti4+"): 40}
    assert supercell_multiplier(supercell) == 1


def test_disordered_site_groups_can_suppress_warning(lto):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        disordered_site_groups(lto, warn=False)


def test_count_configurations_llzo(llzo):
    assert count_configurations(llzo) == math.comb(24, 13) * math.comb(48, 43)


def test_count_configurations_multinomial_with_vacancies():
    lattice = Lattice.orthorhombic(4.0, 5.0, 6.0)
    parent = Structure(lattice, [{"A": 0.5, "B": 0.25}] * 4, [[x, 0, 0] for x in (0, 0.25, 0.5, 0.75)])
    # 2 A, 1 B, 1 vacancy on 4 sites: 4! / (2! 1! 1!) = 12.
    assert count_configurations(parent) == 12


def test_set_site_occupancies_adds_dopant(llzo):
    tetrahedral = disordered_site_groups(llzo)[0].site_indices
    doped = set_site_occupancies(llzo, tetrahedral, {"Li+": 7 / 24, "Ga3+": 2 / 24})
    assert doped[tetrahedral[0]].species[get_el_sp("Ga3+")] == pytest.approx(2 / 24)
    assert doped.charge == pytest.approx(0, abs=1e-8)
    assert disordered_site_groups(doped)[0].counts == {
        get_el_sp("Ga3+"): 2,
        get_el_sp("Li+"): 7,
        None: 15,
    }
    # Input untouched.
    assert get_el_sp("Ga3+") not in llzo[tetrahedral[0]].species


def test_set_site_occupancies_rejects_bad_occupancies(mixed_site_structure):
    with pytest.raises(ValueError):
        set_site_occupancies(mixed_site_structure, [0], {"Li+": 0.7, "Fe3+": 0.4})
    with pytest.raises(ValueError):
        set_site_occupancies(mixed_site_structure, [0], {"Li+": -0.1})
    with pytest.raises(ValueError):
        set_site_occupancies(mixed_site_structure, [0], {"Li+": 0.0})
