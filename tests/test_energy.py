import pytest
from pymatgen.core import Lattice, Structure

from disorder_mc.energy import disordered_ewald_energy, effective_site_charges

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
def nacl() -> Structure:
    lattice = Lattice.cubic(5.64)
    species = ["Na+"] * 4 + ["Cl-"] * 4
    return Structure(lattice, species, ROCK_SALT_COORDS)


@pytest.fixture
def no_oxi_states() -> Structure:
    lattice = Lattice.cubic(5.64)
    species = ["Na"] * 4 + ["Cl"] * 4
    return Structure(lattice, species, ROCK_SALT_COORDS)


def test_ordered_ewald_energy_is_negative_and_finite(nacl):
    energy = disordered_ewald_energy(nacl)
    assert energy < 0  # rock-salt electrostatics is stabilizing
    assert energy == pytest.approx(energy)  # finite (no nan/inf)


def test_effective_site_charges_ordered(nacl):
    charges = effective_site_charges(nacl)
    assert charges == pytest.approx([1, 1, 1, 1, -1, -1, -1, -1])


def test_effective_site_charges_mixed_occupancy():
    lattice = Lattice.cubic(5.64)
    species = [{"Na+": 0.5, "Ca2+": 0.5}] * 4 + ["Cl-"] * 4
    structure = Structure(lattice, species, ROCK_SALT_COORDS)
    charges = effective_site_charges(structure)
    # q_site = sum(occupancy_i * charge_i) = 0.5*1 + 0.5*2 = 1.5
    assert charges[:4] == pytest.approx([1.5, 1.5, 1.5, 1.5])
    assert charges[4:] == pytest.approx([-1, -1, -1, -1])


def test_disordered_ewald_energy_uses_effective_charge():
    """A disordered site whose species all carry the *same* charge as a
    pure ordered site should reproduce the ordered energy exactly, since
    the occupancy-weighted average charge is unchanged."""
    lattice = Lattice.cubic(5.64)
    ordered = Structure(lattice, ["Na+"] * 4 + ["Cl-"] * 4, ROCK_SALT_COORDS)
    disordered = Structure(
        lattice, [{"Na+": 0.5, "K+": 0.5}] * 4 + ["Cl-"] * 4, ROCK_SALT_COORDS
    )
    assert disordered_ewald_energy(disordered) == pytest.approx(
        disordered_ewald_energy(ordered)
    )


def test_missing_oxidation_state_raises_clear_error(no_oxi_states):
    with pytest.raises(ValueError, match="oxidation"):
        disordered_ewald_energy(no_oxi_states)
    with pytest.raises(ValueError, match="oxidation"):
        effective_site_charges(no_oxi_states)


def test_does_not_mutate_input(nacl):
    original_species = [str(site.species) for site in nacl]
    disordered_ewald_energy(nacl)
    assert [str(site.species) for site in nacl] == original_species


@pytest.fixture
def lto_parent() -> Structure:
    from pathlib import Path

    return Structure.from_file(Path(__file__).parent / "test_structures" / "LTO_COD_1001098.cif")


def test_precomputed_ewald_matches_full_ewald_for_ordered_configurations(lto_parent):
    import numpy as np

    from disorder_mc.energy import PrecomputedEwald
    from disorder_mc.sampling import random_ordered_configuration

    model = PrecomputedEwald(lto_parent)
    rng = np.random.default_rng(0)
    for _ in range(3):
        # random_ordered_configuration lists sites in a different order than
        # the parent, which exercises position-based matching.
        configuration = random_ordered_configuration(lto_parent, rng)
        assert model(configuration) == pytest.approx(disordered_ewald_energy(configuration), abs=1e-8)
        assert model.evaluate_props(configuration) == {"energy": model(configuration)}


def test_precomputed_ewald_matches_full_ewald_for_disordered_parent(lto_parent):
    from disorder_mc.energy import PrecomputedEwald

    assert PrecomputedEwald(lto_parent)(lto_parent) == pytest.approx(disordered_ewald_energy(lto_parent), abs=1e-8)


def test_precomputed_ewald_charged_cell_with_vacancy(nacl):
    """A missing site carries zero charge and leaves the cell charged; the
    charged-cell term must match pymatgen's. Convergence parameters differ
    slightly because the full calculation sees one fewer site."""
    from disorder_mc.energy import PrecomputedEwald

    model = PrecomputedEwald(nacl)
    vacant = nacl.copy()
    vacant.remove_sites([0])
    assert model(vacant) == pytest.approx(disordered_ewald_energy(vacant), rel=1e-4)


def test_precomputed_ewald_rejects_off_lattice_structures(nacl):
    from disorder_mc.energy import PrecomputedEwald

    model = PrecomputedEwald(nacl)
    strained = nacl.copy()
    strained.apply_strain(0.01)
    with pytest.raises(ValueError, match="reference lattice"):
        model(strained)
    moved = nacl.copy()
    moved.translate_sites([0], [0.25, 0, 0])
    with pytest.raises(ValueError, match="nearest reference position"):
        model(moved)
