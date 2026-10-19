import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from disorderly.ordering import ordering_key, structure_from_ordering_key
from disorderly.sampling import random_ordered_configuration


def _species_by_frac(structure):
    return {
        tuple(round(float(x), 8) for x in site.frac_coords): site.species
        for site in structure
    }


def test_ordering_key_round_trip_full_occupancy():
    lattice = Lattice.orthorhombic(4.0, 5.0, 6.0)
    parent = Structure(
        lattice,
        [{"Na+": 0.5, "K+": 0.5}] * 4 + ["Cl-"],
        [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0], [0, 0.5, 0]],
    )
    ordered = random_ordered_configuration(parent, np.random.default_rng(4))

    code = ordering_key(parent, ordered)
    recovered = structure_from_ordering_key(parent, code)

    assert code.startswith("ok1_")
    assert _species_by_frac(recovered) == _species_by_frac(ordered)


def test_ordering_key_distinguishes_different_orderings():
    lattice = Lattice.orthorhombic(4.0, 5.0, 6.0)
    parent = Structure(
        lattice,
        [{"Na+": 0.5, "K+": 0.5}] * 2,
        [[0, 0, 0], [0.5, 0, 0]],
    )
    left = Structure(lattice, ["Na+", "K+"], [[0, 0, 0], [0.5, 0, 0]])
    right = Structure(lattice, ["K+", "Na+"], [[0, 0, 0], [0.5, 0, 0]])

    assert ordering_key(parent, left) != ordering_key(parent, right)


def test_ordering_key_round_trip_with_vacancies():
    lattice = Lattice.orthorhombic(4.0, 5.0, 6.0)
    parent = Structure(
        lattice,
        [{"Li+": 0.5}] * 4 + ["O2-"],
        [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0], [0, 0.5, 0]],
    )
    ordered = random_ordered_configuration(parent, np.random.default_rng(0))

    code = ordering_key(parent, ordered)
    recovered = structure_from_ordering_key(parent, code)

    assert len(recovered) == len(ordered)
    assert _species_by_frac(recovered) == _species_by_frac(ordered)


def test_ordering_key_rejects_species_not_allowed_by_parent():
    lattice = Lattice.orthorhombic(4.0, 5.0, 6.0)
    parent = Structure(lattice, [{"Na+": 0.5, "K+": 0.5}], [[0, 0, 0]])
    ordered = Structure(lattice, ["Li+"], [[0, 0, 0]])

    with pytest.raises(ValueError, match="not allowed"):
        ordering_key(parent, ordered)


def test_ordering_key_rejects_missing_fully_occupied_site():
    lattice = Lattice.orthorhombic(4.0, 5.0, 6.0)
    parent = Structure(lattice, [{"Na+": 0.5, "K+": 0.5}], [[0, 0, 0]])
    ordered = Structure(lattice, [], [])

    with pytest.raises(ValueError, match="missing"):
        ordering_key(parent, ordered)
