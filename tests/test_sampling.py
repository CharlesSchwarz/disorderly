import warnings

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from disorderly.sampling import random_energy_sampling, random_ewald_sampling, random_ordered_configuration
from disorderly.symmetry import symmetrically_equivalent
from disorderly.ordering import structure_from_ordering_key


@pytest.fixture
def lattice() -> Lattice:
    return Lattice.orthorhombic(4.0, 5.0, 6.0)


def _species_by_frac(structure):
    return {
        tuple(round(float(x), 8) for x in site.frac_coords): site.species
        for site in structure
    }


def test_random_ordered_configuration_full_occupancy_counts(lattice):
    """4 sites disordered {A:0.75, B:0.25} -> exactly 3 A and 1 B, no vacancies."""
    parent = Structure(
        lattice,
        [{"A": 0.75, "B": 0.25}] * 4,
        [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0]],
    )
    rng = np.random.default_rng(0)
    config = random_ordered_configuration(parent, rng)
    assert len(config) == 4
    counts = config.composition.get_el_amt_dict()
    assert counts == {"A": 3, "B": 1}


def test_random_ordered_configuration_partial_occupancy_leaves_vacancies(lattice):
    """4 sites disordered {A:0.5} -> 2 occupied, 2 left vacant (absent)."""
    parent = Structure(
        lattice,
        [{"A": 0.5}] * 4,
        [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0]],
    )
    rng = np.random.default_rng(0)
    config = random_ordered_configuration(parent, rng)
    assert len(config) == 2
    assert config.composition.get_el_amt_dict() == {"A": 2}


def test_random_ordered_configuration_preserves_ordered_sites(lattice):
    parent = Structure(
        lattice,
        [{"A": 0.5}] * 2 + ["C"],
        [[0, 0, 0], [0.25, 0, 0], [0.75, 0, 0]],
    )
    rng = np.random.default_rng(0)
    config = random_ordered_configuration(parent, rng)
    assert "C1" in str(config.composition.formula) or config.composition["C"] == 1


def test_random_ordered_configuration_is_reproducible_with_seeded_rng(lattice):
    parent = Structure(
        lattice,
        [{"A": 0.5, "B": 0.5}] * 4,
        [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.75, 0, 0]],
    )
    c1 = random_ordered_configuration(parent, np.random.default_rng(42))
    c2 = random_ordered_configuration(parent, np.random.default_rng(42))
    assert [str(s.species) for s in c1] == [str(s.species) for s in c2]


@pytest.fixture
def oxi_decorated_parent(lattice) -> Structure:
    """A 4-site cation pool disordered between Na+/K+ (same charge -- many
    orderings will be energetically degenerate but not necessarily
    symmetrically equivalent under this low-symmetry lattice), plus a fixed
    anion sublattice for charge balance context."""
    parent = Structure(
        lattice,
        [{"Na+": 0.5, "K+": 0.5}] * 4 + ["Cl-"] * 4,
        [
            [0, 0, 0],
            [0.25, 0, 0],
            [0.5, 0, 0],
            [0.75, 0, 0],
            [0, 0.5, 0],
            [0.25, 0.5, 0],
            [0.5, 0.5, 0],
            [0.75, 0.5, 0],
        ],
    )
    return parent


def test_random_ewald_sampling_returns_sorted_distinct_results(oxi_decorated_parent):
    results = random_ewald_sampling(oxi_decorated_parent, n_attempts=100, n_return=6, seed=0)
    assert 1 <= len(results) <= 6
    energies = [r.energy for r in results]
    assert energies == sorted(energies)
    # Pairwise verification that returned configurations are genuinely
    # symmetrically distinct (not accidentally deduplicated too little).
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not symmetrically_equivalent(results[i].structure, results[j].structure)


def test_random_ewald_sampling_is_reproducible(oxi_decorated_parent):
    kwargs = dict(n_attempts=50, n_return=5, seed=123)
    r1 = random_ewald_sampling(oxi_decorated_parent, **kwargs)
    r2 = random_ewald_sampling(oxi_decorated_parent, **kwargs)
    assert [r.energy for r in r1] == [r.energy for r in r2]


def test_random_ewald_sampling_respects_n_return(oxi_decorated_parent):
    results = random_ewald_sampling(oxi_decorated_parent, n_attempts=100, n_return=2, seed=1)
    assert len(results) <= 2


def test_random_ewald_sampling_preserves_composition(oxi_decorated_parent):
    results = random_ewald_sampling(oxi_decorated_parent, n_attempts=30, n_return=3, seed=2)
    for r in results:
        assert r.structure.composition.get_el_amt_dict() == {"Na": 2.0, "K": 2.0, "Cl": 4.0}


def test_random_ewald_sampling_records_recoverable_ordering_keys(oxi_decorated_parent):
    results = random_ewald_sampling(oxi_decorated_parent, n_attempts=30, n_return=3, seed=2)
    assert all(r.ordering_key for r in results)
    for result in results:
        recovered = structure_from_ordering_key(oxi_decorated_parent, result.ordering_key)
        assert _species_by_frac(recovered) == _species_by_frac(result.structure)


def test_random_ewald_sampling_rejects_bad_arguments(oxi_decorated_parent):
    with pytest.raises(ValueError):
        random_ewald_sampling(oxi_decorated_parent, n_attempts=0, n_return=1)
    with pytest.raises(ValueError):
        random_ewald_sampling(oxi_decorated_parent, n_attempts=10, n_return=0)


def test_random_energy_sampling_uses_custom_scalar_evaluator(oxi_decorated_parent):
    def count_na_sites(structure):
        return sum(1 for site in structure if site.species_string == "Na+")

    results = random_energy_sampling(oxi_decorated_parent, count_na_sites, n_attempts=20, n_return=2, seed=4)

    assert results
    assert all(r.energy == 2.0 for r in results)
    assert all(r.properties == {} for r in results)


def test_random_energy_sampling_preserves_custom_properties(oxi_decorated_parent):
    def evaluator(structure):
        energy = float(len(structure))
        return {"energy": energy, "num_sites": len(structure)}

    results = random_energy_sampling(oxi_decorated_parent, evaluator, n_attempts=20, n_return=2, seed=4)

    assert results
    assert all(r.energy == r.properties["energy"] for r in results)
    assert all(r.properties["num_sites"] == len(r.structure) for r in results)


def test_random_energy_sampling_ignores_future_warnings(oxi_decorated_parent):
    def evaluator(structure):
        warnings.warn("upstream deprecation", FutureWarning)
        return 0.0

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        random_energy_sampling(
            oxi_decorated_parent,
            evaluator,
            n_attempts=2,
            n_return=1,
            seed=4,
            verbose=False,
        )

    assert emitted == []


def test_random_energy_sampling_keeps_non_future_warnings(oxi_decorated_parent):
    def evaluator(structure):
        warnings.warn("important runtime warning", RuntimeWarning)
        return 0.0

    with pytest.warns(RuntimeWarning, match="important runtime warning"):
        random_energy_sampling(
            oxi_decorated_parent,
            evaluator,
            n_attempts=1,
            n_return=1,
            seed=4,
            verbose=False,
        )


def test_random_energy_sampling_parallel_matches_serial(oxi_decorated_parent):
    def evaluator(structure):
        li_or_na_x = sum(
            float(site.frac_coords[0])
            for site in structure
            if site.species_string in {"Na+", "K+"}
        )
        return {"energy": li_or_na_x, "num_sites": len(structure)}

    kwargs = dict(
        n_attempts=20,
        n_return=3,
        seed=7,
        candidate_pool_size=10,
        verbose=False,
    )

    serial = random_energy_sampling(oxi_decorated_parent, evaluator, n_jobs=1, **kwargs)
    parallel = random_energy_sampling(oxi_decorated_parent, evaluator, n_jobs=2, **kwargs)

    assert [r.step for r in parallel] == [r.step for r in serial]
    assert [r.energy for r in parallel] == [r.energy for r in serial]
    assert [r.ordering_key for r in parallel] == [r.ordering_key for r in serial]


def test_random_energy_sampling_prepares_parallel_workers_before_evaluation(oxi_decorated_parent):
    events = []

    class WarmupEvaluator:
        def prepare_worker(self):
            events.append("prepare")

        def __call__(self, structure):
            events.append("evaluate")
            return 0.0

    random_energy_sampling(
        oxi_decorated_parent,
        WarmupEvaluator(),
        n_attempts=2,
        n_return=1,
        seed=4,
        n_jobs=2,
        verbose=False,
    )

    assert events[:2] == ["prepare", "prepare"]
    assert events[2:] == ["evaluate", "evaluate"]


def test_random_energy_sampling_parallel_reports_completed_futures(oxi_decorated_parent, monkeypatch, capsys):
    def reversed_as_completed(futures):
        return reversed(list(futures))

    monkeypatch.setattr("disorderly.sampling.as_completed", reversed_as_completed)

    def evaluator(structure):
        return 0.0

    random_energy_sampling(
        oxi_decorated_parent,
        evaluator,
        n_attempts=2,
        n_return=1,
        seed=4,
        n_jobs=2,
        verbose=True,
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Attempt       1")
    assert lines[1].startswith("Attempt       0")


def test_random_energy_sampling_time_limit_returns_best_so_far(oxi_decorated_parent, monkeypatch):
    calls = 0

    def evaluator(structure):
        nonlocal calls
        calls += 1
        return float(100 - calls)

    times = iter([0.0, 0.5, 2.0])
    monkeypatch.setattr("disorderly.sampling.time.perf_counter", lambda: next(times))

    results = random_energy_sampling(
        oxi_decorated_parent,
        evaluator,
        n_attempts=10,
        n_return=3,
        seed=4,
        time_limit=1.0,
    )

    assert calls == 2
    assert results
    assert all(result.step < 2 for result in results)


def test_random_energy_sampling_requires_energy_key(oxi_decorated_parent):
    with pytest.raises(ValueError, match="energy"):
        random_energy_sampling(
            oxi_decorated_parent,
            lambda structure: {"not_energy": 0.0},
            n_attempts=1,
            n_return=1,
        )


def test_random_energy_sampling_rejects_bad_n_jobs(oxi_decorated_parent):
    with pytest.raises(ValueError, match="n_jobs"):
        random_energy_sampling(
            oxi_decorated_parent,
            lambda structure: 0.0,
            n_attempts=1,
            n_return=1,
            n_jobs=0,
        )


def test_random_ordered_configuration_default_rng_is_shared_seed_zero(lattice, monkeypatch):
    import disorderly.sampling as sampling

    parent = Structure(
        lattice,
        [{"A": 0.5, "B": 0.5}] * 8,
        [[i / 8, 0, 0] for i in range(8)],
    )
    monkeypatch.setattr(sampling, "_DEFAULT_RNG", np.random.default_rng(0))
    reference_rng = np.random.default_rng(0)

    def species(structure):
        # Sites are emitted grouped by species, so compare the species found
        # at each coordinate rather than the site order.
        return sorted((round(float(s.frac_coords[0]), 6), str(s.species)) for s in structure)

    defaults = [species(random_ordered_configuration(parent)) for _ in range(5)]
    expected = [species(random_ordered_configuration(parent, reference_rng)) for _ in range(5)]
    assert defaults == expected
    assert len({tuple(d) for d in defaults}) > 1
