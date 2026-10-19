import sys

import pytest
from pymatgen.core import Lattice, Structure

from disorderly import io as dio
from disorderly.monte_carlo import run_monte_carlo
from disorderly.results import SavedConfiguration
from disorderly.sampling import random_ewald_sampling
from disorderly.settings import MonteCarloPolicy, default_policy

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


def test_save_structures_sequential_naming(tmp_path, na_k_cl):
    paths = dio.save_structures([na_k_cl, na_k_cl], tmp_path / "out", prefix="cfg", fmt="cif")
    assert [p.name for p in paths] == ["cfg_0.cif", "cfg_1.cif"]
    assert all(p.exists() for p in paths)


def test_save_load_trajectory_roundtrip(tmp_path, na_k_cl):
    trajectory = run_monte_carlo(
        na_k_cl, default_policy(), temperature=800, num_steps=15, seed=5, record_structure_every=5,
        num_structures_to_save=3,
    )
    path = dio.save_trajectory(trajectory, tmp_path / "traj.json")
    loaded = dio.load_trajectory(path)

    assert loaded.energies == trajectory.energies
    assert loaded.accepted == trajectory.accepted
    assert loaded.random_seed == trajectory.random_seed
    assert loaded.parameters == trajectory.parameters
    assert len(loaded.saved_configurations) == len(trajectory.saved_configurations)

    recorded = [s for s in trajectory.steps if s.structure is not None]
    loaded_recorded = [s for s in loaded.steps if s.structure is not None]
    assert len(recorded) == len(loaded_recorded)
    for orig, rt in zip(recorded, loaded_recorded):
        assert orig.structure.composition.get_el_amt_dict() == rt.structure.composition.get_el_amt_dict()


def test_save_load_configurations_roundtrip(tmp_path, na_k_cl):
    configs = [
        SavedConfiguration(
            step=0,
            energy=-1.0,
            structure=na_k_cl,
            properties={"energy": -1.0},
            ordering_key="ok1_A",
        ),
        SavedConfiguration(
            step=1,
            energy=-2.0,
            structure=na_k_cl,
            properties={"energy": -2.0},
            ordering_key="ok1_B",
        ),
    ]
    path = dio.save_configurations(configs, tmp_path / "configs.json", include_structures=True)
    loaded = dio.load_configurations(path)
    assert [c.energy for c in loaded] == [c.energy for c in configs]
    assert [c.step for c in loaded] == [c.step for c in configs]
    assert [c.ordering_key for c in loaded] == [c.ordering_key for c in configs]


def test_save_configurations_defaults_to_ordering_key_only(tmp_path, na_k_cl):
    configs = [
        SavedConfiguration(
            step=0,
            energy=-1.0,
            structure=na_k_cl,
            properties={"energy": -1.0},
            ordering_key="ok1_A",
        )
    ]

    path = dio.save_configurations(configs, tmp_path / "configs.json")
    data = dio.load_json(path)

    assert data[0]["ordering_key"] == "ok1_A"
    assert "structure" not in data[0]


def test_load_configurations_accepts_legacy_ordering_hash_key(tmp_path, na_k_cl):
    path = tmp_path / "legacy_configs.json"
    dio.save_json(
        [
            {
                "step": 0,
                "energy": -1.0,
                "structure": na_k_cl.as_dict(),
                "properties": {},
                "ordering_hash": "oh1_A",
            }
        ],
        path,
    )

    loaded = dio.load_configurations(path)

    assert loaded[0].ordering_key == "oh1_A"


def test_random_ewald_sampling_results_roundtrip(tmp_path):
    lattice = Lattice.cubic(5.64)
    parent = Structure(
        lattice, [{"Na+": 0.5, "K+": 0.5}] * 4 + ["Cl-"] * 4, ROCK_SALT_COORDS
    )
    results = random_ewald_sampling(parent, n_attempts=10, n_return=2, seed=0)
    path = dio.save_configurations(results, tmp_path / "sampling.json")
    loaded = dio.load_configurations(path, parent=parent)
    assert [c.energy for c in loaded] == [c.energy for c in results]
    assert [c.ordering_key for c in loaded] == [c.ordering_key for c in results]


def test_load_compact_configurations_requires_parent(tmp_path):
    path = tmp_path / "sampling.json"
    dio.save_json(
        [{"step": 0, "energy": -1.0, "properties": {}, "ordering_key": "ok1_A"}],
        path,
    )

    with pytest.raises(ValueError, match="pass parent"):
        dio.load_configurations(path)


def test_save_json_load_json_roundtrip(tmp_path):
    import numpy as np

    obj = {"a": 1, "b": [1, 2, 3], "c": float(np.float64(1.5))}
    path = dio.save_json(obj, tmp_path / "data.json")
    loaded = dio.load_json(path)
    assert loaded == obj


def test_load_policy_imports_and_calls_factory(tmp_path, monkeypatch):
    module_source = '''
from disorderly.settings import default_policy

def get_policy(context=None):
    return default_policy(context=context)
'''
    (tmp_path / "my_mc_settings.py").write_text(module_source)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("my_mc_settings", None)

    policy = dio.load_policy("my_mc_settings", context={"foo": "bar"})
    assert isinstance(policy, MonteCarloPolicy)
    assert policy.context == {"foo": "bar"}


def test_load_energy_evaluator_imports_and_calls_factory(tmp_path, monkeypatch, na_k_cl):
    module_source = '''
def get_energy_evaluator(offset=0.0):
    def evaluate(structure):
        return {"energy": len(structure) + offset, "offset": offset}
    return evaluate
'''
    (tmp_path / "my_energy.py").write_text(module_source)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("my_energy", None)

    evaluator = dio.load_energy_evaluator("my_energy", offset=1.5)

    assert evaluator(na_k_cl) == {"energy": 9.5, "offset": 1.5}


def test_load_energy_evaluator_requires_callable_return(tmp_path, monkeypatch):
    (tmp_path / "bad_energy.py").write_text("def get_energy_evaluator():\n    return 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("bad_energy", None)

    with pytest.raises(TypeError):
        dio.load_energy_evaluator("bad_energy")


def test_load_policy_requires_policy_return_type(tmp_path, monkeypatch):
    (tmp_path / "bad_settings.py").write_text("def get_policy():\n    return 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("bad_settings", None)
    with pytest.raises(TypeError):
        dio.load_policy("bad_settings")


def test_load_policy_missing_factory_raises(tmp_path, monkeypatch):
    (tmp_path / "empty_settings.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("empty_settings", None)
    with pytest.raises(AttributeError):
        dio.load_policy("empty_settings")


def test_save_load_checkpoint_roundtrip(tmp_path, na_k_cl):
    checkpoints = []
    run_monte_carlo(
        na_k_cl, default_policy(), temperature=1000, num_steps=4, seed=5, num_structures_to_save=2,
        on_step=lambda step, checkpoint: checkpoints.append(checkpoint),
    )
    checkpoint = checkpoints[-1]
    path = dio.save_checkpoint(checkpoint, tmp_path / "checkpoint.json")
    assert not (tmp_path / "checkpoint.json.tmp").exists()
    loaded = dio.load_checkpoint(path)
    assert loaded.step == 4
    assert loaded.configuration == checkpoint.configuration
    assert loaded.energy == checkpoint.energy
    assert loaded.rng_state == checkpoint.rng_state
    assert loaded.random_seed == 5
    assert [(c.step, c.energy) for c in loaded.saved_configurations] == [
        (c.step, c.energy) for c in checkpoint.saved_configurations
    ]


def test_save_checkpoint_overwrites_previous(tmp_path, na_k_cl):
    checkpoints = []
    run_monte_carlo(
        na_k_cl, default_policy(), temperature=1000, num_steps=2, seed=5,
        on_step=lambda step, checkpoint: checkpoints.append(checkpoint),
    )
    path = tmp_path / "checkpoint.json"
    dio.save_checkpoint(checkpoints[0], path)
    dio.save_checkpoint(checkpoints[1], path)
    assert dio.load_checkpoint(path).step == 2
