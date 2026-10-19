import json

from pymatgen.core import Lattice, Structure

from disorderly.cli.run_random_sampling import main


def test_run_random_sampling_cli_uses_settings_file_path(tmp_path):
    parent = Structure(
        Lattice.orthorhombic(4.0, 5.0, 6.0),
        [{"Na": 0.5, "K": 0.5}] * 4 + ["Cl"] * 4,
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
    parent_path = tmp_path / "parent.cif"
    parent.to(filename=str(parent_path))

    settings_path = tmp_path / "random_sampling_settings.py"
    settings_path.write_text(
        """
n_attempts = 8
n_return = 2
time_limit = None
seed = 0
occupancy_tol = 1e-2
candidate_pool_size = 6
n_jobs = 2
output_dir = "out"

def energy_evaluator(structure):
    return {"energy": float(len(structure)) + 0.5, "source": "settings-file"}
"""
    )

    main([str(parent_path), str(settings_path)])

    output_dir = tmp_path / "out"
    data = json.loads((output_dir / "sampled_configurations.json").read_text())

    assert 1 <= len(data) <= 2
    assert all("ordering_key" in item for item in data)
    assert all("structure" not in item for item in data)
    assert all(item["properties"]["source"] == "settings-file" for item in data)
    assert list((output_dir / "structures").glob("ordered_*.cif"))
