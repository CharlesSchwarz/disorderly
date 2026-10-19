import json

from pymatgen.core import Lattice, Structure

from disorder_mc.cli.run_monte_carlo import main


def test_run_monte_carlo_cli_uses_settings_file_and_temperature_output_dir(tmp_path):
    parent = Structure(
        Lattice.cubic(4.0),
        [{"Na": 0.5, "K": 0.5}, {"Na": 0.5, "K": 0.5}, "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    initial = Structure(
        parent.lattice,
        ["Na", "K", "Cl"],
        parent.frac_coords,
    )
    parent_path = tmp_path / "parent.cif"
    initial_path = tmp_path / "initial.cif"
    parent.to(filename=str(parent_path))
    initial.to(filename=str(initial_path))

    settings_path = tmp_path / "monte_carlo_settings.py"
    settings_path.write_text(
        """
from disorder_mc.settings import MonteCarloPolicy, MoveProposal, default_is_move_allowed

num_steps = 4
num_structures_to_save = 2
seed = 0
record_structure_every = 2

def choose_potential_move(configuration, rng, context):
    return MoveProposal(move_type="swap", site_index=0, target_site_index=1)

def evaluate_props(configuration, context):
    return {"energy": float(len(configuration)), "temperature": context["temperature"]}

def get_policy(disordered_structure, initial_structure, temperature):
    return MonteCarloPolicy(
        choose_potential_move=choose_potential_move,
        is_move_allowed=default_is_move_allowed,
        evaluate_props=evaluate_props,
        context={"temperature": temperature, "parent_sites": len(disordered_structure)},
        detailed_balance=True,
    )
"""
    )

    main([str(parent_path), str(initial_path), "750", str(settings_path)])

    output_dir = tmp_path / "parent_T750K_monte_carlo_results"
    data = json.loads((output_dir / "trajectory.json").read_text())

    assert data["parameters"]["temperature"] == 750.0
    assert data["parameters"]["num_steps"] == 4
    assert len(data["steps"]) == 5
    assert data["steps"][1]["properties"]["temperature"] == 750.0
    assert list((output_dir / "best_structures").glob("best_*.cif"))


def test_run_monte_carlo_cli_accepts_module_level_policy_with_run_context(tmp_path):
    parent = Structure(
        Lattice.cubic(4.0),
        [{"Na": 0.5, "K": 0.5}, {"Na": 0.5, "K": 0.5}, "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    initial = Structure(parent.lattice, ["Na", "K", "Cl"], parent.frac_coords)
    parent_path = tmp_path / "parent.cif"
    initial_path = tmp_path / "initial.cif"
    parent.to(filename=str(parent_path))
    initial.to(filename=str(initial_path))

    settings_path = tmp_path / "monte_carlo_settings.py"
    settings_path.write_text(
        """
from disorder_mc.settings import MonteCarloPolicy, MoveProposal, default_is_move_allowed

num_steps = 3
seed = 0

def choose_potential_move(configuration, rng, context):
    return MoveProposal(move_type="swap", site_index=0, target_site_index=1)

def evaluate_props(configuration, context):
    return {
        "energy": 0.0,
        "parent_sites": len(context["disordered_structure"]),
        "initial_sites": len(context["initial_structure"]),
        "temperature": context["temperature"],
        "label": context["label"],
    }

policy = MonteCarloPolicy(
    choose_potential_move=choose_potential_move,
    is_move_allowed=default_is_move_allowed,
    evaluate_props=evaluate_props,
    context={"label": "mine", "temperature": -1.0},
    detailed_balance=True,
)
"""
    )

    main([str(parent_path), str(initial_path), "500", str(settings_path)])

    data = json.loads((tmp_path / "parent_T500K_monte_carlo_results" / "trajectory.json").read_text())
    props = data["steps"][1]["properties"]
    assert props["parent_sites"] == 3
    assert props["initial_sites"] == 3
    assert props["label"] == "mine"
    assert props["temperature"] == -1.0  # keys the policy sets are not overwritten


def test_run_monte_carlo_cli_prints_progress(tmp_path, capsys):
    parent = Structure(
        Lattice.cubic(4.0),
        [{"Na": 0.5, "K": 0.5}, {"Na": 0.5, "K": 0.5}, "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    initial = Structure(parent.lattice, ["Na", "K", "Cl"], parent.frac_coords)
    parent_path = tmp_path / "parent.cif"
    initial_path = tmp_path / "initial.cif"
    parent.to(filename=str(parent_path))
    initial.to(filename=str(initial_path))

    settings_template = """
from disorder_mc.settings import MonteCarloPolicy, MoveProposal, default_is_move_allowed

num_steps = 8
seed = 0
progress_updates = {progress_updates}

policy = MonteCarloPolicy(
    choose_potential_move=lambda configuration, rng, context: MoveProposal(move_type="swap", site_index=0, target_site_index=1),
    is_move_allowed=default_is_move_allowed,
    evaluate_props=lambda configuration, context: {{"energy": 0.0}},
    detailed_balance=True,
)
"""
    settings_path = tmp_path / "monte_carlo_settings.py"
    settings_path.write_text(settings_template.format(progress_updates=4))
    main([str(parent_path), str(initial_path), "500", str(settings_path)])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
    assert [line.split("]")[0] for line in lines] == ["[25%", "[50%", "[75%"]
    assert lines[0].startswith("[25%] step 2/8: energy 0.000000 eV, acceptance rate 1.000")

    settings_path.write_text(settings_template.format(progress_updates=0))
    main([str(parent_path), str(initial_path), "500", str(settings_path)])
    assert not [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
