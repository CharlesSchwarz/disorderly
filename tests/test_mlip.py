import pytest
from pymatgen.core import Lattice, Structure

from disorder_mc.mlip import ASECalculatorEvaluator, mlip_policy
from disorder_mc.monte_carlo import run_monte_carlo
from disorder_mc.settings import default_choose_potential_move, default_is_move_allowed


def _strained_copper() -> Structure:
    structure = Structure(Lattice.cubic(3.6), ["Cu"] * 4, [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    structure.translate_sites([0], [0.03, 0.01, 0.0])
    return structure


def _emt_evaluator(**kwargs) -> ASECalculatorEvaluator:
    pytest.importorskip("ase")
    from ase.calculators.emt import EMT

    return ASECalculatorEvaluator(calculator_factory=EMT, label="EMT", **kwargs)


def test_single_point_matches_direct_ase_calculation():
    from pymatgen.io.ase import AseAtomsAdaptor

    evaluator = _emt_evaluator()
    structure = _strained_copper()

    from ase.calculators.emt import EMT

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = EMT()
    assert evaluator(structure) == pytest.approx(atoms.get_potential_energy())


def test_relaxation_lowers_energy_and_does_not_mutate_input():
    structure = _strained_copper()
    original = structure.copy()

    unrelaxed = _emt_evaluator()(structure)
    props = _emt_evaluator(relax=True, relax_cell=False, fmax=0.01).evaluate_props(structure)

    assert props["energy"] < unrelaxed
    assert props["relax_converged"] is True
    assert structure == original


def test_energy_property_defaults_from_label_and_can_be_overridden():
    evaluator = ASECalculatorEvaluator(calculator_factory=object, label="MyModel")
    assert evaluator.energy_property == "mymodel_energy"
    assert ASECalculatorEvaluator(calculator_factory=object).energy_property == "mlip_energy"
    assert ASECalculatorEvaluator(calculator_factory=object, energy_property="e_ml").energy_property == "e_ml"


def test_evaluate_props_reports_energy_under_both_keys():
    props = _emt_evaluator().evaluate_props(_strained_copper())
    assert set(props) == {"energy", "emt_energy"}
    assert props["energy"] == props["emt_energy"]


def test_missing_calculator_factory_raises():
    evaluator = ASECalculatorEvaluator()
    with pytest.raises(TypeError, match="calculator_factory"):
        evaluator.prepare_worker()


def test_fields_are_keyword_only():
    with pytest.raises(TypeError):
        ASECalculatorEvaluator(object)


def test_mlip_policy_uses_default_moves_and_evaluator():
    evaluator = ASECalculatorEvaluator(calculator_factory=object, label="Fake")
    policy = mlip_policy(evaluator, context={"label": "test", "relax": "kept"})

    assert policy.choose_potential_move is default_choose_potential_move
    assert policy.is_move_allowed is default_is_move_allowed
    assert policy.evaluate_props == evaluator.evaluate_props
    assert policy.context == {"label": "test", "energy_model": "Fake", "relax": "kept"}


def test_mlip_policy_drives_monte_carlo_with_ase_calculator():
    pytest.importorskip("ase")
    from ase.calculators.emt import EMT

    structure = Structure(
        Lattice.cubic(3.6),
        ["Cu", "Cu", "Au", "Au"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    evaluator = ASECalculatorEvaluator(calculator_factory=EMT, label="EMT")
    trajectory = run_monte_carlo(structure, mlip_policy(evaluator), temperature=300, num_steps=3, seed=0)

    assert len(trajectory.steps) == 4
    assert all("emt_energy" in step.properties for step in trajectory.steps)
    assert all(step.properties["emt_energy"] == step.energy for step in trajectory.steps if step.accepted)
