import warnings
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from pymatgen.core import Lattice, Structure

from disorder_mc._warning_filters import should_suppress_logm_warning
from disorder_mc.mace import MACEEnergyEvaluator, mace_policy
from disorder_mc.mlip import _warn_unless_small_logm_error
from disorder_mc.monte_carlo import run_monte_carlo
from disorder_mc.settings import default_choose_potential_move, default_is_move_allowed


def test_mace_policy_uses_default_moves_and_mace_evaluator(monkeypatch):
    def fake_get_energy(self, structure):
        return float(len(structure))

    monkeypatch.setattr(MACEEnergyEvaluator, "get_energy", fake_get_energy)

    policy = mace_policy("fake.model", context={"label": "test"})
    structure = Structure(
        Lattice.cubic(4.0),
        ["Na", "K"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )

    assert policy.choose_potential_move is default_choose_potential_move
    assert policy.is_move_allowed is default_is_move_allowed
    assert policy.context["label"] == "test"
    assert policy.context["energy_model"] == "MACE"
    assert policy.evaluate_props(structure, policy.context) == {"energy": 2.0, "mace_energy": 2.0}


def test_mace_policy_can_drive_monte_carlo(monkeypatch):
    def fake_get_energy(self, structure):
        return float(len(structure))

    monkeypatch.setattr(MACEEnergyEvaluator, "get_energy", fake_get_energy)

    structure = Structure(
        Lattice.cubic(4.0),
        ["Na", "K", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    trajectory = run_monte_carlo(
        structure,
        mace_policy("fake.model"),
        temperature=300,
        num_steps=3,
        seed=0,
    )

    assert trajectory.energies == [3.0, 3.0, 3.0, 3.0]
    assert all(step.properties.get("mace_energy") == 3.0 for step in trajectory.steps)


def test_mace_evaluator_uses_thread_local_calculators(monkeypatch):
    counter = 0

    def fake_build_calculator(self):
        nonlocal counter
        counter += 1
        return {"calculator": counter}

    monkeypatch.setattr(MACEEnergyEvaluator, "_build_calculator", fake_build_calculator)
    evaluator = MACEEnergyEvaluator("fake.model")

    main_thread_calculator = evaluator._get_calculator()
    assert evaluator._get_calculator() is main_thread_calculator

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker_calculator = executor.submit(evaluator._get_calculator).result()

    assert worker_calculator is not main_thread_calculator
    assert counter == 2


def test_mace_evaluator_serializes_first_evaluation_per_thread(monkeypatch):
    evaluator = MACEEnergyEvaluator("fake.model")
    lock = MagicMock()
    lock.__enter__.return_value = None
    lock.__exit__.return_value = None
    evaluator._first_evaluation_lock = lock

    monkeypatch.setattr(MACEEnergyEvaluator, "_calculate_energy", lambda self, structure: 1.0)
    structure = Structure(Lattice.cubic(4.0), ["Li"], [[0, 0, 0]])

    assert evaluator.get_energy(structure) == 1.0
    assert evaluator.get_energy(structure) == 1.0
    assert lock.__enter__.call_count == 1

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(evaluator.get_energy, structure).result() == 1.0

    assert lock.__enter__.call_count == 2


def test_mace_evaluator_serializes_worker_preparation(monkeypatch):
    evaluator = MACEEnergyEvaluator("fake.model")
    lock = MagicMock()
    lock.__enter__.return_value = None
    lock.__exit__.return_value = None
    evaluator._first_evaluation_lock = lock

    monkeypatch.setattr(MACEEnergyEvaluator, "_build_calculator", lambda self: object())

    evaluator.prepare_worker()
    evaluator.prepare_worker()

    assert lock.__enter__.call_count == 1

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(evaluator.prepare_worker).result()

    assert lock.__enter__.call_count == 2


def test_small_scipy_logm_warning_is_suppressed():
    with warnings.catch_warnings(record=True) as source:
        warnings.simplefilter("always")
        warnings.warn(
            "logm result may be inaccurate, approximate err = 5.620164327685667e-13",
            RuntimeWarning,
        )

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        _warn_unless_small_logm_error(source, threshold=1e-6)

    assert emitted == []


def test_logm_warning_filter_uses_threshold():
    assert should_suppress_logm_warning(
        "logm result may be inaccurate, approximate err = 5.5423860547689e-13",
        RuntimeWarning,
        threshold=1e-6,
    )
    assert not should_suppress_logm_warning(
        "logm result may be inaccurate, approximate err = 1.0e-5",
        RuntimeWarning,
        threshold=1e-6,
    )
    assert not should_suppress_logm_warning(
        "logm result may be inaccurate, approximate err = 5.5423860547689e-13",
        UserWarning,
        threshold=1e-6,
    )


def test_large_scipy_logm_warning_is_reemitted():
    with warnings.catch_warnings(record=True) as source:
        warnings.simplefilter("always")
        warnings.warn(
            "logm result may be inaccurate, approximate err = 1.0e-5",
            RuntimeWarning,
        )

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        _warn_unless_small_logm_error(source, threshold=1e-6)

    assert len(emitted) == 1
    assert "approximate err = 1.0e-5" in str(emitted[0].message)


def _ase_relaxation_evaluator(monkeypatch, **kwargs):
    import pytest

    pytest.importorskip("ase")
    from ase.calculators.emt import EMT

    monkeypatch.setattr(MACEEnergyEvaluator, "_build_calculator", lambda self: EMT())
    return MACEEnergyEvaluator("fake.model", relax=True, relax_cell=False, **kwargs)


def _strained_copper() -> Structure:
    structure = Structure(Lattice.cubic(3.6), ["Cu"] * 4, [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    structure.translate_sites([0], [0.03, 0.01, 0.0])
    return structure


def test_mace_evaluator_reports_relaxation_convergence(monkeypatch):
    evaluator = _ase_relaxation_evaluator(monkeypatch, fmax=0.01)
    props = evaluator.evaluate_props(_strained_copper())
    assert props["relax_converged"] is True
    assert props["relax_steps"] > 0
    assert props["energy"] == props["mace_energy"]


def test_mace_evaluator_warns_when_relaxation_hits_step_cap(monkeypatch):
    import pytest

    evaluator = _ase_relaxation_evaluator(monkeypatch, fmax=1e-6, max_steps=1)
    with pytest.warns(RuntimeWarning, match="did not reach fmax"):
        props = evaluator.evaluate_props(_strained_copper())
    assert props["relax_converged"] is False
    assert props["relax_steps"] == 1
