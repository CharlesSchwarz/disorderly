"""CHGNet integration tests.

Most tests replace the ``chgnet`` package with a stub so they run without
CHGNet installed; ``test_real_chgnet_energy`` exercises the real model when
it is available.
"""

import sys
import types

import pytest
from pymatgen.core import Lattice, Structure

from disorderly.chgnet import CHGNetEnergyEvaluator, chgnet_policy, get_energy_evaluator
from disorderly.io import load_energy_evaluator
from disorderly.monte_carlo import run_monte_carlo
from disorderly.settings import default_choose_potential_move, default_is_move_allowed


@pytest.fixture
def fake_chgnet(monkeypatch):
    """Install a stub ``chgnet`` package recording how models are loaded."""
    calls = []

    class FakeCHGNet:
        @classmethod
        def load(cls, **kwargs):
            calls.append(("load", kwargs))
            return "pretrained-model"

        @classmethod
        def from_file(cls, path, **kwargs):
            calls.append(("from_file", path))
            return "file-model"

    class FakeCHGNetCalculator:
        def __init__(self, model=None, **kwargs):
            calls.append(("calculator", model, kwargs))
            self.model = model
            self.kwargs = kwargs

    chgnet = types.ModuleType("chgnet")
    model = types.ModuleType("chgnet.model")
    dynamics = types.ModuleType("chgnet.model.dynamics")
    model.CHGNet = FakeCHGNet
    dynamics.CHGNetCalculator = FakeCHGNetCalculator
    chgnet.model = model
    model.dynamics = dynamics
    monkeypatch.setitem(sys.modules, "chgnet", chgnet)
    monkeypatch.setitem(sys.modules, "chgnet.model", model)
    monkeypatch.setitem(sys.modules, "chgnet.model.dynamics", dynamics)
    return calls


def _nacl_like() -> Structure:
    return Structure(
        Lattice.cubic(4.0),
        ["Na", "K", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )


def test_default_loads_pretrained_model(fake_chgnet):
    calculator = CHGNetEnergyEvaluator()._build_calculator()

    assert fake_chgnet[0] == ("load", {"model_name": "0.3.0", "verbose": False})
    assert calculator.model == "pretrained-model"
    assert calculator.kwargs == {"use_device": "cpu"}


def test_model_name_and_calculator_options_are_forwarded(fake_chgnet):
    evaluator = CHGNetEnergyEvaluator(
        model_name="r2scan", device="cuda", calculator_kwargs={"on_isolated_atoms": "ignore"}
    )
    calculator = evaluator._build_calculator()

    assert fake_chgnet[0] == ("load", {"model_name": "r2scan", "verbose": False})
    assert calculator.kwargs == {"use_device": "cuda", "on_isolated_atoms": "ignore"}


def test_model_path_loads_from_file(fake_chgnet, tmp_path):
    path = tmp_path / "finetuned.pth.tar"
    calculator = CHGNetEnergyEvaluator(model_path=path)._build_calculator()

    assert fake_chgnet[0] == ("from_file", str(path))
    assert calculator.model == "file-model"


def test_missing_chgnet_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "chgnet", None)
    with pytest.raises(ModuleNotFoundError, match="'chgnet' extra"):
        CHGNetEnergyEvaluator()._build_calculator()


def test_evaluator_properties_and_relax_options():
    evaluator = get_energy_evaluator(relax=True, relax_cell=False, fmax=0.02, max_steps=10)

    assert evaluator.energy_property == "chgnet_energy"
    assert evaluator.label == "CHGNet"
    assert (evaluator.relax, evaluator.relax_cell, evaluator.fmax, evaluator.max_steps) == (True, False, 0.02, 10)


def test_chgnet_policy_context_and_moves(monkeypatch):
    monkeypatch.setattr(CHGNetEnergyEvaluator, "get_energy", lambda self, structure: float(len(structure)))

    policy = chgnet_policy(model_name="r2scan", relax=True, context={"label": "test"})

    assert policy.choose_potential_move is default_choose_potential_move
    assert policy.is_move_allowed is default_is_move_allowed
    assert policy.context["label"] == "test"
    assert policy.context["energy_model"] == "CHGNet"
    assert policy.context["chgnet_model"] == "r2scan"
    assert policy.context["chgnet_relax"] is True
    assert chgnet_policy(model_path="my.model").context["chgnet_model"] == "my.model"


def test_chgnet_policy_can_drive_monte_carlo(monkeypatch):
    monkeypatch.setattr(CHGNetEnergyEvaluator, "get_energy", lambda self, structure: float(len(structure)))

    trajectory = run_monte_carlo(_nacl_like(), chgnet_policy(), temperature=300, num_steps=3, seed=0)

    assert trajectory.energies == [3.0, 3.0, 3.0, 3.0]
    assert all(step.properties.get("chgnet_energy") == 3.0 for step in trajectory.steps)


def test_module_works_with_energy_evaluator_loader():
    evaluator = load_energy_evaluator("disorderly.chgnet", device="cuda")
    assert isinstance(evaluator, CHGNetEnergyEvaluator)
    assert evaluator.device == "cuda"


def test_real_chgnet_energy():
    pytest.importorskip("chgnet")
    pytest.importorskip("ase")

    structure = Structure(Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    original = structure.copy()
    props = CHGNetEnergyEvaluator().evaluate_props(structure)

    assert props["energy"] == props["chgnet_energy"]
    assert props["energy"] < 0
    assert structure == original
