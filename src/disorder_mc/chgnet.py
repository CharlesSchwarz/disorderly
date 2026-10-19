"""Optional CHGNet-backed energy evaluation and Monte Carlo policies.

CHGNet/ASE imports are lazy, so the base ``disorder_mc`` package remains
importable without them. Relaxation, threading, and property handling are
shared with other potentials via
:class:`disorder_mc.mlip.ASECalculatorEvaluator`.

CHGNet does not use the oxidation states on the input structure; it infers
charge information itself (via predicted magnetic moments).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from disorder_mc.mlip import ASECalculatorEvaluator, missing_dependency_error, mlip_policy
from disorder_mc.settings import MonteCarloPolicy

__all__ = ["CHGNetEnergyEvaluator", "chgnet_policy", "get_energy_evaluator"]

DEFAULT_CHGNET_MODEL = "0.3.0"


@dataclass
class CHGNetEnergyEvaluator(ASECalculatorEvaluator):
    """Evaluate total CHGNet energies for pymatgen structures.

    Args:
        model_path: Path to a fine-tuned CHGNet model file, loaded with
            ``CHGNet.from_file``. If ``None``, the pretrained model named by
            ``model_name`` is used.
        model_name: Pretrained model passed to ``CHGNet.load(model_name=...)``
            when ``model_path`` is ``None`` (e.g. ``"0.3.0"`` or
            ``"r2scan"``). Ignored when ``model_path`` is given.
        device: Device passed to ``CHGNetCalculator(use_device=...)``
            (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``). ``None`` lets CHGNet
            pick a device itself.
        calculator_kwargs: Extra keyword arguments for ``CHGNetCalculator``.

    Relaxation options (``relax``, ``relax_cell``, ``fmax``, ``max_steps``,
    ``logfile``, ``logm_warning_tol``) are keyword-only and documented on
    :class:`~disorder_mc.mlip.ASECalculatorEvaluator`. Properties are stored
    under ``"energy"`` and ``"chgnet_energy"``; ``CHGNetCalculator`` reports
    total (not per-atom) energies in eV.
    """

    model_path: str | Path | None = None
    model_name: str = DEFAULT_CHGNET_MODEL
    device: str | None = "cpu"
    calculator_kwargs: dict[str, Any] = field(default_factory=dict)
    label: str = field(default="CHGNet", kw_only=True)

    _install_hint: ClassVar[str] = "install disorder-mc with the 'chgnet' extra or install ase and chgnet"

    def _build_calculator(self):
        try:
            from chgnet.model import CHGNet
            from chgnet.model.dynamics import CHGNetCalculator
        except ModuleNotFoundError as exc:
            raise missing_dependency_error(exc, self._install_hint) from exc
        if self.model_path is None:
            model = CHGNet.load(model_name=self.model_name, verbose=False)
        else:
            model = CHGNet.from_file(str(self.model_path))
        return CHGNetCalculator(model=model, use_device=self.device, **self.calculator_kwargs)


def chgnet_policy(
    model_path: str | Path | None = None,
    *,
    model_name: str = DEFAULT_CHGNET_MODEL,
    device: str | None = "cpu",
    relax: bool = False,
    relax_cell: bool = True,
    fmax: float = 0.05,
    max_steps: int = 1000,
    calculator_kwargs: dict[str, Any] | None = None,
    logfile: str | None = None,
    logm_warning_tol: float = 1e-6,
    context: dict[str, Any] | None = None,
) -> MonteCarloPolicy:
    """Create a swap-based Monte Carlo policy scored by CHGNet energy.

    The move-selection and validation behavior matches
    :func:`disorder_mc.settings.default_policy`; only the energy evaluator is
    replaced.
    """
    evaluator = get_energy_evaluator(
        model_path,
        model_name=model_name,
        device=device,
        relax=relax,
        relax_cell=relax_cell,
        fmax=fmax,
        max_steps=max_steps,
        calculator_kwargs=calculator_kwargs,
        logfile=logfile,
        logm_warning_tol=logm_warning_tol,
    )
    policy_context = dict(context or {})
    policy_context.setdefault("chgnet_model", str(model_path) if model_path is not None else model_name)
    policy_context.setdefault("chgnet_relax", relax)
    return mlip_policy(evaluator, policy_context)


def get_energy_evaluator(
    model_path: str | Path | None = None,
    *,
    model_name: str = DEFAULT_CHGNET_MODEL,
    device: str | None = "cpu",
    relax: bool = False,
    relax_cell: bool = True,
    fmax: float = 0.05,
    max_steps: int = 1000,
    calculator_kwargs: dict[str, Any] | None = None,
    logfile: str | None = None,
    logm_warning_tol: float = 1e-6,
) -> CHGNetEnergyEvaluator:
    """Factory returning a sampling-compatible CHGNet energy evaluator."""
    return CHGNetEnergyEvaluator(
        model_path=model_path,
        model_name=model_name,
        device=device,
        relax=relax,
        relax_cell=relax_cell,
        fmax=fmax,
        max_steps=max_steps,
        calculator_kwargs=dict(calculator_kwargs or {}),
        logfile=logfile,
        logm_warning_tol=logm_warning_tol,
    )
