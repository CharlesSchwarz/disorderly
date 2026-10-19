"""Optional MACE-backed energy evaluation and Monte Carlo policies.

This module keeps MACE/ASE imports lazy so the base ``disorderly`` package
remains importable without optional machine-learning potential dependencies.
Relaxation, threading, and property handling are shared with other
potentials via :class:`disorderly.mlip.ASECalculatorEvaluator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from disorderly.mlip import ASECalculatorEvaluator, missing_dependency_error, mlip_policy
from disorderly.settings import MonteCarloPolicy

__all__ = ["MACEEnergyEvaluator", "get_energy_evaluator", "mace_policy"]


@dataclass
class MACEEnergyEvaluator(ASECalculatorEvaluator):
    """Evaluate total MACE energies for pymatgen structures.

    Args:
        model_path: Path to a MACE model file, or any value accepted by
            ``mace.calculators.MACECalculator(model_paths=...)``.
        device: Device passed to ``MACECalculator`` (e.g. ``"cpu"`` or
            ``"cuda"``).
        calculator_kwargs: Extra keyword arguments for ``MACECalculator``.

    Relaxation options (``relax``, ``relax_cell``, ``fmax``, ``max_steps``,
    ``logfile``, ``logm_warning_tol``) are keyword-only and documented on
    :class:`~disorderly.mlip.ASECalculatorEvaluator`. Properties are stored
    under ``"energy"`` and ``"mace_energy"``.
    """

    model_path: str | Path
    device: str = "cpu"
    calculator_kwargs: dict[str, Any] = field(default_factory=dict)
    label: str = field(default="MACE", kw_only=True)

    _install_hint: ClassVar[str] = "install disorderly with the 'mace' extra or install ase and mace-torch"

    def _build_calculator(self):
        try:
            from mace.calculators import MACECalculator
        except ModuleNotFoundError as exc:
            raise missing_dependency_error(exc, self._install_hint) from exc
        return MACECalculator(
            model_paths=str(self.model_path),
            device=self.device,
            **self.calculator_kwargs,
        )


def mace_policy(
    model_path: str | Path,
    *,
    device: str = "cpu",
    relax: bool = False,
    relax_cell: bool = True,
    fmax: float = 0.05,
    max_steps: int = 1000,
    calculator_kwargs: dict[str, Any] | None = None,
    logfile: str | None = None,
    logm_warning_tol: float = 1e-6,
    context: dict[str, Any] | None = None,
) -> MonteCarloPolicy:
    """Create a swap-based Monte Carlo policy scored by MACE energy.

    The move-selection and validation behavior matches
    :func:`disorderly.settings.default_policy`; only the energy evaluator is
    replaced.
    """
    evaluator = get_energy_evaluator(
        model_path,
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
    policy_context.setdefault("mace_model_path", str(model_path))
    policy_context.setdefault("mace_relax", relax)
    return mlip_policy(evaluator, policy_context)


def get_energy_evaluator(
    model_path: str | Path,
    *,
    device: str = "cpu",
    relax: bool = False,
    relax_cell: bool = True,
    fmax: float = 0.05,
    max_steps: int = 1000,
    calculator_kwargs: dict[str, Any] | None = None,
    logfile: str | None = None,
    logm_warning_tol: float = 1e-6,
) -> MACEEnergyEvaluator:
    """Factory returning a sampling-compatible MACE energy evaluator."""
    return MACEEnergyEvaluator(
        model_path=model_path,
        device=device,
        relax=relax,
        relax_cell=relax_cell,
        fmax=fmax,
        max_steps=max_steps,
        calculator_kwargs=dict(calculator_kwargs or {}),
        logfile=logfile,
        logm_warning_tol=logm_warning_tol,
    )
