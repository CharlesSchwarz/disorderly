"""Generic energy evaluation with ASE calculators (e.g. machine-learned potentials).

:class:`ASECalculatorEvaluator` holds the parts of potential-backed energy
evaluation that do not depend on the specific model: conversion to ASE
``Atoms``, optional relaxation, per-thread calculators for threaded
sampling, and Monte-Carlo-compatible property dicts. Model-specific modules
(:mod:`disorder_mc.mace`, :mod:`disorder_mc.chgnet`) only supply the
calculator.

ASE and any model package are imported lazily, so the base ``disorder_mc``
package stays importable without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Callable, ClassVar
import warnings

from pymatgen.core import Structure

from disorder_mc._warning_filters import should_suppress_logm_warning, warning_filter_scope
from disorder_mc.settings import (
    MonteCarloPolicy,
    default_choose_potential_move,
    default_is_move_allowed,
)

__all__ = ["ASECalculatorEvaluator", "mlip_policy"]


def missing_dependency_error(exc: ModuleNotFoundError, install_hint: str) -> ModuleNotFoundError:
    """Return a clearer error for a missing optional dependency."""
    return ModuleNotFoundError(f"Missing optional dependency {exc.name!r}; {install_hint}.")


def _warn_unless_small_logm_error(recorded_warnings: list[warnings.WarningMessage], threshold: float) -> None:
    """Re-emit warnings, suppressing only tiny SciPy logm accuracy warnings."""
    for warning in recorded_warnings:
        if should_suppress_logm_warning(warning.message, warning.category, threshold):
            continue
        warnings.warn(warning.message, warning.category, stacklevel=3)


@dataclass(kw_only=True)
class ASECalculatorEvaluator:
    """Evaluate total energies of pymatgen structures with an ASE calculator.

    Each thread builds its own calculator on first use by calling
    ``calculator_factory`` (or a subclass's ``_build_calculator``), so one
    evaluator can be shared by threaded sampling workers.

    Args:
        calculator_factory: Zero-argument callable returning a fresh ASE
            calculator. Subclasses may leave this unset and override
            ``_build_calculator`` instead.
        label: Short model name, used in warnings and in the default
            ``energy_property``.
        energy_property: Key under which ``evaluate_props`` also stores the
            energy. Defaults to ``"<label lowercased>_energy"``.
        relax: If true, run an ASE LBFGS relaxation before reading the
            potential energy. If false, perform a single-point evaluation.
        relax_cell: If true and ``relax`` is true, relax both positions and
            cell via ``FrechetCellFilter``. If false, relax positions only.
        fmax: Force convergence threshold passed to ASE ``LBFGS.run``.
        max_steps: Maximum optimizer steps for relaxation.
        logfile: ASE optimizer logfile. Defaults to ``None`` for quiet
            relaxation.
        logm_warning_tol: Suppress SciPy matrix-logarithm accuracy warnings
            from ASE relaxation when the reported approximate error is at or
            below this threshold. Set to ``0`` to show all such warnings.
    """

    calculator_factory: Callable[[], Any] | None = None
    label: str = "MLIP"
    energy_property: str | None = None
    relax: bool = False
    relax_cell: bool = True
    fmax: float = 0.05
    max_steps: int = 1000
    logfile: str | None = None
    logm_warning_tol: float = 1e-6

    _install_hint: ClassVar[str] = "install ase in this environment"

    _thread_local: threading.local = field(default_factory=threading.local, init=False, repr=False)
    _first_evaluation_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.energy_property is None:
            self.energy_property = f"{self.label.lower()}_energy"

    def _build_calculator(self):
        if self.calculator_factory is None:
            raise TypeError(
                f"{type(self).__name__} needs a calculator_factory, or a subclass must override _build_calculator()."
            )
        return self.calculator_factory()

    def _get_calculator(self):
        calculator = getattr(self._thread_local, "calculator", None)
        if calculator is None:
            calculator = self._build_calculator()
            self._thread_local.calculator = calculator
        return calculator

    def prepare_worker(self) -> None:
        """Initialize the calculator for the current worker thread."""
        if getattr(self._thread_local, "calculator", None) is not None:
            return
        with self._first_evaluation_lock:
            self._get_calculator()

    def _calculate_energy(self, structure: Structure) -> float:
        """Return the calculator's total energy in eV.

        ``structure`` is converted to a fresh ASE ``Atoms`` object, so the
        input pymatgen structure is not mutated.
        """
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
        except ModuleNotFoundError as exc:
            raise missing_dependency_error(exc, self._install_hint) from exc

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self._get_calculator()

        if self.relax:
            try:
                from ase.filters import FrechetCellFilter
                from ase.optimize import LBFGS
            except ModuleNotFoundError as exc:
                raise missing_dependency_error(exc, self._install_hint) from exc
            target = FrechetCellFilter(atoms) if self.relax_cell else atoms
            opt = LBFGS(target, logfile=self.logfile)
            with warning_filter_scope(logm_warning_tol=self.logm_warning_tol):
                converged = bool(opt.run(fmax=self.fmax, steps=self.max_steps))
            self._thread_local.last_relaxation = {"relax_converged": converged, "relax_steps": int(opt.nsteps)}
            if not converged:
                warnings.warn(
                    f"{self.label} relaxation did not reach fmax={self.fmax} eV/A within {self.max_steps} steps.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return float(atoms.get_potential_energy())

    def get_energy(self, structure: Structure) -> float:
        """Return the total energy in eV."""
        if getattr(self._thread_local, "warmed_up", False):
            return self._calculate_energy(structure)

        with self._first_evaluation_lock:
            energy = self._calculate_energy(structure)
        self._thread_local.warmed_up = True
        return energy

    def evaluate_props(self, configuration: Structure, context: dict[str, Any] | None = None) -> dict[str, float]:
        """Return Monte-Carlo-compatible properties for ``configuration``.

        The result holds the energy under both ``"energy"`` and
        ``energy_property``. With ``relax=True`` it also contains
        ``relax_converged`` (whether the relaxation reached ``fmax`` within
        ``max_steps``) and ``relax_steps``, so unconverged energies can be
        found afterwards.
        """
        energy = self.get_energy(configuration)
        props = {"energy": energy, self.energy_property: energy}
        if self.relax:
            props.update(getattr(self._thread_local, "last_relaxation", None) or {})
        return props

    def __call__(self, structure: Structure) -> float:
        """Evaluate ``structure`` as a plain sampling energy callable."""
        return self.get_energy(structure)


def mlip_policy(evaluator: ASECalculatorEvaluator, context: dict[str, Any] | None = None) -> MonteCarloPolicy:
    """Create a swap-based Monte Carlo policy scored by ``evaluator``.

    The move-selection and validation behavior matches
    :func:`disorder_mc.settings.default_policy`; only the energy evaluator is
    replaced. ``context["energy_model"]`` defaults to ``evaluator.label``.
    """
    policy_context = dict(context or {})
    policy_context.setdefault("energy_model", evaluator.label)
    policy_context.setdefault("relax", evaluator.relax)
    return MonteCarloPolicy(
        choose_potential_move=default_choose_potential_move,
        is_move_allowed=default_is_move_allowed,
        evaluate_props=evaluator.evaluate_props,
        context=policy_context,
        detailed_balance=True,
    )
