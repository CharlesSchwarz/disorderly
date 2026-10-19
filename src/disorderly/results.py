"""Data structures for representing Monte Carlo trajectories and results.

Deliberately simple (dataclasses) rather than a bespoke abstraction.
Depends only on :mod:`pymatgen.core.Structure`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pymatgen.core import Structure

__all__ = ["MonteCarloStep", "SavedConfiguration", "MonteCarloTrajectory", "MonteCarloCheckpoint"]


@dataclass
class MonteCarloStep:
    """A single step of a Monte Carlo trajectory.

    Attributes:
        step: Step number (0 for the initial configuration).
        energy: Energy of the *current* configuration after this step (i.e.
            the accepted energy, which equals the previous step's energy if
            the proposed move was rejected).
        accepted: Whether the proposed move was accepted.
        temperature: Temperature used for the Metropolis criterion at this
            step.
        move: Metadata describing the proposed move (e.g. move type, sites
            and species involved), as produced by the move-selection policy.
            ``None`` for the initial step, which has no proposed move, and
            for a step at which the policy proposed no move.
        properties: Evaluated properties of the *proposed* configuration at
            this step (e.g. ``{"ewald_energy": ...}``), as produced by the
            property-evaluation policy. Empty for a disallowed step, whose
            candidate is not evaluated.
        structure: The current (post-acceptance-decision) configuration, if
            the engine was configured to record structures at this step.
            ``None`` when structures are only recorded periodically to save
            memory (see ``MonteCarloTrajectory``/the Monte Carlo engine's
            ``record_structure_every`` parameter).
        allowed: ``False`` if the step was rejected without evaluation
            because the policy proposed no move or ``is_move_allowed``
            rejected it (the engine's default ``disallowed_moves="reject"``
            mode); such a step has ``accepted=False``.
    """

    step: int
    energy: float
    accepted: bool
    temperature: float
    move: dict[str, Any] | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    structure: Structure | None = None
    allowed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "energy": self.energy,
            "accepted": self.accepted,
            "temperature": self.temperature,
            "move": self.move,
            "properties": self.properties,
            "structure": self.structure.as_dict() if self.structure is not None else None,
            "allowed": self.allowed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MonteCarloStep":
        structure = Structure.from_dict(d["structure"]) if d.get("structure") is not None else None
        return cls(
            step=d["step"],
            energy=d["energy"],
            accepted=d["accepted"],
            temperature=d["temperature"],
            move=d.get("move"),
            properties=dict(d.get("properties") or {}),
            structure=structure,
            allowed=d.get("allowed", True),
        )


@dataclass
class SavedConfiguration:
    """A configuration explicitly retained by a Monte Carlo run, e.g. one of
    the ``num_structures_to_save`` lowest-energy configurations visited."""

    step: int
    energy: float
    structure: Structure
    properties: dict[str, Any] = field(default_factory=dict)
    ordering_key: str | None = None

    def as_dict(self, include_structure: bool = True) -> dict[str, Any]:
        data = {
            "step": self.step,
            "energy": self.energy,
            "properties": self.properties,
            "ordering_key": self.ordering_key,
        }
        if include_structure:
            data["structure"] = self.structure.as_dict()
        return data

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SavedConfiguration":
        return cls(
            step=d["step"],
            energy=d["energy"],
            structure=Structure.from_dict(d["structure"]),
            properties=dict(d.get("properties") or {}),
            ordering_key=d.get("ordering_key", d.get("ordering_hash")),
        )


@dataclass
class MonteCarloTrajectory:
    """The full record of a Monte Carlo ordering simulation.

    Attributes:
        steps: The per-step trajectory record, in step order.
        saved_configurations: Explicitly retained configurations (e.g. the
            lowest-energy structures seen), independent of whether the full
            structure was recorded at every step of ``steps``.
        random_seed: The seed (or seed sequence entropy) used to initialize
            the simulation's random-number generator, for reproducibility.
        parameters: Simulation parameters (temperature, num_steps,
            time_limit, move policy name, etc.).
        metadata: Free-form additional metadata (e.g. software versions,
            timestamps, notes).
    """

    steps: list[MonteCarloStep] = field(default_factory=list)
    saved_configurations: list[SavedConfiguration] = field(default_factory=list)
    random_seed: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, step: MonteCarloStep) -> None:
        self.steps.append(step)

    @property
    def energies(self) -> list[float]:
        return [s.energy for s in self.steps]

    @property
    def accepted(self) -> list[bool]:
        return [s.accepted for s in self.steps]

    def acceptance_rate(self, window: int | None = None) -> float:
        """Fraction of accepted moves, over the last ``window`` steps (all
        steps, excluding the initial step, if ``window`` is ``None``).
        Disallowed steps count as rejected moves."""
        # Step 0 is the initial configuration, not a proposed move.
        moves = [s for s in self.steps if s.move is not None or not s.allowed]
        if window is not None:
            moves = moves[-window:]
        if not moves:
            return 0.0
        return sum(1 for s in moves if s.accepted) / len(moves)

    def lowest_energy_saved(self, n: int | None = None) -> list[SavedConfiguration]:
        """``saved_configurations`` sorted by ascending energy, optionally
        truncated to the ``n`` lowest."""
        ordered = sorted(self.saved_configurations, key=lambda c: c.energy)
        return ordered if n is None else ordered[:n]

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.as_dict() for s in self.steps],
            "saved_configurations": [c.as_dict() for c in self.saved_configurations],
            "random_seed": self.random_seed,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MonteCarloTrajectory":
        return cls(
            steps=[MonteCarloStep.from_dict(s) for s in d.get("steps", [])],
            saved_configurations=[
                SavedConfiguration.from_dict(c) for c in d.get("saved_configurations", [])
            ],
            random_seed=d.get("random_seed"),
            parameters=dict(d.get("parameters") or {}),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class MonteCarloCheckpoint:
    """Everything needed to continue a Monte Carlo run exactly where it
    stopped (see :func:`disorderly.monte_carlo.resume_monte_carlo`).

    Attributes:
        step: The last completed step.
        configuration: The current configuration after ``step``.
        properties: Evaluated properties of ``configuration``; its
            ``"energy"`` is the current Metropolis energy, so resuming does
            not re-evaluate the configuration.
        rng_state: ``numpy.random.Generator.bit_generator.state`` after
            ``step``, so the resumed run draws the same random numbers an
            uninterrupted run would have.
        saved_configurations: The lowest-energy configurations retained so
            far, so the resumed run keeps ranking against them.
        random_seed: The seed the run was originally started with.
    """

    step: int
    configuration: Structure
    properties: dict[str, Any]
    rng_state: dict[str, Any]
    saved_configurations: list[SavedConfiguration] = field(default_factory=list)
    random_seed: int | None = None

    @property
    def energy(self) -> float:
        return self.properties["energy"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "configuration": self.configuration.as_dict(),
            "properties": self.properties,
            "rng_state": self.rng_state,
            "saved_configurations": [c.as_dict() for c in self.saved_configurations],
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MonteCarloCheckpoint":
        return cls(
            step=d["step"],
            configuration=Structure.from_dict(d["configuration"]),
            properties=dict(d["properties"]),
            rng_state=d["rng_state"],
            saved_configurations=[
                SavedConfiguration.from_dict(c) for c in d.get("saved_configurations", [])
            ],
            random_seed=d.get("random_seed"),
        )
