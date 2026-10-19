"""The Monte Carlo ordering engine.

Contains the generic move primitives (:func:`move_ion`, :func:`swap_ion`),
candidate-generation (:func:`propose_next_configuration`), and the Metropolis
simulation loop (:func:`run_monte_carlo`). All problem-specific behavior is
supplied via a :class:`disorderly.settings.MonteCarloPolicy` -- this module
does not hard-code system-specific ordering rules.

May depend on :mod:`disorderly.structures`, :mod:`disorderly.symmetry`,
and :mod:`disorderly.energy`; depends on :mod:`disorderly.settings` for
the policy interface (not the reverse, to avoid a circular import).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Literal
import warnings

import numpy as np
from pymatgen.core import Structure

from disorderly.results import MonteCarloCheckpoint, MonteCarloStep, MonteCarloTrajectory, SavedConfiguration
from disorderly.settings import DetailedBalanceWarning, MonteCarloPolicy, MoveProposal

__all__ = [
    "KB_EV_PER_K",
    "move_ion",
    "swap_ion",
    "propose_next_configuration",
    "run_monte_carlo",
    "resume_monte_carlo",
]

#: Boltzmann constant in eV/K (CODATA 2018), used to convert temperature
#: (K) and energy (eV) into the dimensionless Metropolis exponent.
KB_EV_PER_K = 8.617333262e-5


def move_ion(
    configuration: Structure,
    site_index: int,
    target_coords,
    coords_are_cartesian: bool = False,
    min_distance: float = 1e-2,
) -> Structure:
    """Relocate the species at ``site_index`` to ``target_coords``.

    ``configuration`` is not mutated; a new structure is returned, with the
    moved site kept at index ``site_index`` (via remove-then-reinsert) so
    that other site indices are unaffected.

    Args:
        configuration: Structure to move a species within.
        site_index: Index of the site whose species should be relocated.
        target_coords: Destination coordinates (fractional by default).
        coords_are_cartesian: Whether ``target_coords`` are Cartesian.
        min_distance: Minimum allowed distance (Angstrom) between
            ``target_coords`` and any other existing site; used to detect
            (and reject) moves onto an already-occupied site.

    Returns:
        A new structure with the species relocated.

    Raises:
        IndexError: If ``site_index`` is out of range.
        ValueError: If ``target_coords`` coincide with an existing site
            (other than ``site_index`` itself).
    """
    n_sites = len(configuration)
    if not 0 <= site_index < n_sites:
        raise IndexError(f"site_index {site_index} out of range for a configuration with {n_sites} sites.")

    lattice = configuration.lattice
    target_frac = (
        lattice.get_fractional_coords(target_coords) if coords_are_cartesian else np.asarray(target_coords, dtype=float)
    )
    for i, other in enumerate(configuration):
        if i == site_index:
            continue
        distance, _ = lattice.get_distance_and_image(other.frac_coords, target_frac)
        if distance < min_distance:
            raise ValueError(
                f"Target coordinates are within {distance:.4g} A of existing site {i} "
                f"(minimum allowed distance {min_distance} A); the target is already occupied."
            )

    site = configuration[site_index]
    new_struct = configuration.copy()
    new_struct.remove_sites([site_index])
    new_struct.insert(
        site_index,
        site.species,
        target_frac,
        coords_are_cartesian=False,
        properties=dict(site.properties),
        label=site.label,
    )
    return new_struct


def swap_ion(configuration: Structure, site_index_1: int, site_index_2: int) -> Structure:
    """Swap the species (and site properties) occupying two sites.

    ``configuration`` is not mutated; a new structure is returned. Site
    coordinates are unchanged; only the species (and properties) at the two
    indices are exchanged.

    Args:
        configuration: Structure to swap species within.
        site_index_1, site_index_2: Indices of the two sites to swap.

    Returns:
        A new structure with the two sites' species exchanged.

    Raises:
        IndexError: If either index is out of range.
        ValueError: If the two indices are the same.
    """
    n_sites = len(configuration)
    for idx in (site_index_1, site_index_2):
        if not 0 <= idx < n_sites:
            raise IndexError(f"site index {idx} out of range for a configuration with {n_sites} sites.")
    if site_index_1 == site_index_2:
        raise ValueError("swap_ion requires two distinct site indices.")

    site_1, site_2 = configuration[site_index_1], configuration[site_index_2]
    new_struct = configuration.copy()
    new_struct.replace(site_index_1, site_2.species, properties=dict(site_2.properties))
    new_struct.replace(site_index_2, site_1.species, properties=dict(site_1.properties))
    return new_struct


def _apply_swap(configuration: Structure, move: MoveProposal) -> Structure:
    if move.target_site_index is None:
        raise ValueError('A "swap" move requires `target_site_index` to be set.')
    return swap_ion(configuration, move.site_index, move.target_site_index)


def _apply_move(configuration: Structure, move: MoveProposal) -> Structure:
    if move.target_frac_coords is None:
        raise ValueError('A "move" move requires `target_frac_coords` to be set.')
    return move_ion(configuration, move.site_index, move.target_frac_coords, coords_are_cartesian=False)


#: Built-in move-type -> applier mapping used by :func:`propose_next_configuration`.
#: Extend by passing additional entries via that function's/``run_monte_carlo``'s
#: ``move_appliers`` argument (new entries are merged on top of these
#: defaults), rather than modifying this module, to support custom move
#: types without changing the engine.
DEFAULT_MOVE_APPLIERS: dict[str, Callable[[Structure, MoveProposal], Structure]] = {
    "swap": _apply_swap,
    "move": _apply_move,
}


def propose_next_configuration(
    configuration: Structure,
    policy: MonteCarloPolicy,
    rng: np.random.Generator,
    max_attempts: int = 1000,
    move_appliers: dict[str, Callable[[Structure, MoveProposal], Structure]] | None = None,
) -> tuple[Structure, MoveProposal, dict[str, float]]:
    """Generate a candidate next configuration using ``policy``.

    Repeatedly asks ``policy.choose_potential_move`` for a candidate move,
    applies it (dispatching on ``move.move_type`` via ``move_appliers``),
    and checks ``policy.is_move_allowed``, until an allowed move is found or
    ``max_attempts`` is exceeded. The accepted candidate's properties are
    evaluated via ``policy.evaluate_props``.

    Redrawing disallowed proposals divides each move's proposal probability
    by the fraction of allowed proposals from ``configuration``. This is how
    :func:`run_monte_carlo` proposes moves only with
    ``disallowed_moves="retry"``; by default it instead rejects a disallowed
    proposal as a Monte Carlo step, which preserves detailed balance.

    Args:
        configuration: Current configuration to propose a move from.
        policy: Policy supplying move selection, validation, and property
            evaluation.
        rng: Random-number generator (drives ``policy.choose_potential_move``
            reproducibly).
        max_attempts: Maximum number of proposal attempts before giving up.
        move_appliers: Extra ``{move_type: applier}`` entries, merged on top
            of :data:`DEFAULT_MOVE_APPLIERS` (``{"swap": ..., "move": ...}``),
            to support custom move types.

    Returns:
        ``(candidate_configuration, move, properties)``, where ``properties``
        is ``policy.evaluate_props``'s return value and is guaranteed to
        contain an ``"energy"`` key.

    Raises:
        RuntimeError: If no allowed move was found within ``max_attempts``
            attempts.
        ValueError: If a proposed move's ``move_type`` has no registered
            applier, or if ``evaluate_props`` did not return an ``"energy"``
            key.
    """
    appliers = {**DEFAULT_MOVE_APPLIERS, **(move_appliers or {})}

    for _ in range(max_attempts):
        candidate, move = _propose_once(configuration, policy, rng, appliers)
        if candidate is None:
            continue
        return candidate, move, _evaluate(candidate, policy)

    raise RuntimeError(f"No allowed Monte Carlo move found after {max_attempts} attempts.")


def _propose_once(
    configuration: Structure,
    policy: MonteCarloPolicy,
    rng: np.random.Generator,
    appliers: dict[str, Callable[[Structure, MoveProposal], Structure]],
) -> tuple[Structure | None, MoveProposal | None]:
    """Draw one proposal. Returns ``(candidate, move)``, with ``candidate``
    ``None`` if the policy proposed no move (``move`` is then ``None`` too)
    or the move is not allowed."""
    move = policy.choose_potential_move(configuration, rng, policy.context)
    if move is None:
        return None, None
    applier = appliers.get(move.move_type)
    if applier is None:
        raise ValueError(
            f"No move applier registered for move_type={move.move_type!r}. "
            f"Known move types: {sorted(appliers)}. Pass a custom `move_appliers` "
            "mapping to support this move type."
        )
    candidate = applier(configuration, move)
    if not policy.is_move_allowed(configuration, candidate, move, policy.context):
        return None, move
    return candidate, move


def _evaluate(candidate: Structure, policy: MonteCarloPolicy) -> dict[str, Any]:
    properties = policy.evaluate_props(candidate, policy.context)
    if "energy" not in properties:
        raise ValueError(
            'policy.evaluate_props(...) must return a dict containing an "energy" '
            f"key (used for the Metropolis acceptance criterion); got keys {sorted(properties)}."
        )
    return properties


def _metropolis_accept(
    current_energy: float,
    candidate_energy: float,
    temperature: float,
    rng: np.random.Generator,
    log_proposal_ratio: float = 0.0,
) -> bool:
    """Metropolis--Hastings acceptance: accept with probability
    ``min(1, exp(-dE / kT + log_proposal_ratio))``. At ``temperature <= 0``
    (greedy descent) the proposal ratio is ignored and only ``dE <= 0`` is
    accepted."""
    delta = candidate_energy - current_energy
    if temperature <= 0:
        return delta <= 0
    log_acceptance = -delta / (KB_EV_PER_K * temperature) + log_proposal_ratio
    if log_acceptance >= 0:
        return True
    return bool(rng.random() < np.exp(log_acceptance))


def _record_saved_configuration(
    saved: list[SavedConfiguration],
    step: int,
    energy: float,
    structure: Structure,
    properties: dict[str, Any],
    num_structures_to_save: int,
) -> None:
    if num_structures_to_save <= 0:
        return
    saved.append(SavedConfiguration(step=step, energy=energy, structure=structure.copy(), properties=dict(properties)))
    _trim_saved_configurations(saved, num_structures_to_save)


def _trim_saved_configurations(saved: list[SavedConfiguration], num_structures_to_save: int) -> None:
    saved.sort(key=lambda c: c.energy)
    del saved[max(num_structures_to_save, 0):]


StepCallback = Callable[[MonteCarloStep, MonteCarloCheckpoint], None]


def run_monte_carlo(
    initial_configuration: Structure,
    policy: MonteCarloPolicy,
    temperature: float,
    num_steps: int,
    num_structures_to_save: int = 10,
    time_limit: float | None = None,
    seed: int | None = None,
    max_move_attempts: int = 1000,
    record_structure_every: int | None = None,
    move_appliers: dict[str, Callable[[Structure, MoveProposal], Structure]] | None = None,
    on_step: StepCallback | None = None,
    disallowed_moves: Literal["reject", "retry"] = "reject",
) -> MonteCarloTrajectory:
    """Run a Metropolis Monte Carlo ordering simulation.

    At each step, a candidate configuration is proposed via ``policy``,
    evaluated, and accepted or rejected according to the
    Metropolis--Hastings criterion ``min(1, exp(-dE / kT + r))`` using the
    candidate's ``"energy"`` and the move's ``log_proposal_ratio`` ``r``.
    Every evaluated candidate (accepted or not) is considered for the
    ``num_structures_to_save`` lowest-energy saved configurations, so that
    promising but-rejected configurations are not lost.

    At ``temperature > 0`` a :class:`~disorderly.settings.DetailedBalanceWarning`
    is emitted if ``policy.detailed_balance`` is not ``True``, and (with
    ``disallowed_moves="retry"``) the first time a proposal is redrawn.

    Args:
        initial_configuration: Starting ordered configuration. Not mutated.
        policy: Move selection / validation / property evaluation policy
            (see :mod:`disorderly.settings`).
        temperature: Simulation temperature in Kelvin. ``0`` (or negative)
            performs greedy (zero-temperature) descent: candidates are
            accepted only if they do not raise the energy.
        num_steps: Maximum number of Monte Carlo steps to attempt.
        num_structures_to_save: Number of lowest-energy configurations to
            retain (across all evaluated candidates, not just accepted
            ones). Set to 0 to disable.
        time_limit: Optional wall-clock time limit in seconds; the
            simulation stops early (without error) once exceeded.
        seed: Random seed for reproducibility. Recorded on the returned
            trajectory.
        max_move_attempts: Passed to :func:`propose_next_configuration` at each step.
        record_structure_every: If given, the full structure is recorded on
            the trajectory step every ``record_structure_every`` steps (plus
            always on the initial step 0), in addition to the
            always-recorded scalar step data, to bound memory use on long
            runs. ``None`` (default) never records per-step structures on
            the trajectory itself (use ``num_structures_to_save`` to retain
            the best structures regardless).
        move_appliers: Passed to :func:`propose_next_configuration`.
        on_step: Optional ``on_step(step, checkpoint)`` called after every
            Monte Carlo step (not the initial step 0) with that step's
            record and a :class:`~disorderly.results.MonteCarloCheckpoint`
            of the state after it. Use it to stream steps to disk and to
            save checkpoints (e.g. with :func:`disorderly.io.save_checkpoint`)
            so a long run can be continued with :func:`resume_monte_carlo`.
        disallowed_moves: What to do when the policy proposes no move or
            ``policy.is_move_allowed`` rejects its proposal.
            ``"reject"`` (default) counts it as a rejected step (recorded
            with ``allowed=False``, its candidate not evaluated), which
            preserves detailed balance: it samples the Boltzmann
            distribution restricted to allowed configurations.
            ``"retry"`` redraws the proposal, up to ``max_move_attempts``
            times, so every step evaluates a candidate. That reweights each
            configuration by its fraction of allowed proposals and breaks
            detailed balance unless the fraction is the same in every
            configuration.

    Returns:
        The completed :class:`~disorderly.results.MonteCarloTrajectory`.
    """
    _check_disallowed_moves(disallowed_moves)
    rng = np.random.default_rng(seed)
    current = initial_configuration.copy()
    current_props = policy.evaluate_props(current, policy.context)
    if "energy" not in current_props:
        raise ValueError('policy.evaluate_props(...) must return a dict containing an "energy" key.')

    trajectory = MonteCarloTrajectory(
        random_seed=seed,
        parameters={
            "temperature": temperature,
            "num_steps": num_steps,
            "num_structures_to_save": num_structures_to_save,
            "time_limit": time_limit,
            "max_move_attempts": max_move_attempts,
            "record_structure_every": record_structure_every,
            "disallowed_moves": disallowed_moves,
            "detailed_balance": policy.detailed_balance,
        },
    )
    saved: list[SavedConfiguration] = []
    _record_saved_configuration(saved, 0, current_props["energy"], current, current_props, num_structures_to_save)
    return _run_metropolis(
        trajectory, current, current_props, 0, saved, rng, policy, temperature, num_steps,
        num_structures_to_save, time_limit, max_move_attempts, record_structure_every, move_appliers, on_step,
        disallowed_moves,
    )


def resume_monte_carlo(
    checkpoint: MonteCarloCheckpoint,
    policy: MonteCarloPolicy,
    temperature: float,
    num_steps: int,
    num_structures_to_save: int = 10,
    time_limit: float | None = None,
    max_move_attempts: int = 1000,
    record_structure_every: int | None = None,
    move_appliers: dict[str, Callable[[Structure, MoveProposal], Structure]] | None = None,
    on_step: StepCallback | None = None,
    disallowed_moves: Literal["reject", "retry"] = "reject",
) -> MonteCarloTrajectory:
    """Continue a Monte Carlo run from a checkpoint.

    The run continues from ``checkpoint.configuration`` without
    re-evaluating it, with its random-number generator restored from
    ``checkpoint.rng_state`` and its saved lowest-energy configurations
    carried over. With the same policy and arguments, the steps after the
    checkpoint are identical to those of the uninterrupted run, provided
    ``policy.evaluate_props`` is deterministic and the policy draws all its
    randomness from the ``rng`` it is given.

    Args:
        checkpoint: State after the last completed step, as passed to an
            ``on_step`` callback of :func:`run_monte_carlo` or a previous
            :func:`resume_monte_carlo` (and possibly saved/loaded with
            :func:`disorderly.io.save_checkpoint`/``load_checkpoint``).
        num_steps: Total number of steps of the whole run, *including* the
            ``checkpoint.step`` steps already done: the run stops after
            step ``num_steps``. Steps are numbered continuing from
            ``checkpoint.step``.
        policy, temperature, num_structures_to_save, time_limit,
        max_move_attempts, record_structure_every, move_appliers, on_step,
        disallowed_moves:
            As for :func:`run_monte_carlo`. ``time_limit`` counts from this
            call, not from the start of the original run.

    Returns:
        A :class:`~disorderly.results.MonteCarloTrajectory` of this
        segment of the run. Its first step (``steps[0]``, ``move=None``)
        records the checkpointed state at step ``checkpoint.step``, like
        step 0 of a fresh run; ``saved_configurations`` covers the whole run.
    """
    if "energy" not in checkpoint.properties:
        raise ValueError('checkpoint.properties must contain an "energy" key.')
    _check_disallowed_moves(disallowed_moves)
    rng = _generator_from_state(checkpoint.rng_state)
    trajectory = MonteCarloTrajectory(
        random_seed=checkpoint.random_seed,
        parameters={
            "temperature": temperature,
            "num_steps": num_steps,
            "num_structures_to_save": num_structures_to_save,
            "time_limit": time_limit,
            "max_move_attempts": max_move_attempts,
            "record_structure_every": record_structure_every,
            "disallowed_moves": disallowed_moves,
            "detailed_balance": policy.detailed_balance,
            "resumed_from_step": checkpoint.step,
        },
    )
    saved = list(checkpoint.saved_configurations)
    _trim_saved_configurations(saved, num_structures_to_save)
    return _run_metropolis(
        trajectory, checkpoint.configuration.copy(), dict(checkpoint.properties), checkpoint.step, saved, rng,
        policy, temperature, num_steps, num_structures_to_save, time_limit, max_move_attempts,
        record_structure_every, move_appliers, on_step, disallowed_moves,
    )


def _check_disallowed_moves(disallowed_moves: str) -> None:
    if disallowed_moves not in ("reject", "retry"):
        raise ValueError(f'disallowed_moves must be "reject" or "retry", got {disallowed_moves!r}.')


def _warn_if_policy_breaks_detailed_balance(policy: MonteCarloPolicy, temperature: float) -> None:
    if temperature <= 0 or policy.detailed_balance is True:
        return
    if policy.detailed_balance is False:
        message = (
            "The Monte Carlo policy declares detailed_balance=False, so this run does not sample the "
            "Boltzmann distribution: its energies are exact and it can still find low-energy "
            "configurations, but averages over the trajectory are biased."
        )
    else:
        message = (
            "The Monte Carlo policy does not declare whether its proposals satisfy detailed balance "
            "(MonteCarloPolicy.detailed_balance is None). If a move and its reverse are not proposed "
            "with equal probability (or corrected by MoveProposal.log_proposal_ratio), averages over "
            "the trajectory are biased. Check the policy with "
            "disorderly.diagnostics.check_proposal_symmetry and set detailed_balance=True or False."
        )
    warnings.warn(message, DetailedBalanceWarning, stacklevel=4)


def _generator_from_state(state: dict[str, Any]) -> np.random.Generator:
    """Rebuild a ``numpy.random.Generator`` from ``bit_generator.state``."""
    bit_generator_cls = getattr(np.random, state["bit_generator"], None)
    if not (isinstance(bit_generator_cls, type) and issubclass(bit_generator_cls, np.random.BitGenerator)):
        raise ValueError(f"Unknown bit generator {state['bit_generator']!r} in checkpoint rng_state.")
    bit_generator = bit_generator_cls()
    bit_generator.state = state
    return np.random.Generator(bit_generator)


def _run_metropolis(
    trajectory: MonteCarloTrajectory,
    current: Structure,
    current_props: dict[str, Any],
    start_step: int,
    saved: list[SavedConfiguration],
    rng: np.random.Generator,
    policy: MonteCarloPolicy,
    temperature: float,
    num_steps: int,
    num_structures_to_save: int,
    time_limit: float | None,
    max_move_attempts: int,
    record_structure_every: int | None,
    move_appliers: dict[str, Callable[[Structure, MoveProposal], Structure]] | None,
    on_step: StepCallback | None,
    disallowed_moves: str,
) -> MonteCarloTrajectory:
    """The Metropolis loop shared by :func:`run_monte_carlo` and
    :func:`resume_monte_carlo`, running steps ``start_step + 1`` to ``num_steps``."""
    _warn_if_policy_breaks_detailed_balance(policy, temperature)
    appliers = {**DEFAULT_MOVE_APPLIERS, **(move_appliers or {})}
    warned_retry = False
    current_energy = current_props["energy"]
    trajectory.append(
        MonteCarloStep(
            step=start_step,
            energy=current_energy,
            accepted=True,
            temperature=temperature,
            move=None,
            properties=current_props,
            structure=current.copy() if record_structure_every else None,
        )
    )

    start_time = time.perf_counter()
    for step in range(start_step + 1, num_steps + 1):
        if time_limit is not None and (time.perf_counter() - start_time) > time_limit:
            break

        candidate, move = _propose_once(current, policy, rng, appliers)
        if candidate is None and disallowed_moves == "retry":
            for _ in range(max_move_attempts - 1):
                candidate, move = _propose_once(current, policy, rng, appliers)
                if candidate is not None:
                    break
            else:
                raise RuntimeError(f"No allowed Monte Carlo move found after {max_move_attempts} attempts.")
            if temperature > 0 and not warned_retry:
                warned_retry = True
                warnings.warn(
                    f"Step {step}: a disallowed Monte Carlo proposal was redrawn (disallowed_moves=\"retry\"). "
                    "This biases sampling, breaking detailed balance, unless the fraction of allowed "
                    'proposals is the same in every configuration; use disallowed_moves="reject" to '
                    "count disallowed proposals as rejected steps instead.",
                    DetailedBalanceWarning,
                    stacklevel=3,
                )

        if candidate is None:
            accepted, props = False, {}
        else:
            props = _evaluate(candidate, policy)
            accepted = _metropolis_accept(current_energy, props["energy"], temperature, rng, move.log_proposal_ratio)
            if accepted:
                current, current_energy, current_props = candidate, props["energy"], props

        record_structure = record_structure_every is not None and step % record_structure_every == 0
        step_record = MonteCarloStep(
            step=step,
            energy=current_energy,
            accepted=accepted,
            temperature=temperature,
            move=dataclasses.asdict(move) if move is not None else None,
            properties=props,
            structure=current.copy() if record_structure else None,
            allowed=candidate is not None,
        )
        trajectory.append(step_record)
        if candidate is not None:
            _record_saved_configuration(saved, step, props["energy"], candidate, props, num_structures_to_save)
        if on_step is not None:
            on_step(
                step_record,
                MonteCarloCheckpoint(
                    step=step,
                    configuration=current.copy(),
                    properties=dict(current_props),
                    rng_state=rng.bit_generator.state,
                    saved_configurations=list(saved),
                    random_seed=trajectory.random_seed,
                ),
            )

    trajectory.saved_configurations = saved
    return trajectory
