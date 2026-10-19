"""Settings-file CLI for Monte Carlo ordering simulations."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from disorderly.cli._settings_file import load_settings_file, setting
from disorderly.io import save_structures, save_trajectory
from disorderly.monte_carlo import run_monte_carlo
from disorderly.settings import MonteCarloPolicy, default_policy


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with only the essential run identifiers."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a Metropolis Monte Carlo ordering simulation using run controls "
            "and policy behavior from a Python settings file."
        )
    )
    parser.add_argument("disordered_structure", type=Path, help="Path to the disordered parent structure.")
    parser.add_argument("initial_structure", type=Path, help="Path to the ordered starting configuration.")
    parser.add_argument("temperature", type=float, help="Simulation temperature in Kelvin.")
    parser.add_argument("settings_path", type=Path, help="Path to the Python settings file.")
    return parser


def _temperature_label(temperature: float) -> str:
    label = f"{temperature:g}".replace("-", "neg").replace(".", "p")
    return f"T{label}K"


def _default_output_dir(disordered_structure_path: Path, temperature: float) -> Path:
    return disordered_structure_path.parent / (
        f"{disordered_structure_path.stem}_{_temperature_label(temperature)}_monte_carlo_results"
    )


def _settings_output_dir(settings, disordered_structure_path: Path, temperature: float) -> Path:
    configured_output_dir = setting(settings, "output_dir", None)
    if configured_output_dir is None:
        return _default_output_dir(disordered_structure_path, temperature)

    output_dir = Path(configured_output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(settings.__file__).resolve().parent / output_dir
    return output_dir


def _call_get_policy(factory, **kwargs: Any) -> MonteCarloPolicy:
    """Call a user policy factory, passing only arguments it accepts."""
    import inspect

    try:
        signature = inspect.signature(factory)
    except ValueError:
        policy = factory()
    else:
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            policy = factory(**kwargs)
        else:
            accepted = {k: v for k, v in kwargs.items() if k in signature.parameters}
            policy = factory(**accepted)

    if not isinstance(policy, MonteCarloPolicy):
        raise TypeError(f"settings.get_policy(...) must return a MonteCarloPolicy, got {type(policy)!r}.")
    return policy


def _format_seconds(seconds: float) -> str:
    minutes, seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def _progress_printer(num_steps: int, progress_updates: int):
    """Return an ``on_step`` callback printing progress ``progress_updates``
    times per run (e.g. 4 prints at 25%, 50% and 75%; the final summary
    covers 100%), or ``None`` if ``progress_updates`` is 0 or less."""
    if progress_updates <= 0 or num_steps <= 0:
        return None
    report_at = {max(1, round(num_steps * k / progress_updates)): k for k in range(1, progress_updates)}
    start = time.perf_counter()
    n_accepted = 0

    def on_step(step, checkpoint) -> None:
        nonlocal n_accepted
        n_accepted += bool(step.accepted)
        if step.step not in report_at:
            return
        elapsed = time.perf_counter() - start
        remaining = elapsed * (num_steps - step.step) / step.step
        percent = 100 * report_at[step.step] / progress_updates
        print(
            f"[{percent:.0f}%] step {step.step}/{num_steps}: energy {step.energy:.6f} eV, "
            f"acceptance rate {n_accepted / step.step:.3f}, "
            f"elapsed {_format_seconds(elapsed)}, about {_format_seconds(remaining)} left",
            flush=True,
        )

    return on_step


def _load_policy(settings, disordered_structure, initial_structure, temperature: float) -> MonteCarloPolicy:
    """Return the settings file's policy with the run inputs added to its context.

    The policy comes from a module-level ``policy``, else ``get_policy(...)``,
    else :func:`~disorderly.settings.default_policy`. ``disordered_structure``,
    ``initial_structure`` and ``temperature`` are added to a copy of its
    ``context`` (keys the policy already sets are kept), so a module-level
    ``policy`` can use the parent structure given on the command line.
    """
    run_context = {
        "disordered_structure": disordered_structure,
        "initial_structure": initial_structure,
        "temperature": temperature,
    }

    policy = getattr(settings, "policy", None)
    factory = getattr(settings, "get_policy", None)
    if policy is not None:
        if not isinstance(policy, MonteCarloPolicy):
            raise TypeError(f"settings.policy must be a MonteCarloPolicy, got {type(policy)!r}.")
    elif factory is not None:
        if not callable(factory):
            raise TypeError("settings.get_policy must be callable.")
        policy = _call_get_policy(factory, **run_context)
    else:
        policy = default_policy()

    return dataclasses.replace(policy, context={**run_context, **policy.context})


def run_monte_carlo_from_settings(
    disordered_structure_path: str | Path,
    initial_structure_path: str | Path,
    temperature: float,
    settings_path: str | Path,
):
    """Run Monte Carlo using paths and a Python settings file."""
    disordered_structure_path = Path(disordered_structure_path).expanduser().resolve()
    initial_structure_path = Path(initial_structure_path).expanduser().resolve()
    settings = load_settings_file(settings_path)

    disordered_structure = Structure.from_file(disordered_structure_path)
    initial_structure = Structure.from_file(initial_structure_path)
    policy = _load_policy(settings, disordered_structure, initial_structure, temperature)

    num_steps = setting(settings, "num_steps", required=True)
    num_structures_to_save = setting(settings, "num_structures_to_save", 10)
    time_limit = setting(settings, "time_limit", None)
    seed = setting(settings, "seed", None)
    max_move_attempts = setting(settings, "max_move_attempts", 1000)
    disallowed_moves = setting(settings, "disallowed_moves", "reject")
    record_structure_every = setting(settings, "record_structure_every", None)
    move_appliers = setting(settings, "move_appliers", None)
    save_best_structures = setting(settings, "save_best_structures", True)
    progress_updates = setting(settings, "progress_updates", 4)

    trajectory = run_monte_carlo(
        initial_structure,
        policy,
        temperature=temperature,
        num_steps=num_steps,
        num_structures_to_save=num_structures_to_save,
        time_limit=time_limit,
        seed=seed,
        max_move_attempts=max_move_attempts,
        record_structure_every=record_structure_every,
        move_appliers=move_appliers,
        disallowed_moves=disallowed_moves,
        on_step=_progress_printer(num_steps, progress_updates),
    )

    output_dir = _settings_output_dir(settings, disordered_structure_path, temperature)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory(trajectory, output_dir / "trajectory.json")

    best = trajectory.lowest_energy_saved()
    if save_best_structures:
        save_structures([c.structure for c in best], output_dir / "best_structures", prefix="best")

    print(
        f"Ran {len(trajectory.steps) - 1} steps "
        f"(acceptance rate {trajectory.acceptance_rate():.3f}); "
        f"final energy {trajectory.energies[-1]:.6f} eV; "
        f"saved {len(best)} configurations and the trajectory to {output_dir}"
    )
    return trajectory


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_monte_carlo_from_settings(
        args.disordered_structure,
        args.initial_structure,
        args.temperature,
        args.settings_path,
    )


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
