"""Settings-file CLI for random sampling with a custom energy evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pymatgen.core import Structure

from disorder_mc.cli._settings_file import load_settings_file, setting
from disorder_mc.io import save_configurations, save_structures
from disorder_mc.sampling import EnergyEvaluator, random_energy_sampling


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample ordered configurations of a disordered structure "
            "using options from a Python settings file."
        )
    )
    parser.add_argument("structure_path", type=Path, help="Path to the disordered parent structure.")
    parser.add_argument("settings_path", type=Path, help="Path to the Python settings file.")
    return parser


def _load_energy_evaluator(settings) -> EnergyEvaluator:
    """Resolve the configured energy evaluator from a settings module."""
    evaluator = getattr(settings, "energy_evaluator", None)
    if evaluator is not None:
        if not callable(evaluator):
            raise TypeError("settings.energy_evaluator must be callable.")
        return evaluator

    factory = getattr(settings, "get_energy_evaluator", None)
    if factory is not None:
        if not callable(factory):
            raise TypeError("settings.get_energy_evaluator must be callable.")
        evaluator = factory()
        if not callable(evaluator):
            raise TypeError("settings.get_energy_evaluator() must return a callable.")
        return evaluator

    raise AttributeError("Settings file must define energy_evaluator or get_energy_evaluator().")


def run_random_sampling(structure_path: str | Path, settings_path: str | Path) -> list:
    """Run random energy sampling using a structure path and settings file path."""
    structure_path = Path(structure_path).expanduser().resolve()
    settings = load_settings_file(settings_path)

    parent = Structure.from_file(structure_path)
    evaluator = _load_energy_evaluator(settings)

    n_attempts = setting(settings, "n_attempts", required=True)
    n_return = setting(settings, "n_return", 10)
    time_limit = setting(settings, "time_limit", None)
    seed = setting(settings, "seed", None)
    occupancy_tol = setting(settings, "occupancy_tol", 1e-2)
    candidate_pool_size = setting(settings, "candidate_pool_size", None)
    symmetry_match_kwargs = setting(settings, "symmetry_match_kwargs", None)
    n_jobs = setting(settings, "n_jobs", 1)
    save_sites_in_json = setting(settings, "save_sites_in_json", False)
    verbose = setting(settings, "verbose", True)

    results = random_energy_sampling(
        parent,
        evaluator,
        n_attempts=n_attempts,
        n_return=n_return,
        seed=seed,
        occupancy_tol=occupancy_tol,
        candidate_pool_size=candidate_pool_size,
        symmetry_match_kwargs=symmetry_match_kwargs,
        time_limit=time_limit,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    configured_output_dir = setting(settings, "output_dir", None)
    if configured_output_dir is None:
        output_dir = Path(settings.__file__).resolve().parent / f"{structure_path.stem}_random_sampling_results"
    else:
        output_dir = Path(configured_output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path(settings.__file__).resolve().parent / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    save_configurations(
        results,
        output_dir / "sampled_configurations.json",
        include_structures=save_sites_in_json,
    )
    save_structures([c.structure for c in results], output_dir / "structures", prefix="ordered")

    print(f"Found {len(results)} symmetrically distinct configurations; saved to {output_dir}")
    for c in results:
        key = f"  key {c.ordering_key}" if c.ordering_key else ""
        print(f"  step {c.step:>7d}  energy {c.energy:.6f} eV{key}")

    return results


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_random_sampling(args.structure_path, args.settings_path)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
