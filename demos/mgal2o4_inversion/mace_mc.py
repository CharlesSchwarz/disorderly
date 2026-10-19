"""MACE Monte Carlo for MgAl2O4 cation inversion: one run in one folder.

:func:`run_temperature` runs Metropolis Monte Carlo over Mg/Al arrangements on
all 48 cation sites of a 112-atom (16 formula unit) cell at one temperature,
scoring every candidate with a relaxed (positions + cell) MACE energy. Swaps
are restricted to cation sites with :func:`disorder_mc.sublattice_swap_policy`,
so oxygen never moves; any Mg/Al pair may exchange, so the inversion x can
change.

Each run lives in its own folder, made by ``setup_sweep.py`` with a
``run_mc.py`` that calls :func:`run_temperature`. Everything the run writes
goes into that folder:

- ``metadata.json`` -- the run's settings. A rerun with different settings is
  refused rather than mixed into the same trajectory.
- ``energy_cache.jsonl`` -- the relaxed energy of every configuration
  evaluated so far. Relaxations are deterministic, so revisited
  configurations cost nothing and a restart loses no relaxation.
- ``steps.jsonl`` -- one line per Monte Carlo step.
- ``checkpoint.json`` -- a :class:`disorder_mc.MonteCarloCheckpoint` written
  after every step. Running again continues with
  :func:`disorder_mc.resume_monte_carlo` exactly as an uninterrupted run
  would, so a job killed at its wall-time limit loses at most the one
  relaxation in progress.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from disorder_mc import (  # noqa: E402
    load_checkpoint,
    resume_monte_carlo,
    run_monte_carlo,
    save_checkpoint,
    sublattice_swap_policy,
)

#: Settings that must match for a run folder to be continued.
FIXED_SETTINGS = ("temperature_K", "initial", "seed", "model", "fmax", "max_relax_steps")


def write_lines_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(tmp, path)


def append_lines(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def run_temperature(
    run_dir: str | Path,
    *,
    temperature_K: float,
    initial: str,
    seed: int,
    model: str | Path,
    device: str = "cpu",
    fmax: float = 0.02,
    max_relax_steps: int = 1000,
    steps: int = 100_000,
    threads: int | None = None,
    log_every: int = 20,
) -> None:
    """Run (or continue) the Monte Carlo run stored in ``run_dir``.

    Args:
        run_dir: Folder for this run's outputs; created if needed.
        temperature_K: Monte Carlo temperature.
        initial: ``"normal"`` (x = 0) or ``"random"`` (x ~ 2/3) starting arrangement.
        seed: Seeds the Monte Carlo moves and the random starting arrangement.
        model: Path to the MACE model file.
        device: Torch device for MACE, e.g. ``"cpu"`` or ``"cuda"``.
        fmax: Relaxation force threshold (eV/A).
        max_relax_steps: Optimizer step cap per relaxation.
        steps: Total Monte Carlo steps for the run, including steps already done.
        threads: Torch threads; ``None`` uses ``SLURM_CPUS_PER_TASK`` if set, else all cores.
        log_every: Steps between progress lines.
    """
    import torch

    from disorder_mc.mace import MACEEnergyEvaluator

    if initial not in ("normal", "random"):
        raise ValueError(f"initial must be 'normal' or 'random', got {initial!r}.")
    if threads is None:
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    torch.set_num_threads(threads)

    parent, tet, octa = common.build_cell()
    cation_sites = tet + octa
    n_tet = len(tet)
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "temperature_K": float(temperature_K),
        "initial": initial,
        "seed": seed,
        "model": Path(model).name,
        "fmax": fmax,
        "max_relax_steps": max_relax_steps,
        "n_atoms": len(parent),
        "n_tetrahedral": n_tet,
        "n_octahedral": len(octa),
        "reference_cif": common.REFERENCE_CIF.name,
    }
    if (out / "metadata.json").exists():
        previous = json.loads((out / "metadata.json").read_text())
        for key in FIXED_SETTINGS:
            if previous.get(key) != metadata[key]:
                raise SystemExit(
                    f"{out} was started with {key}={previous.get(key)!r}, not {metadata[key]!r}; "
                    "refusing to mix runs. Use a new folder for different settings."
                )
    else:
        common.write_json_atomic(out / "metadata.json", metadata)

    cache_path = out / "energy_cache.jsonl"
    cache = {row["key"]: row["props"] for row in common.read_jsonl(cache_path)}
    checkpoint_path = out / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    # A job killed mid-write can leave a partial last line; drop it, and any
    # step written after the last checkpoint, so appends start on a clean line.
    if cache_path.exists():
        write_lines_atomic(cache_path, [{"key": k, "props": v} for k, v in cache.items()])
    steps_path = out / "steps.jsonl"
    steps_done = checkpoint.step if checkpoint else 0
    write_lines_atomic(steps_path, common.read_jsonl(steps_path)[:steps_done])
    print(
        f"T={temperature_K:.0f} K, {initial} start, seed {seed}: "
        f"{'resuming at step ' + str(steps_done) if checkpoint else 'starting'} with {len(cache)} cached energies",
        flush=True,
    )

    evaluator = MACEEnergyEvaluator(
        model_path=str(model),
        device=device,
        relax=True,
        relax_cell=True,
        fmax=fmax,
        max_steps=max_relax_steps,
    )

    def evaluate(structure, context: dict) -> dict:
        key = common.configuration_key(structure, cation_sites)
        if key in cache:
            props = {**cache[key], "cached": True}
        else:
            start = time.perf_counter()
            props = evaluator.evaluate_props(structure)
            props["relax_time_s"] = time.perf_counter() - start
            cache[key] = props
            append_lines(cache_path, [{"key": key, "props": props}])
            props = {**props, "cached": False}
        return {**props, "key": key, "x": key[:n_tet].count("A") / n_tet}

    recent: list[dict] = []
    log_start = time.time()

    def on_step(step, checkpoint) -> None:
        nonlocal log_start
        p = step.properties
        row = {
            "step": step.step,
            "accepted": step.accepted,
            "energy_eV": step.energy,
            "x": checkpoint.properties["x"],
            "proposed_energy_eV": p["energy"],
            "proposed_x": p["x"],
            "relax_converged": p.get("relax_converged"),
            "relax_steps": p.get("relax_steps"),
            "relax_time_s": p.get("relax_time_s"),
            "cached": p["cached"],
            "move": [step.move["site_index"], step.move["target_site_index"]],
        }
        # Step first, then checkpoint: a kill in between leaves an extra
        # step line, which the restart trims.
        append_lines(steps_path, [row])
        save_checkpoint(checkpoint, checkpoint_path)
        recent.append(row)
        if len(recent) == log_every:
            print(
                f"step {row['step']}: x={row['x']:.3f} E={row['energy_eV']:.4f} "
                f"acc={np.mean([r['accepted'] for r in recent]):.2f} "
                f"cache hits={np.mean([r['cached'] for r in recent]):.2f} "
                f"{(time.time() - log_start) / len(recent):.1f} s/step",
                flush=True,
            )
            recent.clear()
            log_start = time.time()

    policy = sublattice_swap_policy(parent, sublattices=[cation_sites], evaluate_props=evaluate)
    options = dict(num_structures_to_save=0, on_step=on_step)
    if checkpoint is None:
        start = common.structure_from_key(parent, tet, octa, common.initial_key(tet, octa, initial, seed))
        run_monte_carlo(start, policy, temperature_K, steps, seed=seed, **options)
    else:
        resume_monte_carlo(checkpoint, policy, temperature_K, steps, **options)
    print(f"Finished {steps} steps.", flush=True)
