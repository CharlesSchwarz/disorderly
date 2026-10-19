"""MACE Monte Carlo for Li/Ti short-range order in Li4Ti5O12: one run in one folder.

:func:`run_temperature` runs Metropolis Monte Carlo over Li/Ti arrangements on
the 48 16d sites of the 168-site cell (the same cell and moves as the Ewald
runs in ``run.py``: :func:`disorderly.sublattice_swap_policy`, so the 8a Li
and O never move) at one temperature, scoring every candidate with a MACE
energy, relaxed (positions + cell) by default. Every step also records the
Warren-Cowley alpha of the first two 16d shells and the Ewald energy.

Each run lives in its own folder, made by ``setup_sweep.py`` with a
``run_mc.py`` that calls :func:`run_temperature`. Everything the run writes
goes into that folder:

- ``metadata.json`` -- the run's settings. A rerun with different settings is
  refused rather than mixed into the same trajectory.
- ``energy_cache.jsonl`` -- the MACE energy of every arrangement evaluated so
  far, keyed by :func:`disorderly.ordering_key`. Energies are deterministic,
  so revisited arrangements cost nothing and a restart loses no relaxation.
- ``steps.jsonl`` -- one line per Monte Carlo step.
- ``checkpoint.json`` -- a :class:`disorderly.MonteCarloCheckpoint` written
  after every step. Running again continues with
  :func:`disorderly.resume_monte_carlo` exactly as an uninterrupted run
  would, so a job killed at its wall-time limit loses at most the one
  energy evaluation in progress.
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

from disorderly import (  # noqa: E402
    PrecomputedEwald,
    load_checkpoint,
    ordering_key,
    resume_monte_carlo,
    run_monte_carlo,
    save_checkpoint,
    sublattice_swap_policy,
)

#: Settings that must match for a run folder to be continued.
FIXED_SETTINGS = ("temperature_K", "initial", "seed", "model", "relax", "relax_cell", "fmax", "max_relax_steps")


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
    relax: bool = True,
    relax_cell: bool = True,
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
        initial: ``"ordered"`` (lowest-energy Ewald arrangement, alpha_1 = -0.2)
            or ``"random"`` starting arrangement.
        seed: Seeds the Monte Carlo moves and the random starting arrangement.
        model: Path to the MACE model file.
        device: Torch device for MACE, e.g. ``"cpu"`` or ``"cuda"``.
        relax: Relax each arrangement before reading its energy; ``False``
            uses single-point energies at the ideal positions.
        relax_cell: With ``relax``, relax the cell as well as the positions.
        fmax: Relaxation force threshold (eV/A).
        max_relax_steps: Optimizer step cap per relaxation.
        steps: Total Monte Carlo steps for the run, including steps already done.
        threads: Torch threads; ``None`` uses ``SLURM_CPUS_PER_TASK`` if set, else all cores.
        log_every: Steps between progress lines.
    """
    import torch

    from disorderly.mace import MACEEnergyEvaluator

    if threads is None:
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    torch.set_num_threads(threads)

    parent = common.build_parent()
    ewald = PrecomputedEwald(parent)
    sro = common.ShortRangeOrder(parent, ewald)
    start_structure, start_source = common.initial_configuration(parent, initial, seed)
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "temperature_K": float(temperature_K),
        "initial": initial,
        "initial_source": start_source,
        "seed": seed,
        "model": Path(model).name,
        "relax": relax,
        "relax_cell": relax_cell,
        "fmax": fmax,
        "max_relax_steps": max_relax_steps,
        "n_sites": len(parent),
        "n_16d_sites": len(sro.sites),
        "shell_radii_A": sro.radii,
        "shell_coordination": sro.coordination,
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
        relax=relax,
        relax_cell=relax_cell,
        fmax=fmax,
        max_steps=max_relax_steps,
    )

    def evaluate(structure, context: dict) -> dict:
        key = ordering_key(parent, structure)
        if key in cache:
            props = {**cache[key], "cached": True}
        else:
            start = time.perf_counter()
            props = evaluator.evaluate_props(structure)
            props["eval_time_s"] = time.perf_counter() - start
            cache[key] = props
            append_lines(cache_path, [{"key": key, "props": props}])
            props = {**props, "cached": False}
        alphas = sro(structure)
        return {
            **props,
            "key": key,
            "ewald_eV": ewald(structure),
            **{f"alpha_{n + 1}": a for n, a in enumerate(alphas)},
        }

    recent: list[dict] = []
    log_start = time.time()

    def on_step(step, checkpoint) -> None:
        nonlocal log_start
        p = step.properties
        current = checkpoint.properties
        row = {
            "step": step.step,
            "accepted": step.accepted,
            "energy_eV": step.energy,
            "alpha_1": current["alpha_1"],
            "alpha_2": current["alpha_2"],
            "ewald_eV": current["ewald_eV"],
            "proposed_energy_eV": p["energy"],
            "proposed_alpha_1": p["alpha_1"],
            "relax_converged": p.get("relax_converged"),
            "relax_steps": p.get("relax_steps"),
            "eval_time_s": p.get("eval_time_s"),
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
                f"step {row['step']}: alpha_1={row['alpha_1']:+.3f} alpha_2={row['alpha_2']:+.3f} "
                f"E={row['energy_eV']:.4f} acc={np.mean([r['accepted'] for r in recent]):.2f} "
                f"cache hits={np.mean([r['cached'] for r in recent]):.2f} "
                f"{(time.time() - log_start) / len(recent):.1f} s/step",
                flush=True,
            )
            recent.clear()
            log_start = time.time()

    policy = sublattice_swap_policy(parent, evaluate_props=evaluate)
    options = dict(num_structures_to_save=0, on_step=on_step)
    if checkpoint is None:
        run_monte_carlo(start_structure, policy, temperature_K, steps, seed=seed, **options)
    else:
        resume_monte_carlo(checkpoint, policy, temperature_K, steps, **options)
    print(f"Finished {steps} steps.", flush=True)
