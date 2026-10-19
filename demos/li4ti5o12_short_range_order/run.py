"""Li4Ti5O12 short-range-order demo: Li/Ti short-range order on the 16d sublattice of Li4Ti5O12.

In spinel Li4Ti5O12 the 16d (octahedral) sites are shared by Li+ (1/6) and
Ti4+ (5/6). This demo runs Metropolis Monte Carlo over Li/Ti arrangements on
those sites -- with :func:`disorderly.sublattice_swap_policy`, so 8a Li and
O never move -- scored by Ewald energy via :class:`disorderly.PrecomputedEwald`,
and records the Warren-Cowley short-range-order parameter for Li-Li pairs on
the first few 16d neighbor shells as a function of temperature.

The cell is a 2x2x3 supercell of the primitive spinel cell: 168 sites with
48 16d sites holding 8 Li and 40 Ti. (The conventional cell's 16 16d sites
would need 2.67 Li, which cannot be ordered at the correct composition.)

With ``--mace MODEL``, it instead checks whether MACE agrees with Ewald's
short-range order: it compares MACE energies of Monte Carlo snapshots from
low and intermediate temperatures with those of random arrangements. The
snapshots are stored as ordering keys in ``results/sro_summary.json``, so this
step only needs the committed results.

Usage::

    python demos/li4ti5o12_short_range_order/run.py [--steps 100000] [--jobs 6]
    python demos/li4ti5o12_short_range_order/run.py --mace path/to/model.model
"""

from __future__ import annotations

import os

# One BLAS thread per process: temperatures run in parallel processes.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
from pymatgen.core import Structure

from disorderly import (
    KB_EV_PER_K,
    PrecomputedEwald,
    ordering_key,
    random_ordered_configuration,
    run_monte_carlo,
    structure_from_ordering_key,
    sublattice_swap_policy,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_SHELLS, RAW, RESULTS, SUPERCELL, ShortRangeOrder, build_parent  # noqa: E402

#: Default temperature grid (K). Point-charge Ewald keeps perfect first-shell
#: Li-Li avoidance up to ~3e4 K, so the grid extends far beyond physical
#: temperatures to show the full loss of short-range order.
TEMPERATURES = (2000, 5000, 10000, 20000, 30000, 45000, 70000, 100000, 150000, 250000, 500000, 1000000)
#: Snapshots kept per temperature (evenly spaced over production steps).
SNAPSHOTS_PER_TEMPERATURE = 10
#: Runs accepting fewer moves than this are frozen rather than equilibrated.
MIN_ACCEPTANCE = 0.01


def _current_series(trajectory, key: str) -> np.ndarray:
    """Per-step value of a property for the *current* configuration.

    Step properties describe the proposed configuration, so the current value
    is the last accepted proposal's.
    """
    values = np.empty(len(trajectory.steps))
    for s, step in enumerate(trajectory.steps):
        values[s] = step.properties[key] if step.accepted else values[s - 1]
    return values


def _mc_worker(temperature: float, num_steps: int, equilibration: int, seed: int) -> dict:
    parent = build_parent()
    model = PrecomputedEwald(parent)
    sro = ShortRangeOrder(parent, model)

    def evaluate(structure: Structure, context: dict) -> dict:
        alphas = sro(structure)
        return {"energy": model(structure), **{f"alpha_{n + 1}": a for n, a in enumerate(alphas)}}

    policy = sublattice_swap_policy(parent, evaluate_props=evaluate)
    initial = random_ordered_configuration(parent, np.random.default_rng(seed))
    record_every = max(1, (num_steps - equilibration) // SNAPSHOTS_PER_TEMPERATURE)

    start = time.perf_counter()
    trajectory = run_monte_carlo(
        initial,
        policy,
        temperature,
        num_steps,
        num_structures_to_save=1,
        seed=seed,
        record_structure_every=record_every,
    )
    runtime = time.perf_counter() - start

    snapshots = [
        {"step": step.step, "ordering_key": ordering_key(parent, step.structure)}
        for step in trajectory.steps
        if step.structure is not None and step.step > equilibration
    ]
    energies = np.array(trajectory.energies)
    alphas = np.array([_current_series(trajectory, f"alpha_{n + 1}") for n in range(N_SHELLS)])
    # Full trajectory JSON would be ~60 MB per temperature; keep the series.
    np.savez_compressed(
        RAW / f"series_T{temperature:.0f}K.npz", energy=energies, alpha=alphas, accepted=np.array(trajectory.accepted)
    )

    return {
        "temperature": temperature,
        "seed": seed,
        "energies": energies,
        "alphas": alphas,
        "acceptance_rate": trajectory.acceptance_rate(),
        "runtime_s": runtime,
        "snapshots": snapshots,
        "lowest_energy": trajectory.saved_configurations[0].energy,
        "lowest_energy_key": ordering_key(parent, trajectory.saved_configurations[0].structure),
    }


def block_mean(values: np.ndarray, n_blocks: int = 20) -> tuple[float, float]:
    """Mean and block-averaged standard error."""
    means = np.array([b.mean() for b in np.array_split(values, n_blocks)])
    return float(values.mean()), float(means.std(ddof=1) / np.sqrt(n_blocks))


def random_baseline(parent: Structure, n_samples: int, seed: int) -> list[float]:
    """Mean Warren-Cowley parameters of random arrangements in this finite
    cell (slightly negative: with a fixed Li count, a Li's neighbors are drawn
    from one fewer Li)."""
    model = PrecomputedEwald(parent)
    sro = ShortRangeOrder(parent, model)
    rng = np.random.default_rng(seed)
    samples = np.array([sro(random_ordered_configuration(parent, rng)) for _ in range(n_samples)])
    return samples.mean(axis=0).tolist()


def run_sweep(args: argparse.Namespace) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    parent = build_parent()
    model = PrecomputedEwald(parent)
    sro = ShortRangeOrder(parent, model)
    temperatures = [float(t) for t in args.temperatures]
    print(
        f"Parent: {len(parent)} sites, {len(sro.sites)} 16d sites; shells at {sro.radii} A "
        f"with {sro.coordination} neighbors",
        flush=True,
    )
    baseline = random_baseline(parent, 2000, args.seed)
    print(f"Random-arrangement baseline alpha: {np.round(baseline, 4).tolist()}", flush=True)

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(_mc_worker, T, args.steps, args.equilibration, args.seed + k)
            for k, T in enumerate(temperatures)
        ]
        runs = [future.result() for future in futures]
    print(f"Monte Carlo finished in {time.perf_counter() - start:.0f} s", flush=True)

    summary = []
    production = slice(args.equilibration, None)
    for run in runs:
        T = run["temperature"]
        energies = run["energies"][production]
        mean_e, err_e = block_mean(energies)
        row = {
            "temperature_K": T,
            "seed": run["seed"],
            "steps": args.steps,
            "equilibration": args.equilibration,
            "acceptance_rate": run["acceptance_rate"],
            "runtime_s": run["runtime_s"],
            "mean_energy_eV": mean_e,
            "mean_energy_err": err_e,
            "heat_capacity_eV_per_K": float(energies.var() / (KB_EV_PER_K * T**2)),
            "lowest_energy_eV": run["lowest_energy"],
            "lowest_energy_key": run["lowest_energy_key"],
            "snapshots": run["snapshots"],
        }
        for n in range(N_SHELLS):
            series = run["alphas"][n][production]
            mean, err = block_mean(series)
            half = len(series) // 2
            row[f"alpha_{n + 1}"] = mean
            row[f"alpha_{n + 1}_err"] = err
            # Equilibration check: change between halves of the production run.
            row[f"alpha_{n + 1}_drift"] = float(series[half:].mean() - series[:half].mean())
        summary.append(row)
        print(
            f"T={T:>7.0f} K  acc={row['acceptance_rate']:.3f}  "
            + "  ".join(
                f"a{n + 1}={row[f'alpha_{n + 1}']:+.3f}+/-{row[f'alpha_{n + 1}_err']:.3f}" for n in range(N_SHELLS)
            )
            + f"  drift a1={row['alpha_1_drift']:+.3f}",
            flush=True,
        )

    with open(RESULTS / "sro_summary.json", "w") as f:
        json.dump(
            {
                "supercell_of_primitive": SUPERCELL,
                "n_sites": len(parent),
                "n_16d_sites": len(sro.sites),
                "shell_radii_A": sro.radii,
                "shell_coordination": sro.coordination,
                "random_baseline_alpha": baseline,
                "runs": summary,
            },
            f,
            indent=1,
        )


def run_mace(args: argparse.Namespace) -> None:
    import torch
    from scipy.stats import spearmanr

    from disorderly.mace import MACEEnergyEvaluator

    summary_path = RESULTS / "sro_summary.json"
    if not summary_path.exists():
        raise SystemExit("No sro_summary.json found; run this script once without --mace first.")
    summary = json.loads(summary_path.read_text())
    runs = sorted(summary["runs"], key=lambda r: r["temperature_K"])
    parent = build_parent()

    # Groups: the lowest equilibrated temperature (frozen runs accept almost
    # no moves), the temperature where alpha_1 is closest to halfway between
    # that run and the random baseline, and random arrangements.
    equilibrated = [r for r in runs if r["acceptance_rate"] >= MIN_ACCEPTANCE]
    low = equilibrated[0]
    target = 0.5 * (low["alpha_1"] + summary["random_baseline_alpha"][0])
    middle = min(equilibrated[1:], key=lambda r: abs(r["alpha_1"] - target))
    n = args.mace_samples
    rng = np.random.default_rng(args.seed + 1000)
    groups = {
        f"MC {low['temperature_K']:.0f} K": [s["ordering_key"] for s in low["snapshots"][-n:]],
        f"MC {middle['temperature_K']:.0f} K": [s["ordering_key"] for s in middle["snapshots"][-n:]],
        "random": [ordering_key(parent, random_ordered_configuration(parent, rng)) for _ in range(n)],
    }
    names = list(groups)
    print(f"MACE: {sum(len(k) for k in groups.values())} structures in groups {names}", flush=True)

    # One process using every core: relaxing 168 atoms single-threaded in
    # parallel workers is several times slower per structure.
    torch.set_num_threads(os.cpu_count())
    ewald = PrecomputedEwald(parent)
    sro = ShortRangeOrder(parent, ewald)
    single_point = MACEEnergyEvaluator(model_path=args.mace, device=args.mace_device, relax=False)
    relaxed = MACEEnergyEvaluator(
        model_path=args.mace, device=args.mace_device, relax=True, relax_cell=True, fmax=args.mace_fmax
    )
    # Finished structures are cached so an interrupted run resumes.
    RAW.mkdir(parents=True, exist_ok=True)
    cache_path = RAW / f"mace_cache_{Path(args.mace).stem}_fmax{args.mace_fmax}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    start = time.perf_counter()
    rows = []
    for name in names:
        for key in groups[name]:
            if key not in cache:
                structure = structure_from_ordering_key(parent, key)
                cache[key] = {
                    "mace_single_point_eV": float(single_point(structure)),
                    "mace_relaxed_eV": float(relaxed(structure)),
                }
                cache_path.write_text(json.dumps(cache, indent=1))
                print(f"  {len(cache)} done ({time.perf_counter() - start:.0f} s)", flush=True)
            structure = structure_from_ordering_key(parent, key)
            rows.append({"group": name, "ordering_key": key, "alpha": sro(structure), "ewald_eV": ewald(structure), **cache[key]})
    print(f"MACE calculations finished in {time.perf_counter() - start:.0f} s", flush=True)

    n_li_16d = 8
    reference = {k: np.mean([r[k] for r in rows if r["group"] == "random"]) for k in ("ewald_eV", "mace_single_point_eV", "mace_relaxed_eV")}
    group_stats = {}
    for name in names:
        members = [r for r in rows if r["group"] == name]
        stats = {"alpha_1": float(np.mean([r["alpha"][0] for r in members]))}
        for k in ("ewald_eV", "mace_single_point_eV", "mace_relaxed_eV"):
            values = (np.array([r[k] for r in members]) - reference[k]) / n_li_16d
            stats[f"{k}_per_16d_Li_vs_random"] = float(values.mean())
            stats[f"{k}_per_16d_Li_vs_random_err"] = float(values.std(ddof=1) / np.sqrt(len(values)))
        group_stats[name] = stats
    alpha_1 = [r["alpha"][0] for r in rows]
    correlations = {
        k: float(spearmanr(alpha_1, [r[k] for r in rows]).statistic)
        for k in ("ewald_eV", "mace_single_point_eV", "mace_relaxed_eV")
    }
    print(json.dumps({"groups": group_stats, "spearman_alpha_1_vs_energy": correlations}, indent=2))

    with open(RESULTS / "mace_check.json", "w") as f:
        json.dump(
            {
                "model": Path(args.mace).name,
                "device": args.mace_device,
                "fmax": args.mace_fmax,
                "groups": group_stats,
                "spearman_alpha_1_vs_energy": correlations,
                "structures": rows,
            },
            f,
            indent=1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=100_000, help="Monte Carlo steps per temperature.")
    parser.add_argument("--equilibration", type=int, default=20_000, help="Initial steps discarded from averages.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=TEMPERATURES)
    parser.add_argument("--jobs", type=int, default=6, help="Parallel processes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mace", metavar="MODEL", help="Check Ewald short-range order against this MACE model.")
    parser.add_argument("--mace-samples", type=int, default=6, help="Structures per group for the MACE check.")
    parser.add_argument("--mace-device", default="cpu")
    parser.add_argument("--mace-fmax", type=float, default=0.05, help="Relaxation force threshold (eV/A).")
    args = parser.parse_args()
    if args.mace:
        run_mace(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
