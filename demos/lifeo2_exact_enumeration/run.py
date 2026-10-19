"""LiFeO2 exact-enumeration demo: validate the Monte Carlo engine against exact enumeration.

System: disordered rock-salt alpha-LiFeO2 (Li+/Fe3+ sharing the cation
sublattice 50/50), in a 32-site supercell that keeps the full Fm-3m
symmetry. Its 16 cation sites have C(16, 8) = 12870 orderings, few enough
to evaluate every one with the same disordered Ewald energy the Monte Carlo
runs use.

The script:

1. Enumerates all orderings and their Ewald energies.
2. Groups them into symmetry classes by canonicalizing each ordering under
   the parent's site permutations, and checks that (a) the class sizes match
   :func:`disorderly.configuration_degeneracy`, (b) they sum to 12870, and
   (c) every member of a class has the same energy.
3. Computes exact <E>(T) and C(T) from the full Boltzmann sum.
4. Runs Metropolis Monte Carlo with :func:`disorderly.sublattice_swap_policy`
   at temperatures around the exact heat-capacity peak, in parallel.

With ``--mace MODEL``, it instead reuses the saved enumeration and computes
exact thermodynamics for MACE energies. Because energies are identical
within a symmetry class, this needs only one relaxed MACE calculation per
class (40 in total) rather than one per ordering. A second, randomly chosen
member of each class gets a single-point calculation to check that.

Results are written to ``results/`` (summaries) and ``results/raw/``
(trajectories, not committed). Run ``analysis.py`` afterwards for figures.

Usage::

    python demos/lifeo2_exact_enumeration/run.py [--steps 60000] [--jobs 6]
    python demos/lifeo2_exact_enumeration/run.py --mace path/to/model.model
"""

from __future__ import annotations

import os

# One BLAS thread per process: Monte Carlo temperatures run in parallel processes.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import time

import numpy as np
from pymatgen.core import Lattice, Structure

from disorderly import (
    KB_EV_PER_K,
    configuration_degeneracy,
    disordered_ewald_energy,
    get_spacegroup,
    run_monte_carlo,
    save_trajectory,
    site_permutations,
    sublattice_swap_policy,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW = RESULTS / "raw"

#: Approximate lattice parameter of disordered alpha-LiFeO2 (Angstrom).
LATTICE_PARAMETER = 4.158
#: Supercell of the conventional rock-salt cell with 16 cation sites whose
#: lattice keeps the full cubic point group (768 site permutations).
SUPERCELL = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
#: Monte Carlo temperatures as multiples of the exact heat-capacity peak.
#: Below ~0.75x the system almost never leaves its ground state, so short
#: runs see no fluctuations to compare.
TEMPERATURE_FACTORS = (0.75, 1.0, 1.5, 2.5, 5.0, 10.0)


def build_parent() -> Structure:
    parent = Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(LATTICE_PARAMETER),
        [{"Li+": 0.5, "Fe3+": 0.5}, "O2-"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    parent.make_supercell(SUPERCELL)
    return parent


def cation_indices(parent: Structure) -> list[int]:
    return [i for i, site in enumerate(parent) if not site.is_ordered]


def decorate(parent: Structure, cations: list[int], occupation) -> Structure:
    """Ordered structure with Li+ on cation sites where ``occupation`` is
    true and Fe3+ elsewhere. Site order matches ``parent``."""
    structure = parent.copy()
    for i, is_li in zip(cations, occupation):
        structure.replace(i, "Li+" if is_li else "Fe3+")
    return structure


def occupation_codes(occupations: np.ndarray) -> np.ndarray:
    """Encode boolean occupation rows (Li = True) as integers."""
    return occupations.astype(np.int64) @ (1 << np.arange(occupations.shape[1], dtype=np.int64))


def _energies_worker(occupations: np.ndarray) -> np.ndarray:
    parent = build_parent()
    cations = cation_indices(parent)
    return np.array([disordered_ewald_energy(decorate(parent, cations, occ)) for occ in occupations])


def enumerate_orderings(n_cations: int, jobs: int) -> tuple[np.ndarray, np.ndarray]:
    combos = list(itertools.combinations(range(n_cations), n_cations // 2))
    occupations = np.zeros((len(combos), n_cations), dtype=bool)
    for k, li_sites in enumerate(combos):
        occupations[k, list(li_sites)] = True
    chunks = np.array_split(occupations, 4 * jobs)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        energies = np.concatenate(list(executor.map(_energies_worker, chunks)))
    return occupations, energies


def symmetry_classes(parent: Structure, cations: list[int], occupations: np.ndarray):
    """Assign each ordering to a symmetry class.

    Returns ``(class_id, perms)``, where ``class_id[k]`` labels ordering
    ``k`` and classes are numbered in order of first appearance.
    """
    perms = site_permutations(parent)
    local = {site: a for a, site in enumerate(cations)}
    # Same convention as configuration_degeneracy: new_label[j] = label[p[j]].
    cation_perms = np.array([[local[p[i]] for i in cations] for p in perms])
    weights = 1 << np.arange(len(cations), dtype=np.int64)
    canonical = np.empty(len(occupations), dtype=np.int64)
    for start in range(0, len(occupations), 1000):
        block = occupations[start : start + 1000]
        canonical[start : start + 1000] = (block[:, cation_perms].astype(np.int64) @ weights).min(axis=1)
    _, first, class_id = np.unique(canonical, return_index=True, return_inverse=True)
    # Renumber classes by first appearance so the numbering is stable.
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[class_id], perms


def exact_thermodynamics(
    energies: np.ndarray, temperatures: np.ndarray, degeneracy: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Exact canonical averages.

    ``energies`` are either one per ordering (each counted once) or one per
    symmetry class, with ``degeneracy`` giving each class's multiplicity.
    """
    g = np.ones(len(energies)) if degeneracy is None else np.asarray(degeneracy, dtype=float)
    shifted = energies - energies.min()
    beta = 1.0 / (KB_EV_PER_K * temperatures[:, None])
    weights = g[None, :] * np.exp(-beta * shifted[None, :])
    z = weights.sum(axis=1)
    mean = (weights * energies).sum(axis=1) / z
    mean_sq = (weights * energies**2).sum(axis=1) / z
    heat_capacity = (mean_sq - mean**2) / (KB_EV_PER_K * temperatures**2)
    return {"mean_energy": mean, "heat_capacity": heat_capacity}


def class_probabilities(class_energy: np.ndarray, degeneracy: np.ndarray, temperature: float) -> np.ndarray:
    """Exact probability of each symmetry class, g_i exp(-E_i/kT) / Z."""
    weights = degeneracy * np.exp(-(class_energy - class_energy.min()) / (KB_EV_PER_K * temperature))
    return weights / weights.sum()


def _mc_worker(temperature: float, num_steps: int, seed: int) -> dict:
    data = np.load(RESULTS / "enumeration.npz")
    index_of_code = {int(code): k for k, code in enumerate(data["codes"])}
    parent = build_parent()
    cations = cation_indices(parent)
    weights = 1 << np.arange(len(cations), dtype=np.int64)

    def evaluate(structure: Structure, context: dict) -> dict:
        occupation = np.array([structure[i].specie.symbol == "Li" for i in cations])
        return {
            "energy": disordered_ewald_energy(structure),
            "config_index": index_of_code[int(occupation @ weights)],
        }

    rng = np.random.default_rng(seed)
    initial = decorate(parent, cations, data["occupations"][rng.integers(len(data["codes"]))])
    policy = sublattice_swap_policy(parent, evaluate_props=evaluate)

    start = time.perf_counter()
    trajectory = run_monte_carlo(initial, policy, temperature, num_steps, num_structures_to_save=1, seed=seed)
    runtime = time.perf_counter() - start
    save_trajectory(trajectory, RAW / f"trajectory_T{temperature:.0f}K.json")

    # Each step's properties describe the *proposed* configuration, so the
    # current configuration is the last accepted proposal.
    current = np.empty(len(trajectory.steps), dtype=np.int64)
    for s, step in enumerate(trajectory.steps):
        current[s] = step.properties["config_index"] if step.accepted else current[s - 1]
    energies = np.array(trajectory.energies)
    mismatch = np.abs(energies - data["energies"][current]).max()
    return {
        "temperature": temperature,
        "seed": seed,
        "energies": energies,
        "config_index": current,
        "acceptance_rate": trajectory.acceptance_rate(),
        "runtime_s": runtime,
        "max_energy_mismatch_vs_enumeration": float(mismatch),
    }


def block_stats(values: np.ndarray, temperature: float, n_blocks: int = 20) -> dict[str, float]:
    """Mean energy and heat capacity with block-averaged standard errors."""
    blocks = np.array_split(values, n_blocks)
    means = np.array([b.mean() for b in blocks])
    caps = np.array([b.var() for b in blocks]) / (KB_EV_PER_K * temperature**2)
    return {
        "mean_energy": float(values.mean()),
        "mean_energy_err": float(means.std(ddof=1) / np.sqrt(n_blocks)),
        "heat_capacity": float(values.var() / (KB_EV_PER_K * temperature**2)),
        "heat_capacity_err": float(caps.std(ddof=1) / np.sqrt(n_blocks)),
    }


def _z_score(value: float, reference: float, err: float) -> float | None:
    """(value - reference) / err, or None when there were no fluctuations."""
    return (value - reference) / err if err > 0 else None


def _fmt(z: float | None) -> str:
    return "n/a" if z is None else f"{z:+.2f}"


_MACE_EVALUATORS: dict = {}


def _mace_worker(model_path: str, device: str, fmax: float, occupations: np.ndarray) -> dict:
    """Single-point MACE energies of every ordering in ``occupations`` and a
    relaxed (positions + cell) energy of the first."""
    from disorderly.mace import MACEEnergyEvaluator

    key = (model_path, device, fmax)
    if key not in _MACE_EVALUATORS:
        _MACE_EVALUATORS[key] = (
            MACEEnergyEvaluator(model_path=model_path, device=device, relax=False),
            MACEEnergyEvaluator(model_path=model_path, device=device, relax=True, relax_cell=True, fmax=fmax),
        )
    single_point, relaxed = _MACE_EVALUATORS[key]
    parent = build_parent()
    cations = cation_indices(parent)
    structures = [decorate(parent, cations, occupation) for occupation in occupations]
    return {
        "single_point": [float(single_point(s)) for s in structures],
        "relaxed": float(relaxed(structures[0])),
    }


def run_mace(args: argparse.Namespace) -> None:
    from scipy.stats import spearmanr

    if not (RESULTS / "enumeration.npz").exists() or not (RESULTS / "exact.json").exists():
        raise SystemExit("No saved enumeration found; run this script once without --mace first.")
    data = np.load(RESULTS / "enumeration.npz")
    classes = json.loads((RESULTS / "exact.json").read_text())["classes"]  # sorted by Ewald energy

    rng = np.random.default_rng(args.seed)
    tasks = []
    for c in classes:
        members = np.flatnonzero(data["class_id"] == c["class"])
        picks = [members[0]] + ([int(rng.choice(members[1:]))] if len(members) > 1 else [])
        tasks.append(data["occupations"][picks])

    print(f"MACE: {len(tasks)} symmetry classes, model {args.mace}", flush=True)
    start = time.perf_counter()
    n = len(tasks)
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        results = list(
            executor.map(_mace_worker, [args.mace] * n, [args.mace_device] * n, [args.mace_fmax] * n, tasks)
        )
    print(f"MACE calculations finished in {time.perf_counter() - start:.0f} s")

    degeneracy = np.array([c["degeneracy"] for c in classes])
    ewald = np.array([c["energy_eV"] for c in classes])
    single_point = np.array([r["single_point"][0] for r in results])
    relaxed = np.array([r["relaxed"] for r in results])
    single_point_spread = max(np.ptp(r["single_point"]) for r in results)

    grid = np.geomspace(10, 1e6, 500)
    curves = {}
    for name, energies in (("relaxed", relaxed), ("single_point", single_point)):
        thermo = exact_thermodynamics(energies, grid, degeneracy)
        curves[name] = {
            "heat_capacity_peak_K": float(grid[np.argmax(thermo["heat_capacity"])]),
            "mean_energy_eV": thermo["mean_energy"].tolist(),
            "heat_capacity_eV_per_K": thermo["heat_capacity"].tolist(),
        }
    ground = int(np.argmin(relaxed))
    checks = {
        "max_single_point_spread_within_class_eV": float(single_point_spread),
        "spearman_ewald_vs_mace_relaxed": float(spearmanr(ewald, relaxed).statistic),
        "mace_relaxed_ground_state": classes[ground]["spacegroup"],
        "mace_relaxed_ground_state_ewald_rank": ground + 1,
        "heat_capacity_peak_K_relaxed": curves["relaxed"]["heat_capacity_peak_K"],
        "heat_capacity_peak_K_single_point": curves["single_point"]["heat_capacity_peak_K"],
    }
    print(json.dumps(checks, indent=2))

    with open(RESULTS / "mace_exact.json", "w") as f:
        json.dump(
            {
                "model": Path(args.mace).name,
                "device": args.mace_device,
                "fmax": args.mace_fmax,
                "checks": checks,
                "temperature_K": grid.tolist(),
                "curves": curves,
                "classes": [
                    {**c, "mace_single_point_eV": float(single_point[k]), "mace_relaxed_eV": float(relaxed[k])}
                    for k, c in enumerate(classes)
                ],
            },
            f,
            indent=1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=60_000, help="Monte Carlo steps per temperature.")
    parser.add_argument("--equilibration", type=int, default=6_000, help="Initial steps discarded from averages.")
    parser.add_argument("--jobs", type=int, default=len(TEMPERATURE_FACTORS), help="Parallel processes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mace", metavar="MODEL", help="Compute exact MACE thermodynamics with this model instead.")
    parser.add_argument("--mace-device", default="cpu")
    parser.add_argument("--mace-fmax", type=float, default=0.05, help="Relaxation force threshold (eV/A).")
    args = parser.parse_args()
    if args.mace:
        run_mace(args)
        return
    RAW.mkdir(parents=True, exist_ok=True)

    parent = build_parent()
    cations = cation_indices(parent)
    print(f"Parent: {parent.composition.reduced_formula}, {len(parent)} sites, {len(cations)} cation sites")

    # 1. Enumerate.
    start = time.perf_counter()
    occupations, energies = enumerate_orderings(len(cations), args.jobs)
    print(f"Enumerated {len(energies)} orderings in {time.perf_counter() - start:.0f} s")

    # 2. Symmetry classes and checks.
    class_id, perms = symmetry_classes(parent, cations, occupations)
    n_classes = class_id.max() + 1
    degeneracy = np.bincount(class_id)
    representatives = np.array([np.flatnonzero(class_id == c)[0] for c in range(n_classes)])
    class_energy = energies[representatives]
    energy_spread = np.array([np.ptp(energies[class_id == c]) for c in range(n_classes)])
    module_degeneracy = np.array(
        [configuration_degeneracy(parent, decorate(parent, cations, occupations[k]), perms=perms) for k in representatives]
    )
    spacegroups = [get_spacegroup(decorate(parent, cations, occupations[k])).symbol for k in representatives]
    checks = {
        "n_orderings": int(len(energies)),
        "n_symmetry_operations": len(perms),
        "n_classes": int(n_classes),
        "sum_of_degeneracies": int(degeneracy.sum()),
        "degeneracy_matches_configuration_degeneracy": bool((module_degeneracy == degeneracy).all()),
        "max_energy_spread_within_class_eV": float(energy_spread.max()),
    }
    print(json.dumps(checks, indent=2))

    np.savez_compressed(
        RESULTS / "enumeration.npz",
        occupations=occupations,
        codes=occupation_codes(occupations),
        energies=energies,
        class_id=class_id,
    )
    order = np.argsort(class_energy)
    classes = [
        {
            "class": int(c),
            "energy_eV": float(class_energy[c]),
            "degeneracy": int(degeneracy[c]),
            "spacegroup": spacegroups[c],
            "li_sites": np.flatnonzero(occupations[representatives[c]]).tolist(),
        }
        for c in order
    ]

    # 3. Exact thermodynamics.
    grid = np.geomspace(100, 1e6, 400)
    exact = exact_thermodynamics(energies, grid)
    t_peak = float(grid[np.argmax(exact["heat_capacity"])])
    temperatures = [float(round(f * t_peak, -1)) for f in TEMPERATURE_FACTORS]
    print(f"Exact heat-capacity peak at {t_peak:.0f} K; Monte Carlo temperatures {temperatures}")
    with open(RESULTS / "exact.json", "w") as f:
        json.dump(
            {
                "checks": checks,
                "heat_capacity_peak_K": t_peak,
                "temperature_K": grid.tolist(),
                "mean_energy_eV": exact["mean_energy"].tolist(),
                "heat_capacity_eV_per_K": exact["heat_capacity"].tolist(),
                "classes": classes,
            },
            f,
            indent=1,
        )

    # 4. Monte Carlo.
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(_mc_worker, T, args.steps, args.seed + k) for k, T in enumerate(temperatures)
        ]
        runs = [future.result() for future in futures]
    print(f"Monte Carlo finished in {time.perf_counter() - start:.0f} s")

    summary = []
    for run in runs:
        T = run["temperature"]
        production = slice(args.equilibration, None)
        stats = block_stats(run["energies"][production], T)
        exact_T = exact_thermodynamics(energies, np.array([T]))
        p_exact = class_probabilities(class_energy, degeneracy, T)
        p_mc = np.bincount(class_id[run["config_index"][production]], minlength=n_classes) / len(
            run["config_index"][production]
        )
        row = {
            "temperature_K": T,
            "seed": run["seed"],
            "steps": args.steps,
            "equilibration": args.equilibration,
            "acceptance_rate": run["acceptance_rate"],
            "runtime_s": run["runtime_s"],
            "max_energy_mismatch_vs_enumeration_eV": run["max_energy_mismatch_vs_enumeration"],
            **{f"mc_{k}": v for k, v in stats.items()},
            "exact_mean_energy": float(exact_T["mean_energy"][0]),
            "exact_heat_capacity": float(exact_T["heat_capacity"][0]),
            "class_total_variation_distance": float(0.5 * np.abs(p_mc - p_exact).sum()),
            "class_probability_exact": p_exact[order].tolist(),
            "class_probability_mc": p_mc[order].tolist(),
        }
        row["mean_energy_z"] = _z_score(row["mc_mean_energy"], row["exact_mean_energy"], row["mc_mean_energy_err"])
        row["heat_capacity_z"] = _z_score(
            row["mc_heat_capacity"], row["exact_heat_capacity"], row["mc_heat_capacity_err"]
        )
        summary.append(row)
        print(
            f"T={T:>8.0f} K  <E> MC {row['mc_mean_energy']:.4f} +/- {row['mc_mean_energy_err']:.4f}"
            f"  exact {row['exact_mean_energy']:.4f}  (z={_fmt(row['mean_energy_z'])})"
            f"  C z={_fmt(row['heat_capacity_z'])}  TVD={row['class_total_variation_distance']:.3f}"
            f"  acc={row['acceptance_rate']:.2f}"
        )

    with open(RESULTS / "mc_summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
