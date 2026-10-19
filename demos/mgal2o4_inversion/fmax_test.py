"""How tight must MACE relaxations be for Monte Carlo on MgAl2O4?

Relaxes a few 112-atom configurations (positions + cell, LBFGS) while
tightening fmax step by step, recording the energy, cumulative optimizer
steps, and time at each threshold. Continuing the same optimizer reproduces
exactly what a fresh run stopped at each threshold would give, so one
relaxation per structure covers every threshold.

For Monte Carlo, the error in the energy *difference* between configurations
must be small compared with kT (0.09-0.16 eV over 1000-1873 K); the script
reports exchange energies at each fmax and their change from the tightest.

Usage::

    python demos/mgal2o4_inversion/fmax_test.py --model path/to/model.model
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402  (shared structure helpers)

#: Inverted tetrahedral sites (of 16) for the test structures.
INVERSIONS = (0, 1, 4, 8)
FMAX_STEPS = (0.2, 0.1, 0.05, 0.02, 0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS
    from mace.calculators import MACECalculator
    from pymatgen.io.ase import AseAtomsAdaptor

    torch.set_num_threads(args.threads)
    calculator = MACECalculator(model_paths=args.model, device=args.device, default_dtype="float64")
    parent = common.load_parent(common.REFERENCE_CIF).get_primitive_structure()
    parent.make_supercell([2, 2, 2])
    tet, octa = common.cation_sites(parent)
    rng = np.random.default_rng(args.seed)

    common.RAW.mkdir(parents=True, exist_ok=True)
    cache_path = common.RAW / f"fmax_test_{Path(args.model).stem}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for k in INVERSIONS:
        al_tet = rng.choice(tet, k, replace=False)
        mg_oct = rng.choice(octa, k, replace=False)
        if str(k) in cache:
            continue
        atoms = AseAtomsAdaptor.get_atoms(common.decorate(parent, tet, octa, al_tet, mg_oct))
        atoms.calc = calculator
        opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
        start = time.perf_counter()
        rows = []
        for fmax in FMAX_STEPS:
            converged = bool(opt.run(fmax=fmax, steps=args.max_steps - opt.nsteps))
            rows.append(
                {
                    "fmax": fmax,
                    "energy_eV": float(atoms.get_potential_energy()),
                    "steps": int(opt.nsteps),
                    "time_s": time.perf_counter() - start,
                    "converged": converged,
                }
            )
            print(f"k={k} fmax={fmax}: E={rows[-1]['energy_eV']:.5f} steps={opt.nsteps} t={rows[-1]['time_s']:.0f}s", flush=True)
        cache[str(k)] = rows
        cache_path.write_text(json.dumps(cache, indent=1))

    n_fu = len(tet)
    summary = []
    print(f"\nExchange energy (E_k - E_0)/k in eV, and change from fmax={FMAX_STEPS[-1]}:")
    for j, fmax in enumerate(FMAX_STEPS):
        e0 = cache["0"][j]["energy_eV"]
        row = {"fmax": fmax, "mean_time_s": float(np.mean([cache[str(k)][j]["time_s"] for k in INVERSIONS]))}
        for k in INVERSIONS[1:]:
            exchange = (cache[str(k)][j]["energy_eV"] - e0) / k
            tight = (cache[str(k)][-1]["energy_eV"] - cache["0"][-1]["energy_eV"]) / k
            row[f"exchange_k{k}_eV"] = exchange
            row[f"error_k{k}_eV"] = exchange - tight
        # Worst error in the energy of a whole-cell configuration difference.
        row["max_cell_energy_difference_error_eV"] = max(
            abs((cache[str(k)][j]["energy_eV"] - e0) - (cache[str(k)][-1]["energy_eV"] - cache["0"][-1]["energy_eV"]))
            for k in INVERSIONS[1:]
        )
        summary.append(row)
        print(
            f"  fmax {fmax:<5} "
            + "  ".join(f"k={k}: {row[f'exchange_k{k}_eV']:.4f} ({row[f'error_k{k}_eV']:+.4f})" for k in INVERSIONS[1:])
            + f"   max cell error {row['max_cell_energy_difference_error_eV']:.4f} eV   mean time {row['mean_time_s']:.0f} s"
        )
    with open(common.RESULTS / "fmax_test.json", "w") as f:
        json.dump(
            {"model": Path(args.model).name, "n_atoms": len(parent), "n_formula_units": n_fu, "threads": args.threads, "summary": summary, "relaxations": cache},
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
