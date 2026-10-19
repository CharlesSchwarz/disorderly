"""Write the inputs for the B2 ordering demo.

Writes the disordered parent ``parent.cif`` (an L x L x L bcc supercell with
every site half Cu, half Zn) and two starting structures in
``initial_structures/``:

- ``ordered.cif`` -- perfect B2 (Cu on the corners, Zn on the body centres),
  for heating runs;
- ``random_seed<seed>.cif`` -- a random arrangement from
  :func:`disorderly.random_ordered_configuration`, for cooling runs.

Above T_c both should end disordered and below T_c both ordered. Then prints
the ``run-monte-carlo`` commands (the energy model is in
``monte_carlo_settings.py``).

Usage::

    python demos/bcc_ising_ordering/make_inputs.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from pymatgen.core import Lattice, Structure

from disorderly import random_ordered_configuration

HERE = Path(__file__).resolve().parent
# pymatgen writes one CIF row per species on a mixed site, repeating its label.
warnings.filterwarnings("ignore", message="Site labels are not unique")

# --- Settings: edit these ----------------------------------------------------
LATTICE_A = 2.95  # A, beta-brass
SUPERCELL = 6  # L: an L x L x L supercell of the 2-atom cube has 2 L^3 sites (432 for L = 6)
SEEDS = [0]  # random starting arrangements
TEMPERATURES = [500, 600, 650, 700, 725, 750, 775, 800, 900, 1000]  # K; only used to print the run commands
# ------------------------------------------------------------------------------

BCC = [[0, 0, 0], [0.5, 0.5, 0.5]]


def main() -> None:
    parent = Structure(Lattice.cubic(LATTICE_A), [{"Cu": 0.5, "Zn": 0.5}] * 2, BCC)
    parent.make_supercell(SUPERCELL)
    ordered = Structure(Lattice.cubic(LATTICE_A), ["Cu", "Zn"], BCC)
    ordered.make_supercell(SUPERCELL)

    out = HERE / "initial_structures"
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "ordered.cif"]
    ordered.to(filename=str(paths[0]))
    for seed in SEEDS:
        random = random_ordered_configuration(parent, np.random.default_rng(seed))
        paths.append(out / f"random_seed{seed}.cif")
        random.to(filename=str(paths[-1]))
    parent_path = HERE / "parent.cif"
    parent.to(filename=str(parent_path))

    print(f"Wrote {parent_path.name} ({len(parent)} sites) and {len(paths)} starting structures\n")
    print("Run (from the repository root):")
    root = HERE.parent.parent
    for path in paths:
        for temperature in TEMPERATURES:
            print(
                f"run-monte-carlo {parent_path.relative_to(root).as_posix()} {path.relative_to(root).as_posix()} "
                f"{temperature} {(HERE / 'monte_carlo_settings.py').relative_to(root).as_posix()}"
            )


if __name__ == "__main__":
    main()
