"""Write the inputs for the Cu3Au ordering demo.

Writes the disordered parent ``parent.cif`` (an L x L x L fcc supercell with
every site 3/4 Cu, 1/4 Au) and two starting structures in
``initial_structures/``:

- ``ordered.cif`` -- perfect L1_2 (Au on the corners, Cu on the face
  centres), for heating runs;
- ``random_seed<seed>.cif`` -- a random arrangement from
  :func:`disorderly.random_ordered_configuration`, for cooling runs.

Then prints the ``run-monte-carlo`` commands (MACE settings are in
``monte_carlo_settings.py``).

Usage::

    python demos/cu3au_ordering/make_inputs.py
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
LATTICE_A = 3.75  # A, Cu3Au
SUPERCELL = 3  # L: an L x L x L supercell of the 4-atom cube has 4 L^3 sites (108 for L = 3)
SEEDS = [0]  # random starting arrangements
TEMPERATURES = [400, 500, 550, 600, 625, 650, 675, 700, 750, 800, 900]  # K; only used to print the run commands
# ------------------------------------------------------------------------------

FCC = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]


def main() -> None:
    parent = Structure(Lattice.cubic(LATTICE_A), [{"Cu": 0.75, "Au": 0.25}] * 4, FCC)
    parent.make_supercell(SUPERCELL)
    ordered = Structure(Lattice.cubic(LATTICE_A), ["Au", "Cu", "Cu", "Cu"], FCC)
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
