"""Settings for ``run-monte-carlo``: B2 ordering in a nearest-neighbour pair model.

Run with (from the repository root):

    run-monte-carlo demos/bcc_ising_ordering/parent.cif \\
        demos/bcc_ising_ordering/initial_structures/ordered.cif 700 \\
        demos/bcc_ising_ordering/monte_carlo_settings.py

A 50/50 Cu/Zn alloy on a bcc lattice (beta-brass) with energy

    E = J * sum over nearest-neighbour pairs of s_i * s_j,   s = +1 for Cu, -1 for Zn

so each unlike pair lowers the energy by 2J relative to a like pair. This is
the bcc Ising model, whose ordering temperature is known to high precision:
J / (k_B T_c) = 0.157373, so T_c = 737 K for J = 0.010 eV. Below T_c the
alloy orders into B2 (CsCl): Cu on the cube corners, Zn on the body centres.

Moves are the default composition-preserving swaps of a Cu and a Zn atom.
Each step records the energy and the B2 order parameter ``order`` (1 =
perfect B2, ~0 = random) in ``trajectory.json``.

Same as the reference settings file ``src/disorder_mc/cli/monte_carlo_settings_reference.py``, with
the MACE energy replaced by the pair model.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from disorder_mc import MonteCarloPolicy
from disorder_mc.settings import default_choose_potential_move


num_steps = 200_000
num_structures_to_save = 0
time_limit = None
seed = 0
max_move_attempts = 1000
record_structure_every = None
move_appliers = None
save_best_structures = False

# One results folder per run, named from the initial structure and temperature
# on the command line, e.g. results/raw/ordered_T700K. Relative custom paths
# are resolved relative to this settings file.
output_dir = f"results/raw/{Path(sys.argv[2]).stem}_T{float(sys.argv[3]):g}K" if len(sys.argv) > 3 else None

# Pair-model settings.
J = 0.010  # eV per nearest-neighbour pair; T_c = J / (0.157373 k_B) = 737 K
cutoff = 2.75  # A; between the 1st (2.55 A) and 2nd (2.95 A) neighbour distances for a = 2.95 A


def corner_sublattice(structure):
    """+1 for sites on the cube corners, -1 for body centres.

    ``structure`` is an L x L x L supercell of the 2-atom bcc cube, so a site's
    fractional coordinate times 2L is even on the corners and odd on the
    body centres.
    """
    n_cells = round((len(structure) / 2) ** (1 / 3))
    half_steps = np.round(structure.frac_coords[:, 0] * 2 * n_cells).astype(int)
    return np.where(half_steps % 2 == 0, 1, -1)


def get_policy(disordered_structure, initial_structure, temperature):
    """Return the Monte Carlo policy for this run."""
    # Swaps never move sites, so the neighbour pairs of the starting structure
    # hold for every configuration of the run.
    centres, neighbours, _, _ = initial_structure.get_neighbor_list(cutoff)
    eta = corner_sublattice(initial_structure)

    def evaluate_props(configuration, context):
        s = np.array([1 if site.specie.symbol == "Cu" else -1 for site in configuration])
        return {
            # Each pair appears twice in the neighbour list.
            "energy": J * float(np.sum(s[centres] * s[neighbours])) / 2,
            "order": abs(float(eta @ s)) / len(s),
        }

    return MonteCarloPolicy(
        choose_potential_move=default_choose_potential_move,
        evaluate_props=evaluate_props,
        context={"energy_model": "bcc nearest-neighbour pair model", "J_eV": J, "temperature": temperature},
    )
