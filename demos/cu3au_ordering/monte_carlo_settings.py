"""Settings for ``run-monte-carlo``: L1_2 ordering in Cu3Au with MACE.

Run with (from the repository root):

    run-monte-carlo demos/cu3au_ordering/parent.cif \\
        demos/cu3au_ordering/initial_structures/ordered.cif 600 \\
        demos/cu3au_ordering/monte_carlo_settings.py

Cu3Au is fcc. Below about 663 K it orders into L1_2: Au on the cube corners,
Cu on the face centres. Above it, Cu and Au mix randomly over all sites.

Moves are the default composition-preserving swaps of a Cu and an Au atom,
scored by MACE energy. Each step records the energy and the L1_2 order
parameter ``order`` (1 = perfect L1_2, ~0 = random) in ``trajectory.json``.

Same as the reference settings file ``src/disorder_mc/cli/monte_carlo_settings_reference.py``, with
the order parameter added to the recorded properties.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from disorder_mc import MonteCarloPolicy
from disorder_mc.mace import MACEEnergyEvaluator
from disorder_mc.settings import default_choose_potential_move


num_steps = 10000
num_structures_to_save = 10
time_limit = None
seed = 0
max_move_attempts = 1000
record_structure_every = None
move_appliers = None
save_best_structures = True

# One results folder per run, named from the initial structure and temperature
# on the command line, e.g. results/raw/ordered_T600K. Relative custom paths
# are resolved relative to this settings file.
output_dir = f"results/raw/{Path(sys.argv[2]).stem}_T{float(sys.argv[3]):g}K" if len(sys.argv) > 3 else None

# MACE energy settings. Set model_path to your trained MACE model file.
# Au is much larger than Cu, so relaxation matters: one Cu/Au swap in perfect
# L1_2 costs 0.78 eV on the ideal lattice but 0.34 eV with atoms relaxed
# (0.343 eV with the cell relaxed too, at twice the cost). relax_cell = False
# keeps the cell fixed at the parent's lattice parameter.
model_path = "/path/to/mace.model"
device = "cpu"
relax = True
relax_cell = False
fmax = 0.02
max_steps = 1000
logfile = None
logm_warning_tol = 1e-6


def fcc_sublattices(structure):
    """Which of the four simple-cubic sublattices of fcc each site is on
    (0 = cube corners, 1-3 = face centres).

    ``structure`` is an L x L x L supercell of the 4-atom fcc cube, so a
    site's fractional coordinates times 2L are integers whose parities
    (even, even, even) mark a corner and (odd, odd, even) etc. a face centre.
    """
    n_cells = round((len(structure) / 4) ** (1 / 3))
    parity = np.round(structure.frac_coords * 2 * n_cells).astype(int) % 2
    return parity[:, 0] * 2 + parity[:, 1]  # (0,0) corner, (1,1), (1,0), (0,1) faces


def l12_order(configuration, sublattice):
    """(f - x) / (1 - x): f is the Au fraction of the sublattice holding the most
    Au, x the overall Au fraction. 1 for perfect L1_2, ~0 for a random alloy."""
    is_au = np.array([site.specie.symbol == "Au" for site in configuration])
    x = is_au.mean()
    f = max(is_au[sublattice == k].mean() for k in range(4))
    return float((f - x) / (1 - x))


def get_policy(disordered_structure, initial_structure, temperature):
    """Return the Monte Carlo policy for this run."""
    sublattice = fcc_sublattices(initial_structure)
    evaluator = MACEEnergyEvaluator(
        model_path=model_path,
        device=device,
        relax=relax,
        relax_cell=relax_cell,
        fmax=fmax,
        max_steps=max_steps,
        logfile=logfile,
        logm_warning_tol=logm_warning_tol,
    )

    def evaluate_props(configuration, context):
        props = evaluator.evaluate_props(configuration)
        props["order"] = l12_order(configuration, sublattice)
        return props

    return MonteCarloPolicy(
        choose_potential_move=default_choose_potential_move,
        evaluate_props=evaluate_props,
        context={"energy_model": "MACE", "mace_model_path": str(model_path), "temperature": temperature},
    )
