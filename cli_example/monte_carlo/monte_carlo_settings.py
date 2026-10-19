"""Editable settings for ``run-monte-carlo``: Li hops in cubic LLZO.

Run with (from this directory):

    run-monte-carlo cubic_LLZO_parent.cif cLLZO_init_all_tet_sites_occupied.cif 1200 monte_carlo_settings.py

``cubic_LLZO_parent.cif`` holds 56 Li on 72 partially occupied sites: 24
tetrahedral (24d) and 48 octahedral (48g). Ordered configurations list only the
56 occupied sites. Each move hops one Li to a vacant Li site, tetrahedral or
octahedral, so the tetrahedral/octahedral split can change; La, Zr and O never
move. The number of tetrahedral Li is recorded as ``n_tet`` in the trajectory.

The CLI only reads the parent structure path, initial structure path,
temperature, and this settings path. Edit the values below to control the run.
"""

from __future__ import annotations

import numpy as np

from disorder_mc import MonteCarloPolicy, MoveProposal, disordered_ewald_energy
from disorder_mc.settings import default_is_move_allowed


## Note: the run temperature is not included in the settings file, it is passed directly as a parameter.

num_steps = 10000  # Number of moves to propose; about 0.35 s each with CHGNet on a laptop CPU.
num_structures_to_save = 10  # Lowest-energy configurations saved to best_structures/.
time_limit = None  # Optional wall-clock limit in seconds; the run stops and saves what it has.
seed = 0  # Random seed for reproducible moves; set to None for non-deterministic runs.
max_move_attempts = 1000  # only used with disallowed_moves = "retry"
# "reject" counts a disallowed proposal as a rejected step, preserving detailed
# balance; "retry" redraws it, which biases sampling (see run_monte_carlo).
disallowed_moves = "reject"
record_structure_every = None  # Also store the current configuration every N steps; None stores none.
move_appliers = None
save_best_structures = True
progress_updates = 4  # Print progress this many times per run (4: at 25%, 50%, 75%); 0 prints none.

# Leave as None to save next to the parent CIF as
# "<cif_stem>_T<temperature>K_monte_carlo_results".
# Relative custom paths are resolved relative to this settings file.
output_dir = None

# Li move settings.
site_tol = 0.5  # Angstrom; a Li site counts as occupied if a Li lies within this distance.

############ Energy Evaluator ##############
## energy_evaluator takes an ordered structure and returns its energy in eV (a float), or a dict with an "energy" key plus any other properties to record.

## CHGNet energy of the unrelaxed configuration (Li on the ideal parent site positions). Needs pip install -e ".[chgnet]"; the pretrained model downloads on first use.
from disorder_mc.chgnet import CHGNetEnergyEvaluator
energy_evaluator = CHGNetEnergyEvaluator(
    model_name="0.3.0",  # Pretrained CHGNet model; or set model_path to a fine-tuned model file.
    device="cpu",  # CHGNet device; use "cpu" or a GPU device such as "cuda".
    relax=False,  # If True, relax each configuration before reading its energy (much slower).
)

## Ewald energy: faster and needs no extra dependencies, but it treats ions as point charges, so energy differences are exaggerated and temperatures of several thousand K are needed for moves to be accepted.
# energy_evaluator = disordered_ewald_energy


############ MC Policy ##############
# The CLI adds the parent structure, initial structure and temperature to policy.context as "disordered_structure", "initial_structure" and "temperature".

def choose_li_hop(configuration, rng, context):
    """Hop a random Li to a random vacant Li site of the parent.

    Every Li and every vacant site is equally likely, so a hop and its reverse
    are proposed with the same probability and plain Metropolis acceptance
    satisfies detailed balance.
    """
    parent = context["disordered_structure"]
    sites = np.array([site.frac_coords for site in parent if "Li" in site.species.get_el_amt_dict()])
    li = configuration.indices_from_symbol("Li")
    vacant = np.flatnonzero(configuration.lattice.get_all_distances(sites, configuration.frac_coords[list(li)]).min(axis=1) > site_tol)
    if not len(vacant):
        return None
    site_index = li[int(rng.integers(len(li)))]
    target = int(vacant[rng.integers(len(vacant))])
    return MoveProposal(move_type="move", site_index=site_index, target_frac_coords=tuple(sites[target]))


def evaluate_props(configuration, context):
    """Energy of ``configuration`` plus its number of tetrahedral Li (``n_tet``):
    Li with 4 O within 2.5 A (octahedral Li have 6)."""
    if hasattr(energy_evaluator, "evaluate_props"):  # e.g. CHGNetEnergyEvaluator: also records chgnet_energy
        props = dict(energy_evaluator.evaluate_props(configuration))
    else:
        props = {"energy": float(energy_evaluator(configuration))}
    li = list(configuration.indices_from_symbol("Li"))
    oxygen = list(configuration.indices_from_symbol("O"))
    n_oxygen = (configuration.lattice.get_all_distances(configuration.frac_coords[li], configuration.frac_coords[oxygen]) < 2.5).sum(axis=1)
    props["n_tet"] = int((n_oxygen == 4).sum())
    return props


policy = MonteCarloPolicy(
    choose_potential_move=choose_li_hop,
    is_move_allowed=default_is_move_allowed,
    evaluate_props=evaluate_props,
    # choose_li_hop proposes a hop and its reverse with equal probability.
    detailed_balance=True,
)
