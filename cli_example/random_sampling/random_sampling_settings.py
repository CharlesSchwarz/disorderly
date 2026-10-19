"""Editable settings for ``run-random-sampling``.

Run with:

    run-random-sampling parent.cif path/to/this_file.py

The CLI only reads the structure path and this settings path. Edit the values
below to control sampling behavior.
"""

from __future__ import annotations


n_attempts = 10  # Maximum number of random ordered configurations to generate and score.
n_return = 5  # Maximum number of lowest-energy, symmetrically distinct configurations to return.
time_limit = None  # Optional wall-clock limit in seconds; stops between attempts and saves best found so far.
seed = 0  # Random seed for reproducible ordering generation; set to None for non-deterministic runs.
occupancy_tol = 1e-2  # Occupancy tolerance for converting partial occupancies into whole-site orderings.
candidate_pool_size = None  # Low-energy candidates retained before deduplication; None uses max(5 * n_return, 50).
symmetry_match_kwargs = None  # Optional symmetry matching kwargs, e.g. {"ltol": 0.2, "stol": 0.3}.
n_jobs = 4  # Number of worker threads for energy evaluations; 1 runs serially.
save_sites_in_json = False  # False stores compact ordering keys only; True embeds full site data in JSON.
verbose = True  # If True, print every attempted configuration's energy as it is evaluated.
output_dir = None  # None saves next to the CIF as "<cif_stem>_random_sampling_results"; relative paths use this file.

############ENERGY_EVALUATOR##############
'''
energy_evaluator is a callable that takes a structure and returns a float energy. It can be defined in one of three ways:
# 1. Define a callable directly:
# energy_evaluator = lambda structure: 0.0  # Replace with your own energy
# 2. Define a function that returns a callable:
# def get_energy_evaluator():
#     def evaluator(structure):
#         return 0.0  # Replace with your own energy
#     return evaluator
# 3. Use a pre-defined energy evaluator class, such as MACEEnergyEvaluator:
'''

## Example Ewald energy evaluator. This uses pymatgen's EwaldSummation to compute the electrostatic energy of a structure.
#from pymatgen.analysis.ewald import EwaldSummation
#energy_evaluator = lambda structure: EwaldSummation(structure).total_energy  # Replace with your own energy


## Example MACE energy evaluator settings. Set model_path to your trained MACE model file.
from disorder_mc.mace import MACEEnergyEvaluator
energy_evaluator = MACEEnergyEvaluator(  
    model_path="/path/to/mace.model",  # Path to the trained MACE model file.
    device="cpu",  # MACE device; use "cpu" or a GPU device such as "cuda".
    relax=True,  # If True, relax each sampled structure before reading its MACE energy.
    relax_cell=True,  # If relax is True, relax both positions and cell; False relaxes positions only.
    fmax=0.5,  # Force convergence threshold for ASE/LBFGS relaxation, in eV/Angstrom.
    max_steps=1000,  # Maximum ASE/LBFGS optimization steps per sampled configuration.
    logfile=None,  # ASE optimizer log file; keep None for parallel runs because workers sharing one log can interleave.
    logm_warning_tol=1e-6,  # Suppress SciPy logm warnings at or below this approximate error.
)
