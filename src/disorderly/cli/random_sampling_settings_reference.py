"""Reference settings file for ``run-random-sampling``, listing every option.

Copy this file, edit it, and run with:

    run-random-sampling parent.cif path/to/this_file.py

The CLI only reads the structure path and this settings path. Edit the values
below to control sampling behavior.
"""

from __future__ import annotations

from disorderly.mace import MACEEnergyEvaluator


n_attempts = 10  # Maximum number of random ordered configurations to generate and score.
n_return = 5  # Maximum number of lowest-energy, symmetrically distinct configurations to return.
time_limit = None  # Optional wall-clock limit in seconds; stops between attempts and saves the best found so far.
seed = 0  # Random seed for reproducible ordering generation; set to None for non-deterministic runs.
occupancy_tol = 1e-2  # Occupancy tolerance for converting partial occupancies into whole-site orderings.
candidate_pool_size = None  # Low-energy candidates kept before removing symmetric duplicates; None uses max(5 * n_return, 50).
symmetry_match_kwargs = None  # Optional StructureMatcher kwargs for duplicate removal, e.g. {"ltol": 0.2, "stol": 0.3}.
n_jobs = 1  # Number of worker threads for energy evaluations; 1 runs serially.
save_sites_in_json = False  # False stores compact ordering keys only; True embeds full site data in the JSON.
verbose = False  # If True, print every attempted configuration's energy as it is evaluated.

# Leave as None to save next to the CIF as "<cif_stem>_random_sampling_results".
# Relative custom paths are resolved relative to this settings file.
output_dir = None

# MACE energy settings. Set model_path to your trained MACE model file.
model_path = "/path/to/mace.model"  # Path to the trained MACE model file.
device = "cpu"  # MACE device; use "cpu" or a GPU device such as "cuda".
relax = True  # If True, relax each configuration before reading its MACE energy.
relax_cell = True  # If relax is True, relax both positions and cell; False relaxes positions only.
fmax = 0.05  # Force convergence threshold for the LBFGS relaxation, in eV/Angstrom.
max_steps = 1000  # Maximum LBFGS optimization steps per relaxation.
logfile = None  # ASE optimizer log file; keep None with n_jobs > 1, since workers sharing one log interleave.
logm_warning_tol = 1e-6  # Suppress SciPy logm warnings at or below this approximate error; 0 shows all.

energy_evaluator = MACEEnergyEvaluator(
    model_path=model_path,
    device=device,
    relax=relax,
    relax_cell=relax_cell,
    fmax=fmax,
    max_steps=max_steps,
    logfile=logfile,
    logm_warning_tol=logm_warning_tol,
)
