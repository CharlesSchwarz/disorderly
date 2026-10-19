"""Reference settings file for ``run-monte-carlo``, listing every option.

Copy this file, edit it, and run with:

    run-monte-carlo parent.cif initial.cif 1000 path/to/this_file.py

The CLI only reads the parent structure path, initial structure path,
temperature, and this settings path. Edit the values below to control the run.
"""

from __future__ import annotations

from disorderly.mace import mace_policy


num_steps = 50000  # Number of Monte Carlo moves to propose.
num_structures_to_save = 10  # Lowest-energy configurations to keep (from all evaluated candidates, accepted or not).
time_limit = None  # Optional wall-clock limit in seconds; the run stops early and saves what it has.
seed = 0  # Random seed for reproducible moves; set to None for non-deterministic runs.
max_move_attempts = 1000  # Maximum redraws per step; only used with disallowed_moves = "retry".
# "reject" counts a disallowed proposal as a rejected step, preserving detailed
# balance; "retry" redraws it, which biases sampling (see run_monte_carlo).
disallowed_moves = "reject"
record_structure_every = None  # Also store the current configuration every N steps; None stores only step 0.
move_appliers = None  # Optional {move_type: function} for custom move types; None uses the built-in "swap" and "move".
save_best_structures = True  # If True, write the saved lowest-energy configurations as CIFs to best_structures/.
progress_updates = 4  # Print progress this many times per run (4: at 25%, 50%, 75%); 0 prints none.

# Leave as None to save next to the parent CIF as
# "<cif_stem>_T<temperature>K_monte_carlo_results".
# Relative custom paths are resolved relative to this settings file.
output_dir = None

# MACE energy settings. Set model_path to your trained MACE model file.
model_path = "/path/to/mace.model"  # Path to the trained MACE model file.
device = "cpu"  # MACE device; use "cpu" or a GPU device such as "cuda".
relax = True  # If True, relax each configuration before reading its MACE energy.
relax_cell = True  # If relax is True, relax both positions and cell; False relaxes positions only.
fmax = 0.05  # Force convergence threshold for the LBFGS relaxation, in eV/Angstrom.
max_steps = 1000  # Maximum LBFGS optimization steps per relaxation.
logfile = None  # ASE optimizer log file; None writes no log.
logm_warning_tol = 1e-6  # Suppress SciPy logm warnings at or below this approximate error; 0 shows all.


# The policy can be given either as a module-level ``policy = MonteCarloPolicy(...)``
# or, as here, by a ``get_policy`` function. Either way, the CLI adds
# "disordered_structure", "initial_structure" and "temperature" to the
# policy's context.
def get_policy(disordered_structure, initial_structure, temperature):
    """Return the Monte Carlo policy for this run."""
    return mace_policy(
        model_path=model_path,
        device=device,
        relax=relax,
        relax_cell=relax_cell,
        fmax=fmax,
        max_steps=max_steps,
        logfile=logfile,
        logm_warning_tol=logm_warning_tol,
        context={
            "disordered_structure": disordered_structure,
            "initial_structure_formula": initial_structure.composition.reduced_formula,
            "temperature": temperature,
        },
    )
