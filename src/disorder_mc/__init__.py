"""disorder_mc: tools for exploring ordered configurations of disordered
crystal structures, built on pymatgen.

The public API re-exported here is intentionally curated; internal
helpers (e.g. ``disorder_mc.symmetry._site_label``) are not part of the
supported interface and may change without notice.
"""

__version__ = "0.1.0"

from disorder_mc.analysis import (
    ThermalAverage,
    plot_trajectory,
    property_series,
    rolling_acceptance_rate,
    thermal_average,
)
from disorder_mc.diagnostics import ProposalSymmetryReport, check_proposal_symmetry
from disorder_mc.enumeration import enumerate_ordered_configurations
from disorder_mc.energy import PrecomputedEwald, disordered_ewald_energy, effective_site_charges
from disorder_mc.ordering import ordering_key, structure_from_ordering_key
from disorder_mc.io import (
    load_checkpoint,
    load_configurations,
    load_energy_evaluator,
    load_policy,
    load_trajectory,
    save_checkpoint,
    save_configurations,
    save_json,
    save_structures,
    save_trajectory,
)
from disorder_mc.monte_carlo import (
    KB_EV_PER_K,
    move_ion,
    propose_next_configuration,
    resume_monte_carlo,
    run_monte_carlo,
    swap_ion,
)
from disorder_mc.chgnet import CHGNetEnergyEvaluator, chgnet_policy
from disorder_mc.mace import MACEEnergyEvaluator, get_energy_evaluator, mace_policy
from disorder_mc.mlip import ASECalculatorEvaluator, mlip_policy
from disorder_mc.results import MonteCarloCheckpoint, MonteCarloStep, MonteCarloTrajectory, SavedConfiguration
from disorder_mc.sampling import random_energy_sampling, random_ewald_sampling, random_ordered_configuration
from disorder_mc.settings import (
    DetailedBalanceWarning,
    MonteCarloPolicy,
    MoveProposal,
    default_policy,
    preserves_composition,
    sublattice_swap_policy,
)
from disorder_mc.structures import (
    DisorderedSiteGroup,
    count_configurations,
    disordered_site_groups,
    disordered_site_indices,
    is_disordered,
    replace_species_and_occupancy,
    set_site_occupancies,
    supercell_multiplier,
)
from disorder_mc.symmetry import (
    configuration_degeneracy,
    count_distinct_configurations,
    describe_wyckoff_sites,
    get_spacegroup,
    get_wyckoff_labels,
    get_wyckoff_sites,
    label_wyckoff_sites,
    site_permutations,
    symmetrically_equivalent,
)

__all__ = [
    "__version__",
    # structures
    "replace_species_and_occupancy",
    "set_site_occupancies",
    "is_disordered",
    "disordered_site_indices",
    "DisorderedSiteGroup",
    "disordered_site_groups",
    "supercell_multiplier",
    "count_configurations",
    # symmetry
    "get_spacegroup",
    "get_wyckoff_sites",
    "get_wyckoff_labels",
    "label_wyckoff_sites",
    "describe_wyckoff_sites",
    "symmetrically_equivalent",
    "configuration_degeneracy",
    "count_distinct_configurations",
    "site_permutations",
    # energy
    "disordered_ewald_energy",
    "effective_site_charges",
    "PrecomputedEwald",
    # ordering keys
    "ordering_key",
    "structure_from_ordering_key",
    # monte carlo
    "KB_EV_PER_K",
    "move_ion",
    "swap_ion",
    "propose_next_configuration",
    "run_monte_carlo",
    "resume_monte_carlo",
    # diagnostics
    "check_proposal_symmetry",
    "ProposalSymmetryReport",
    # settings / policy
    "DetailedBalanceWarning",
    "MonteCarloPolicy",
    "MoveProposal",
    "default_policy",
    "preserves_composition",
    "sublattice_swap_policy",
    # results
    "MonteCarloStep",
    "MonteCarloTrajectory",
    "SavedConfiguration",
    "MonteCarloCheckpoint",
    # analysis
    "property_series",
    "rolling_acceptance_rate",
    "plot_trajectory",
    "thermal_average",
    "ThermalAverage",
    # sampling
    "random_ordered_configuration",
    "random_ewald_sampling",
    # io
    "save_structures",
    "save_trajectory",
    "load_trajectory",
    "save_checkpoint",
    "load_checkpoint",
    "save_configurations",
    "load_configurations",
    "save_json",
    "load_energy_evaluator",
    "load_policy",
    # machine-learned potentials (ASE calculators)
    "ASECalculatorEvaluator",
    "mlip_policy",
    "MACEEnergyEvaluator",
    "get_energy_evaluator",
    "mace_policy",
    "CHGNetEnergyEvaluator",
    "chgnet_policy",
    # generic sampling
    "random_energy_sampling",
    # enumeration (icet)
    "enumerate_ordered_configurations",
]
