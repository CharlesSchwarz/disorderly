# Design notes

This document records the scientific conventions and design decisions in
`disorder_mc`: what the package assumes, how its definitions are chosen, and
where those choices have consequences for results. Function docstrings give
the full parameter details.

## Terminology

- **Parent structure**: a disordered `pymatgen.core.Structure`, in which one or
  more sites have partial occupancy or several possible species.
- **Configuration / ordered structure**: one fully ordered realization of a
  parent. Each site holds exactly one species.
- **Monte Carlo move**: a transformation from one configuration to another, such
  as swapping the species on two sites or moving a species to a vacant site.
- **Trajectory**: the sequence of configurations a Monte Carlo run evaluates,
  with their energies, properties, moves, and accept/reject decisions.

## Configurations and vacancies

A configuration is an ordinary `pymatgen` `Structure` on the parent's lattice.
Vacancies are represented by **absence**: when a parent site's total occupancy
is below 1 and the configuration leaves it empty, that site is simply missing
from the configuration. There is no explicit vacancy species. A configuration
can therefore have fewer sites than its parent.

Functions that need to relate a configuration back to its parent
(`configuration_degeneracy`, `ordering_key`, `sublattice_swap_policy`,
`PrecomputedEwald`) match configuration sites to parent sites by position,
within a distance tolerance. Configurations must stay on the parent's lattice
for these functions to work; relaxed structures should be mapped back first.

Input structures are never modified in place.

## Composition of ordered configurations

`disordered_site_groups` defines how many atoms of each species an ordered
configuration places on the disordered sites:

- Disordered sites are pooled by their exact species/occupancy signature (for
  example, all sites with `{Li+: 0.5, Fe3+: 0.5}`). Sites in a pool are treated
  as interchangeable. Physically distinct sublattices that happen to share a
  signature are pooled together; split them first (for example with
  `replace_species_and_occupancy`) if that is not wanted.
- Each pool's occupancies are converted to whole numbers of sites. If the
  occupancies sum to 1 (within `occupancy_tol`), every site is filled and the
  last species absorbs any rounding remainder. Otherwise each species' count
  is rounded independently and the remaining sites are vacant.
- If occupancies are not whole numbers of sites in the given cell, a warning
  reports how many times the cell must be multiplied (`supercell_multiplier`).
  Results in a cell that is too small have the wrong composition.

These counts are fixed for the whole workflow: random sampling, exhaustive
enumeration, and the swap-based Monte Carlo policies all preserve them. The
package samples the canonical ensemble (fixed composition per pool); it does
not change the overall composition.

## Oxidation states and electrostatic energy

Oxidation states are **never guessed**. Ewald energies require every site to
carry an explicit oxidation state (for example via
`Structure.add_oxidation_state_by_element`) or site charge; otherwise a
`ValueError` explains how to add them.

For a disordered site, `disordered_ewald_energy` uses the occupancy-weighted
effective charge

    q_site = sum_i occupancy_i * charge_i

which is what pymatgen's `EwaldSummation` computes internally, so the function
is a validated wrapper around it rather than a new Ewald implementation.
`effective_site_charges` returns the per-site values. This mean-field
approximation captures the average electrostatics of the disordered site but
not correlations within it. For an ordered configuration it reduces to the
ordinary Ewald energy.

`PrecomputedEwald` gives the same energies for configurations on a fixed set
of positions much faster: the Ewald energy of point charges on fixed positions
is a quadratic form in the charges, so the matrix is computed once and each
evaluation is a matrix-vector product. Vacant positions carry zero charge.

**Units**: all energies in the package are in eV. Ewald energies are total
energies of the cell, including the neutralizing-background term for charged
cells.

## Symmetry

The package uses two distinct notions of equivalence, for different purposes.

### Symmetric equivalence of two structures

`symmetrically_equivalent(s1, s2)` returns true when pymatgen's
`StructureMatcher.fit` finds a mapping between the two periodic structures.
Species are compared exactly, including oxidation states. By default
structures are reduced to primitive cells and compared with pymatgen's
default tolerances (`ltol=0.2`, `stol=0.3`, `angle_tol=5`). This is a direct
comparison of two periodic structures and does not refer to any parent.

### Equivalence and degeneracy under the parent's symmetry

`site_permutations(parent)` converts each symmetry operation of the parent
(found by spglib, `symprec=1e-3` Å by default) into a permutation of the
parent's sites. Two configurations of the same parent cell are equivalent if
one of these permutations maps one onto the other.

- `configuration_degeneracy(parent, ordered)` is the size of the orbit of
  `ordered` under these permutations: the number of distinct site decorations
  related to it by the parent's symmetry. Equivalently,
  `|G| / |stabilizer of ordered|`. This is the multiplicity `g_i` in
  `Z = sum_i g_i exp(-E_i / kT)` over symmetrically distinct configurations.
- `count_distinct_configurations(parent)` counts the orbits exactly with
  Burnside's lemma, without listing the configurations. The degeneracies of
  all distinct configurations sum to `count_configurations(parent)`.

Both are with respect to the parent's symmetry in the given cell only;
configurations of different supercells are not compared.

The two notions agree in one direction: configurations in the same orbit
under the parent's symmetry always match with `StructureMatcher`. The
converse need not hold: `StructureMatcher` can match two configurations
through an operation that is not a symmetry of the specific parent supplied
(for example, when the parent was given with lower symmetry than its true
average structure), or because they agree within its tolerances.

### Tolerances

Space groups and Wyckoff positions come from spglib through pymatgen
(`symprec=1e-3` Å and `angle_tolerance=5°` by default). spglib treats a
disordered site as occupied by all of its species, so a disordered
structure's space group is the symmetry of its average (parent) structure.

## Random sampling

`random_energy_sampling(parent, energy_evaluator, ...)` (and the Ewald
version, `random_ewald_sampling`):

1. generates up to `n_attempts` random configurations with the compositions
   above;
2. evaluates their energies;
3. keeps a bounded, energy-sorted pool of the lowest-energy candidates
   (`candidate_pool_size`, default `max(5 * n_return, 50)`) rather than every
   attempt, so memory does not grow with `n_attempts`;
4. removes symmetry duplicates within that pool with `symmetrically_equivalent`;
5. returns up to `n_return` distinct configurations, lowest energy first.

Because only the pool is deduplicated, if many low-energy attempts are
duplicates of one another, fewer than `n_return` distinct configurations can
be returned even though others were generated. Increase
`candidate_pool_size` if that matters.

The energy evaluator may return a float, or a dict with an `"energy"` key plus
additional properties to store with each configuration.

## Exhaustive enumeration

`enumerate_ordered_configurations(parent, sizes=1)` lists every symmetrically
distinct configuration using icet's implementation of the Hart–Forcade
algorithm (optional `icet` extra). Each (pool, species) combination is given
its own placeholder element with an exact concentration restriction, so the
compositions are exactly those defined above, and the result is mapped back to
the parent's species with oxidation states kept.

- For `sizes=1`, the result has exactly `count_distinct_configurations(parent)`
  members, and the cells share the parent lattice so they work with
  `ordering_key` and `configuration_degeneracy`.
- For `sizes=n > 1`, icet returns only configurations that do not repeat in a
  smaller cell, over all inequivalent supercell shapes of that size. To list
  every ordering of one particular supercell, build that supercell and use
  `sizes=1`. Sizes at which occupancies are not whole numbers of sites are
  skipped.

## Monte Carlo

### Policies

The engine (`run_monte_carlo`) contains no system-specific rules. All
problem-specific behavior is supplied by a `MonteCarloPolicy`, which bundles:

- `choose_potential_move(configuration, rng, context)`: returns a
  `MoveProposal` (move type, site indices, optional target coordinates and
  metadata) or `None`;
- `is_move_allowed(configuration, candidate, move, context)`: rejects
  candidates that violate problem-specific constraints;
- `evaluate_props(configuration, context)`: returns a dict of properties;
- `context`: a free-form dict for problem-specific state, such as a parent
  lattice or the list of mobile species.

A proposal is turned into a candidate structure by a move applier chosen by its
`move_type`. `"swap"` and `"move"` are built in; new move types can be added by
passing `move_appliers` without changing the engine.

Ready-made policies:

- `sublattice_swap_policy(parent)` swaps species only within groups of parent
  sites (by default, the disordered pools defined above). This is the usual
  choice for substitutional disorder.
- `default_policy()` swaps any two sites holding different species, which
  includes swapping anions with cations. It is a generic baseline and a
  template, not a physically motivated move set for most ionic systems.
- `mlip_policy`, `mace_policy`, and `chgnet_policy` score swaps with a
  machine-learned potential.

Problems with a partially vacant sublattice (ion hopping) need a custom
`choose_potential_move` that also proposes `"move"` moves to vacant parent
positions; `cli_example/monte_carlo/monte_carlo_settings.py` (Li hops in LLZO) is an
example.

### Energy and acceptance

The scalar energy used for acceptance is **`evaluate_props(...)["energy"]`**,
in eV; it is the one required key of the properties dict. A missing key raises
an error.

Each step:

1. One proposal is drawn and applied. If the policy proposes no move or
   `is_move_allowed` rejects it, the step is a rejected step: the chain stays
   put, the candidate is not evaluated, and the step is recorded with
   `allowed=False` (and counts as rejected in `acceptance_rate`).
2. Otherwise the candidate's properties are evaluated.
3. The candidate is accepted with the Metropolis--Hastings probability
   `min(1, exp(-ΔE / (k_B T) + r))`, where `r` is the move's
   `log_proposal_ratio` (0 for symmetric proposals) and
   `k_B = 8.617333262e-5` eV/K. At `T <= 0`, only moves with `ΔE <= 0` are
   accepted and `r` is ignored.

With `disallowed_moves="retry"`, step 1 instead redraws disallowed proposals,
up to `max_move_attempts` (default 1000) times, after which a `RuntimeError`
is raised. This was the only behavior before the `disallowed_moves` option was
added.

### Detailed balance

The run samples the Boltzmann distribution (restricted to allowed
configurations) only if, for every move `a -> b`, the reverse move can be
proposed and `r = ln[g(b -> a) / g(a -> b)]`, where `g` is the probability
that `choose_potential_move` proposes the move. For symmetric proposals `r = 0`.
The built-in swap policies are symmetric: a swap leaves every species count
unchanged, so it and its reverse are equally likely to be proposed.

Common ways a custom policy breaks this:

- choosing a target uniformly among the moves available *now*, such as the
  vacant neighbors of an ion, when that number differs between `a` and `b`.
  Report `r = ln(n_a / n_b)` or choose from a fixed set instead;
- choosing between move types with probabilities that depend on the
  configuration;
- `disallowed_moves="retry"`, which reweights each configuration by its
  fraction of allowed proposals.

A policy that breaks detailed balance still gives exact energies and can
still find low-energy configurations, but averages over its trajectory (site
occupancies, order parameters, `<E>`, transition temperatures) are biased.
The bias does not shrink with longer runs.

Policies may break it deliberately; the engine warns rather than refuses.
`MonteCarloPolicy.detailed_balance` declares whether the policy satisfies it,
and the built-in policies declare `True`. At `T > 0` the engine emits a
`DetailedBalanceWarning` if the declaration is `False` or `None` (undeclared),
and the first time a `"retry"` run redraws a proposal. The declaration and the
`disallowed_moves` mode are recorded in the trajectory parameters.

The engine cannot verify a declaration itself.
`check_proposal_symmetry(policy, configuration)` checks one empirically: it
counts how often moves out of `configuration` are proposed and how often
their reverses are proposed from the resulting candidates, and compares the
ratio with each move's `log_proposal_ratio`. It evaluates no energies. Use a
small cell, where each move is proposed often enough to measure.

### Saved configurations

Every evaluated candidate, accepted or rejected, is considered for the
`num_structures_to_save` lowest-energy configurations kept with the
trajectory. A low-energy configuration is kept even if the chain did not move
to it.

### Reproducibility and checkpointing

All randomness comes from one `numpy.random.Generator` created from the `seed`
argument and recorded in the trajectory. Policies receive this generator and
must draw any randomness from it. Given the same seed, a deterministic
`evaluate_props`, and a policy that only uses the provided generator, a run is
reproducible step for step.

`run_monte_carlo(..., on_step=callback)` calls `callback(step, checkpoint)`
after each step with a `MonteCarloCheckpoint`: the current configuration and
its properties, the generator state, and the saved configurations.
`resume_monte_carlo(checkpoint, ...)` continues from it without re-evaluating
the checkpointed configuration, and reproduces the uninterrupted run under
the same conditions. `num_steps` is the total for the whole run, step numbers
continue from the checkpoint, and the returned trajectory covers only the
resumed segment. `save_checkpoint` writes atomically, so a job killed while
writing keeps its previous checkpoint. This lets long runs (for example with
relaxed machine-learned-potential energies on a cluster) survive wall-time
limits.

### Trajectory analysis

`MonteCarloStep.properties` describes the *proposed* configuration at a step,
while `MonteCarloStep.energy` is the energy of the *current* configuration.
The analysis functions in `disorder_mc.analysis` always report the current
configuration: `property_series` carries each property forward from the most
recent accepted step. `plot_trajectory` plots the energy, optionally the
acceptance rate over a trailing window of moves, and the properties named in
`properties` (none by default) against step, with optional `height_ratios`
(matplotlib is optional, via the `plot` extra).

`thermal_average` is the plain mean of a property over the current
configuration at every step from `equilibration_steps` on, with rejected steps
counting the current configuration again. No Boltzmann reweighting is applied:
a Metropolis chain with detailed balance already samples the Boltzmann
distribution. It refuses steps run at more than one temperature and warns
(`DetailedBalanceWarning`) above 0 K unless the trajectory records
`detailed_balance=True`. Its `standard_error` is a batch-means estimate
(standard deviation of `n_blocks` contiguous block means over
`sqrt(n_blocks)`). This is valid only when blocks are much longer than the
autocorrelation time, and is otherwise an underestimate. `std` is the thermal
fluctuation, not the uncertainty of the mean.

## Machine-learned potentials

`ASECalculatorEvaluator` wraps any ASE calculator through a zero-argument
`calculator_factory`. It converts structures to ASE atoms (without modifying
the input), optionally relaxes them with LBFGS (positions only, or positions
and cell with `FrechetCellFilter`), reports `relax_converged` and
`relax_steps`, creates one calculator per thread for parallel sampling, and
returns the energy as `{"energy": ..., "<label>_energy": ...}`.

`disorder_mc.mace` and `disorder_mc.chgnet` only construct the calculator.
They are separate optional extras because their dependencies may conflict, and
both are imported only when used. Energies are total energies in eV. Neither
backend reads oxidation states from the structure. Other potentials (for
example M3GNet through `matgl`) can be used through `ASECalculatorEvaluator`
with a custom factory.

## Ordering keys

`ordering_key(parent, ordered)` returns a short, reversible code (prefix
`ok1_`) for a configuration relative to its parent. It encodes the species
choice, or vacancy, on each disordered parent site; the parent supplies the
lattice, positions, ordered sites, and species list needed by
`structure_from_ordering_key(parent, key)` to rebuild the configuration. It is
not a hash: it is only meaningful together with the same parent structure, and
it does not identify symmetry-equivalent configurations with each other.
Vacancies are encoded only for sites whose total occupancy is below 1.

Random sampling records the key on each returned configuration, so saved JSON
results can be compact and still reconstructible.

## Settings files and input/output

A Monte Carlo policy is a set of Python functions, which have no data
serialization format. The command-line workflows (`run-monte-carlo`,
`run-random-sampling`) therefore take a **Python settings file** by path:
module-level variables set the run controls, and a `policy` object or a
`get_policy(...)` function (Monte Carlo) or an `energy_evaluator` (random
sampling) provides the behavior. `run-monte-carlo` adds the parent structure,
initial structure and temperature to the policy's `context` (as
`"disordered_structure"`, `"initial_structure"` and `"temperature"`, without
overwriting keys the policy sets), so a module-level `policy` can use the
parent given on the command line. Commented reference settings files are in
`src/disorder_mc/cli/` (`*_settings_reference.py`). `io.load_policy` provides the same pattern for
an importable module by dotted name.

Trajectories, checkpoints, and saved configurations are written as JSON, using
pymatgen's `Structure.as_dict()` for structures; structures are also written as
CIF files where useful. Each JSON format has a matching loader, with round-trip
tests.
