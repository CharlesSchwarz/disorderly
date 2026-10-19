<p align="center">
  <img src="docs/images/logo.png" alt="Disorderly logo" width="480">
</p>

# disorder_mc

`disorder_mc` takes a disordered crystal structure and finds its low-energy ordered configurations. It can sample
or exhaustively enumerate the symmetrically distinct orderings, or explore
them at finite temperature with Metropolis Monte Carlo. Each configuration is
scored directly by the energy model you choose: Ewald electrostatics, a
machine-learned interatomic potential such as MACE or CHGNet (optionally after
relaxation), any other ASE calculator, or your own function.

Built on [pymatgen](https://pymatgen.org/). The API is designed to be easy to read and to call from
scripts and LLM-based workflows. Features:

- **Symmetry tools**: space groups and Wyckoff sites, counts of total and
  symmetrically distinct configurations, configuration degeneracies, and
  removal of symmetry-equivalent duplicates.
- **Energy-ranked sampling**: random sampling or exhaustive enumeration of
  ordered configurations, returning the lowest-energy distinct ones.
- **Monte Carlo**: a Metropolis engine with pluggable moves and constraints
  (site swaps, ion hops into vacancies, or your own), reproducible seeding,
  and checkpoints for resuming long runs.
- **Command-line workflows** for running sampling and Monte Carlo from a
  structure file and a Python settings file.

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/CharlesSchwarz/disorderly.git"
```

Or clone the repository to run the tests, notebooks, and examples:

```bash
git clone https://github.com/CharlesSchwarz/disorderly.git
cd disorderly
pip install -e ".[test,notebooks]"
```

Optional extras add machine-learned potentials and exhaustive enumeration:

```bash
pip install -e ".[mace]"    # MACE energies
pip install -e ".[chgnet]"  # CHGNet energies
pip install -e ".[icet]"    # exhaustive enumeration of distinct configurations
pip install -e ".[plot]"    # matplotlib, for plot_trajectory
```

The `mace` and `chgnet` dependencies may require conflicting versions of the
torch stack; if so, install them in separate environments.

Requires Python >= 3.10.

## Getting started

- **`notebooks/`** -- tutorials on the key concepts and the Python API.
  Start here.
- **`cli_example/`** -- running sampling and Monte Carlo from the command
  line with settings files.
- **`demos/`** -- longer studies of real materials.

## Tutorial notebooks

The notebooks are written to be read in order, assuming no prior experience
with Monte Carlo. Install the `notebooks` extra to run them.

1. `01_disordered_structure_tools.ipynb` -- disordered structures, ordered
   configurations, counting configurations, symmetry, and editing occupancies.
2. `02_random_sampling_walkthrough.ipynb` -- energy models (Ewald) and random
   sampling for low-energy configurations.
3. `03_simple_MC_example.ipynb` -- how Monte Carlo works, and a run compared with
   random sampling. Saves its run for the next notebook.
4. `04_MC_analysis.ipynb` -- equilibration, acceptance rate, the effect of
   temperature, and the distinct low-energy structures found.

## Command-line workflows

Two console commands are installed with the package. Each takes a structure
and a Python settings file holding the run controls and the energy model:

```bash
run-random-sampling parent.cif random_sampling_settings.py
run-monte-carlo parent.cif initial.cif 700 monte_carlo_settings.py  # temperature in K
```

A Monte Carlo settings file defines its move policy either as
`policy = MonteCarloPolicy(...)` or with a `get_policy(...)` function; the
command adds the parent structure, initial structure, and temperature to the
policy's `context`. Commented reference settings files, listing every option,
are `src/disorder_mc/cli/random_sampling_settings_reference.py` and
`src/disorder_mc/cli/monte_carlo_settings_reference.py`: copy one and edit it.
Both use MACE by default; set `model_path` to your model file or swap in
another energy.

Worked examples, each with a README:

- `cli_example/random_sampling/` -- random sampling of Li4Ti5O12 with MACE,
  with its output.
- `cli_example/monte_carlo/` -- Li hops between tetrahedral and octahedral
  sites in LLZO, scored by CHGNet; about 6 minutes on a laptop CPU.
- `demos/bcc_ising_ordering/` -- `run-monte-carlo` with a simple pair
  energy; runs in minutes without MACE.

## Demos

`demos/` contains worked examples, validation runs, and studies of real
materials, ordered from simplest to most expensive: random sampling vs.
exact enumeration on a toy 2D grid (a single notebook), the order-disorder
transition of a bcc pair model (beta-brass), Monte Carlo vs. exact
enumeration for LiFeO2, ordering in Cu3Au, cation inversion in MgAl2O4
against neutron data, and short-range order in Li4Ti5O12. Apart from the toy 2D notebook, each demo has
a README, run scripts, and committed summary figures. See `demos/README.md`.
Some take hours and need MACE.

## Design notes

`docs/design.md` documents the package's scientific conventions: how
compositions of ordered configurations are fixed, oxidation states and
effective charges, the two notions of symmetry equivalence and configuration
degeneracy, the Monte Carlo energy and acceptance rule, reproducibility, and
the tolerances and units used throughout.

## Package layout

- `disorder_mc.structures` -- disordered-structure inspection and
  manipulation: disordered site groups and their whole-atom counts,
  supercell multipliers, configuration counts, and occupancy editing
  (`replace_species_and_occupancy`, `set_site_occupancies`).
- `disorder_mc.symmetry` -- space groups, Wyckoff sites, labels, and
  per-site summary tables (`describe_wyckoff_sites`),
  symmetric equivalence, configuration degeneracy, and counts of
  symmetrically distinct configurations.
- `disorder_mc.energy` -- disordered-structure Ewald energies.
- `disorder_mc.sampling` -- random ordered-configuration generation and
  energy-ranked sampling.
- `disorder_mc.ordering` -- compact, reversible ordering keys for
  configurations relative to their parent.
- `disorder_mc.enumeration` -- exhaustive enumeration of symmetrically
  distinct ordered configurations (optional, requires the `icet` extra).
- `disorder_mc.mlip` -- generic ASE-calculator energy evaluator (optional
  relaxation, thread-safe) shared by the machine-learned potential backends;
  wrap any ASE calculator with `ASECalculatorEvaluator(calculator_factory=...)`.
- `disorder_mc.mace` -- optional MACE energy evaluators and MC policies.
- `disorder_mc.chgnet` -- optional CHGNet energy evaluators and MC policies
  (pretrained model by default, or a fine-tuned `model_path`).
- `disorder_mc.settings` -- the configurable Monte Carlo policy interface.
- `disorder_mc.monte_carlo` -- move primitives and the Metropolis engine.
- `disorder_mc.diagnostics` -- checks that a policy's move proposals satisfy
  detailed balance.
- `disorder_mc.results` -- trajectory/result dataclasses.
- `disorder_mc.io` -- saving/loading structures, trajectories, and settings.
- `disorder_mc.cli` -- thin CLI wrappers (see `scripts/`).

## Citing

If you use `disorder_mc` in published work, please cite it using the metadata
in `CITATION.cff` (GitHub's "Cite this repository" button gives BibTeX and
APA formats).

## Packages used by disorder_mc

If you publish results obtained with `disorder_mc`, please also cite the
packages it builds on.

- [pymatgen](https://pymatgen.org/) -- structures, symmetry analysis,
  structure matching, and Ewald summation, used throughout.
  S. P. Ong *et al.*, *Comput. Mater. Sci.* **68**, 314 (2013),
  [doi:10.1016/j.commatsci.2012.10.028](https://doi.org/10.1016/j.commatsci.2012.10.028).
- [spglib](https://spglib.readthedocs.io/) -- space-group and Wyckoff
  analysis (via pymatgen).
  A. Togo, K. Shinohara, and I. Tanaka, *Sci. Technol. Adv. Mater.: Methods*
  **4**, 2384822 (2024),
  [doi:10.1080/27660400.2024.2384822](https://doi.org/10.1080/27660400.2024.2384822).
- [MACE](https://github.com/ACEsuit/mace) -- machine-learned interatomic
  potentials for energy evaluation (optional `mace` extra).
  I. Batatia *et al.*, *Adv. Neural Inf. Process. Syst.* **35** (2022),
  [arXiv:2206.07697](https://arxiv.org/abs/2206.07697); for the foundation
  models, I. Batatia *et al.*,
  [arXiv:2401.00096](https://arxiv.org/abs/2401.00096).
- [CHGNet](https://github.com/CederGroupHub/chgnet) -- pretrained universal
  machine-learned potential for energy evaluation (optional `chgnet` extra).
  B. Deng *et al.*, *Nat. Mach. Intell.* **5**, 1031 (2023),
  [doi:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3).
- [ASE](https://wiki.fysik.dtu.dk/ase/) -- atoms objects and structure
  relaxation for the MACE, CHGNet, and icet integrations.
  A. Hjorth Larsen *et al.*, *J. Phys.: Condens. Matter* **29**, 273002 (2017),
  [doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e).
- [icet](https://icet.materialsmodeling.org/) -- structure enumeration for
  exhaustive enumeration of symmetrically distinct configurations (optional
  `icet` extra).
  M. Ångqvist *et al.*, *Adv. Theory Simul.* **2**, 1900015 (2019),
  [doi:10.1002/adts.201900015](https://doi.org/10.1002/adts.201900015).

## Related packages

More tools for ordering disordered structures:

- [supercell](https://orex.github.io/supercell/) -- generates ordered
  supercells from a disordered CIF, merges symmetry-equivalent ones, and
  ranks them by Ewald energy.
  K. Okhotnikov, T. Charpentier, and S. Cadars, *J. Cheminform.* **8**, 17
  (2016),
  [doi:10.1186/s13321-016-0129-3](https://doi.org/10.1186/s13321-016-0129-3).
- [SOD](https://github.com/gcmt-group/sod) (Site Occupancy Disorder) --
  enumerates configurations that are inequivalent under the parent's
  symmetry, with their degeneracies.
  R. Grau-Crespo *et al.*, *J. Phys.: Condens. Matter* **19**, 256201 (2007),
  [doi:10.1088/0953-8984/19/25/256201](https://doi.org/10.1088/0953-8984/19/25/256201).
- [enumlib](https://github.com/msg-byu/enumlib) -- the Hart--Forcade
  derivative-structure enumeration algorithm (also used by icet, and wrapped
  in pymatgen as `EnumerateStructureTransformation`).
  G. L. W. Hart and R. W. Forcade, *Phys. Rev. B* **77**, 224115 (2008),
  [doi:10.1103/PhysRevB.77.224115](https://doi.org/10.1103/PhysRevB.77.224115).

For cluster expansion and MC calculations, you may also check out:

- [smol](https://github.com/CederGroupHub/smol) (Statistical Mechanics on
  Lattices).
  L. Barroso-Luque *et al.*, *J. Open Source Softw.* **7**, 4504 (2022),
  [doi:10.21105/joss.04504](https://doi.org/10.21105/joss.04504).
- [CLEASE](https://gitlab.com/computationalmaterials/clease).
  J. H. Chang *et al.*, *J. Phys.: Condens. Matter* **31**, 325901 (2019),
  [doi:10.1088/1361-648X/ab1bbc](https://doi.org/10.1088/1361-648X/ab1bbc).
- [ATAT](https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/)
  (Alloy Theoretic Automated Toolkit).
  A. van de Walle, M. Asta, and G. Ceder, *Calphad* **26**, 539 (2002),
  [doi:10.1016/S0364-5916(02)80006-2](https://doi.org/10.1016/S0364-5916(02)80006-2).
- [CASM](https://github.com/prisms-center/CASMcode).
  B. Puchala *et al.*, *Comput. Mater. Sci.* **217**, 111897 (2023),
  [doi:10.1016/j.commatsci.2022.111897](https://doi.org/10.1016/j.commatsci.2022.111897).
- https://arxiv.org/pdf/2603.08855 

## License

MIT (see `LICENSE`).
