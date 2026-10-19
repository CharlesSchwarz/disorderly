# Demos

Worked examples and longer-running demonstrations of the package on toy
models, validation problems, and real materials, listed below from simplest
to most expensive. Unlike `notebooks/` (quick walkthroughs) and `tests/` (fast
checks), the demos produce data and figures and take from minutes to days.
`cu3au_ordering` and `mgal2o4_inversion`, and the MACE parts of
`lifeo2_exact_enumeration` and `li4ti5o12_short_range_order`, need the `mace` extra
(`pip install -e ".[mace]"`) and a MACE model file.

See the README within each folder for more details about running each demo.

| Demo | Shows | Expected runtime |
|---|---|---|
| `toy_2d_enumeration` | Simple notebook on a toy 2D system with two species. Compare results from random sampling with exact enumeration | < 1 min |
| `bcc_ising_ordering` | Order-disorder transition for the bcc Ising model. Compare results vs. the known critical temperature value ($`T_c`$ = 737 K). Based on Monte Carlo (MC) sampling with a custom energy function. | < 1 hour |
| `lifeo2_exact_enumeration` | Ordering the Li-Fe sublattice in rock-salt LiFeO2. Check that MC matches the Boltzmann distribution we get from exact enumeration. | < 1 hour |
| `cu3au_ordering` | L1₂ order-disorder transition in Cu3Au. Compare MC results vs. the measured 663 K transition temperature. MC calculations use the MACE energy evaluator with relaxation and are run from the CLI. | multiple days |
| `mgal2o4_inversion` | Cation inversion x(T) in MgAl2O4. Compare MC results vs. in situ neutron data (Redfern et al. 1999). MC calculations use the MACE energy evaluator with relaxation. | multiple days on a cluster |
| `li4ti5o12_short_range_order` | Looking at Warren-Cowley short-range order on the Li4Ti5O12 Li sublattice vs. temperature. Based on MC sampling, first with `PrecomputedEwald` then with MACE. (MACE results pending) | multiple days on a cluster |

Run from the repository root with the package importable, e.g.
`pip install -e .` and then `python demos/lifeo2_exact_enumeration/run.py`.

The MACE Monte Carlo runs of `mgal2o4_inversion` and `li4ti5o12_short_range_order`
are too expensive for one
`run.py`: `setup_sweep.py` creates one folder per run, each with a
`run_mc.py` to run locally or as a cluster job that resumes where it
stopped. See `mgal2o4_inversion/README.md` for running them on a cluster.
