# Monte Carlo CLI example: Li sites in cubic LLZO

Runs `run-monte-carlo` on cubic Li7La3Zr2O12 (LLZO), whose 56 Li are spread over 24 tetrahedral and 48 octahedral sites. The run starts with every tetrahedral site filled (`cLLZO_init_all_tet_sites_occupied.cif`). Each step hops one Li to a vacant Li site, scored by its unrelaxed CHGNet energy, so the tetrahedral/octahedral split can change. The number of tetrahedral Li is recorded as `n_tet` at every step.

Install the CHGNet extra, then run from this directory (1200 is the temperature in K):

```bash
pip install -e "../..[chgnet]"
```

```bash
run-monte-carlo cubic_LLZO_parent.cif cLLZO_init_all_tet_sites_occupied.cif 1200 monte_carlo_settings.py
```

The trajectory and the lowest-energy structures are saved to `cubic_LLZO_parent_T1200K_monte_carlo_results/`. Edit `monte_carlo_settings.py` to change the number of steps, the seed, or the energy model.

**Run speed:** on a laptop CPU, each step takes about 0.35 s (one CHGNet energy evaluation), so the default 1000 steps take about 6 minutes. Setting `relax=True` makes every step a full structure relaxation, which is much slower. The Ewald energy (commented out in the settings file) is about ten times faster but much less realistic.
