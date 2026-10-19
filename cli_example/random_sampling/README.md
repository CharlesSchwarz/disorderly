# Random sampling CLI example: spinel LTO

Runs `run-random-sampling` on spinel Li4Ti5O12 (LTO), whose 16 octahedral sites per cell are shared by Li and Ti. It generates random orderings of those sites, scores each with a MACE energy, and saves the lowest-energy, symmetrically distinct configurations.

Install the MACE extra and set `model_path` in `random_sampling_settings.py` to your MACE model file, then run from this directory:

```bash
pip install -e "../..[mace]"
```

```bash
run-random-sampling LTO_COD_1001098.cif random_sampling_settings.py
```

The results (`sampled_configurations.json` and one CIF per configuration) are saved to `LTO_COD_1001098_random_sampling_results/`. Edit `random_sampling_settings.py` to change the number of attempts, the seed, or the energy model.
