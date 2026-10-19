# %% [markdown]
# # Demo: B2 ordering in a nearest-neighbour pair model
#
# Loads the `run-monte-carlo` trajectories in `results/raw/` and makes the
# figure. Runs as a plain script
# (`python demos/bcc_ising_ordering/analysis.py`) or cell by cell in
# VS Code / Jupyter via the `# %%` markers.

# %%
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Structure

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running as notebook cells
    HERE = Path.cwd()
RESULTS = HERE / "results"
RAW = RESULTS / "raw"

#: Exact ordering temperature of the bcc nearest-neighbour model for J = 0.010 eV.
T_C = 0.010 / (0.157373 * 8.617333262e-5)
#: Fraction of each run discarded as equilibration.
DISCARD = 0.5
#: Blocks for the error bar on each run's mean order parameter.
N_BLOCKS = 5
STARTS = {"ordered": "heating (ordered start)", "random_seed0": "cooling (random start)"}

START_COLORS = {"ordered": "#eb6834", "random_seed0": "#2a78d6"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
REFERENCE = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "legend.frameon": False,
    }
)

# %% [markdown]
# ## Summarize each run
#
# Each step records the *proposed* configuration's order parameter, so the
# current value is carried forward from the last accepted step.

# %%
def summarize(run_dir: Path) -> dict:
    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    steps = trajectory["steps"]
    current = steps[0]["properties"]["order"]
    order = []
    for step in steps[1:]:
        if step["accepted"]:
            current = step["properties"]["order"]
        order.append(current)
    start = int(len(order) * DISCARD)
    order = np.array(order[start:])
    energy = np.array([step["energy"] for step in steps[1:]][start:])
    blocks = [block.mean() for block in np.array_split(order, N_BLOCKS)]
    return {
        "run": run_dir.name,
        "start": run_dir.name.rsplit("_T", 1)[0],
        "temperature_K": trajectory["parameters"]["temperature"],
        "steps": len(steps) - 1,
        "production_steps": len(order),
        "acceptance_rate": float(np.mean([step["accepted"] for step in steps[1:]])),
        "order": float(order.mean()),
        "order_err": float(np.std(blocks, ddof=1) / np.sqrt(N_BLOCKS)),
        "energy_eV": float(energy.mean()),
    }


runs = sorted(
    (summarize(d) for d in RAW.iterdir() if (d / "trajectory.json").exists()),
    key=lambda r: (r["start"], r["temperature_K"]),
)
if not runs:
    raise SystemExit(f"No trajectories in {RAW}; run the commands printed by make_inputs.py first.")
n_sites = len(Structure.from_file(HERE / "parent.cif"))
for r in runs:
    r["energy_per_site_meV"] = 1000 * r["energy_eV"] / n_sites

print(f"T_c (exact, infinite lattice) = {T_C:.0f} K; {n_sites} sites; first {DISCARD:.0%} of each run discarded")
print(f"{'start':>13} {'T (K)':>6} {'acc':>6} {'order':>15} {'E/site (meV)':>13}")
for r in runs:
    print(
        f"{r['start']:>13} {r['temperature_K']:>6.0f} {r['acceptance_rate']:>6.3f} "
        f"{r['order']:.3f} +/- {r['order_err']:.3f} {r['energy_per_site_meV']:>13.2f}"
    )

(RESULTS / "summary.json").write_text(
    json.dumps(
        {"T_c_exact_K": T_C, "n_sites": n_sites, "discard_fraction": DISCARD, "n_blocks": N_BLOCKS, "runs": runs},
        indent=1,
    )
)

# %% [markdown]
# ## Order parameter and energy vs. temperature

# %%
fig, (ax_order, ax_energy) = plt.subplots(1, 2, figsize=(10, 4))
for ax in (ax_order, ax_energy):
    ax.axvline(T_C, color=REFERENCE, lw=1, ls="--")
ax_order.text(T_C, 1.0, f" exact $T_c$ = {T_C:.0f} K", fontsize=8, color=INK_SECONDARY, va="top")
for start, label in STARTS.items():
    rs = [r for r in runs if r["start"] == start]
    if not rs:
        continue
    T = [r["temperature_K"] for r in rs]
    ax_order.errorbar(
        T, [r["order"] for r in rs], yerr=[r["order_err"] for r in rs],
        fmt="-o", ms=4, lw=1.5, capsize=2, color=START_COLORS[start], label=label,
    )
    ax_energy.plot(T, [r["energy_per_site_meV"] for r in rs], "-o", ms=4, lw=1.5, color=START_COLORS[start], label=label)
ax_order.set(xlabel="Temperature (K)", ylabel="B2 order parameter", ylim=(0, 1.02))
ax_energy.set(xlabel="Temperature (K)", ylabel="Energy per site (meV)")
ax_order.legend(loc="lower left")
fig.suptitle(f"bcc nearest-neighbour pair model, {n_sites} sites", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "order_vs_temperature.png", dpi=150)
print(f"Wrote {RESULTS / 'summary.json'} and {RESULTS / 'order_vs_temperature.png'}")
