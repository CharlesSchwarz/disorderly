# %% [markdown]
# # Demo: L1_2 ordering in Cu3Au with MACE
#
# Loads the `run-monte-carlo` trajectories in `results/raw/` and makes the
# figures. Runs as a plain script
# (`python demos/cu3au_ordering/analysis.py`) or cell by cell in
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

#: Measured L1_2 order-disorder temperature of Cu3Au (Okamoto et al. 1987).
T_EXPERIMENT = 663.0
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
# current value is carried forward from the last accepted step. Runs stopped
# at their time limit have fewer steps than `num_steps`.

# %%
def order_series(steps: list[dict]) -> np.ndarray:
    current = steps[0]["properties"]["order"]
    order = [current]
    for step in steps[1:]:
        if step["accepted"]:
            current = step["properties"]["order"]
        order.append(current)
    return np.array(order)


def summarize(run_dir: Path, n_sites: int) -> tuple[dict, np.ndarray]:
    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    steps = trajectory["steps"]
    order_all = order_series(steps)
    start = int((len(steps) - 1) * DISCARD)
    order = order_all[1:][start:]
    energy = np.array([step["energy"] for step in steps[1:]][start:])
    blocks = [block.mean() for block in np.array_split(order, N_BLOCKS)]
    evaluated = [step["properties"] for step in steps[1:] if "relax_converged" in step["properties"]]
    row = {
        "run": run_dir.name,
        "start": run_dir.name.rsplit("_T", 1)[0],
        "temperature_K": trajectory["parameters"]["temperature"],
        "steps": len(steps) - 1,
        "planned_steps": trajectory["parameters"]["num_steps"],
        "production_steps": len(order),
        "acceptance_rate": float(np.mean([step["accepted"] for step in steps[1:]])),
        "unconverged_relaxations": int(sum(not p["relax_converged"] for p in evaluated)),
        "order": float(order.mean()),
        "order_err": float(np.std(blocks, ddof=1) / np.sqrt(N_BLOCKS)),
        "order_final": float(order_all[-1]),
        "energy_per_site_meV": float(1000 * energy.mean() / n_sites),
    }
    return row, order_all


if not RAW.exists() or not any((d / "trajectory.json").exists() for d in RAW.iterdir()):
    raise SystemExit(f"No trajectories in {RAW}; run the commands printed by make_inputs.py first.")
n_sites = len(Structure.from_file(HERE / "parent.cif"))
summaries = [summarize(d, n_sites) for d in RAW.iterdir() if (d / "trajectory.json").exists()]
summaries.sort(key=lambda item: (item[0]["start"], item[0]["temperature_K"]))
runs = [row for row, _ in summaries]
series = {row["run"]: order for row, order in summaries}

print(f"{n_sites} sites; first {DISCARD:.0%} of each run discarded; measured T_c = {T_EXPERIMENT:.0f} K")
print(f"{'start':>13} {'T (K)':>6} {'steps':>11} {'acc':>6} {'unconv':>6} {'order':>15} {'E/site (meV)':>13}")
for r in runs:
    print(
        f"{r['start']:>13} {r['temperature_K']:>6.0f} {r['steps']:>5}/{r['planned_steps']:<5} "
        f"{r['acceptance_rate']:>6.3f} {r['unconverged_relaxations']:>6} "
        f"{r['order']:.3f} +/- {r['order_err']:.3f} {r['energy_per_site_meV']:>13.2f}"
    )

(RESULTS / "summary.json").write_text(
    json.dumps(
        {"T_experiment_K": T_EXPERIMENT, "n_sites": n_sites, "discard_fraction": DISCARD, "n_blocks": N_BLOCKS, "runs": runs},
        indent=1,
    )
)

# %% [markdown]
# ## Order parameter and energy vs. temperature
#
# Energies are relative to the perfectly ordered structure's mean energy at
# the lowest temperature.

# %%
reference = min((r for r in runs if r["start"] == "ordered"), key=lambda r: r["temperature_K"])["energy_per_site_meV"]
fig, (ax_order, ax_energy) = plt.subplots(1, 2, figsize=(10, 4))
for ax in (ax_order, ax_energy):
    ax.axvline(T_EXPERIMENT, color=REFERENCE, lw=1, ls="--")
ax_order.text(T_EXPERIMENT, 0.03, f" measured $T_c$ = {T_EXPERIMENT:.0f} K", fontsize=8, color=INK_SECONDARY, va="bottom")
for start, label in STARTS.items():
    rs = [r for r in runs if r["start"] == start]
    if not rs:
        continue
    T = [r["temperature_K"] for r in rs]
    ax_order.errorbar(
        T, [r["order"] for r in rs], yerr=[r["order_err"] for r in rs],
        fmt="-o", ms=4, lw=1.5, capsize=2, color=START_COLORS[start], label=label,
    )
    ax_energy.plot(
        T, [r["energy_per_site_meV"] - reference for r in rs], "-o", ms=4, lw=1.5,
        color=START_COLORS[start], label=label,
    )
ax_order.set(xlabel="Temperature (K)", ylabel="L1$_2$ order parameter", ylim=(0, 1.05))
ax_energy.set(xlabel="Temperature (K)", ylabel="Energy per site, relative (meV)")
ax_order.legend(loc="upper right")
fig.suptitle(f"Cu3Au, relaxed MACE, {n_sites} sites (second half of each run)", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "order_vs_temperature.png", dpi=150)

# %% [markdown]
# ## Order parameter vs. step
#
# The whole run, including the discarded first half, for each temperature.

# %%
temperatures = sorted({r["temperature_K"] for r in runs})
n_cols = 4
n_rows = -(-len(temperatures) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.2 * n_rows), sharex=True, sharey=True)
for ax, temperature in zip(axes.flat, temperatures):
    for start, label in STARTS.items():
        run = f"{start}_T{temperature:g}K"
        if run in series:
            ax.plot(series[run], lw=0.8, color=START_COLORS[start], label=label)
    ax.set_title(f"{temperature:g} K", fontsize=9, loc="left")
    ax.set_ylim(0, 1.05)
for ax in axes.flat[len(temperatures):]:
    ax.set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel("Step")
for ax in axes[:, 0]:
    ax.set_ylabel("Order")
if len(temperatures) < axes.size:  # use the first empty panel for the legend
    handles, labels = axes.flat[0].get_legend_handles_labels()
    spare = axes.flat[len(temperatures)]
    spare.set_visible(True)
    spare.axis("off")
    spare.legend(handles, labels, loc="center", fontsize=9)
else:
    axes.flat[0].legend(loc="lower left", fontsize=7)
fig.tight_layout()
fig.savefig(RESULTS / "order_vs_step.png", dpi=150)
print(f"Wrote {RESULTS / 'summary.json'}, {RESULTS / 'order_vs_temperature.png'} and {RESULTS / 'order_vs_step.png'}")
