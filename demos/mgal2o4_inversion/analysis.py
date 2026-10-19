# %% [markdown]
# # Demo: Cation inversion in MgAl2O4
#
# Summarizes the MACE Monte Carlo runs in the sweep folder and compares the
# inversion x(T) with Redfern et al. (1999). Runs as a plain script
# (`python demos/mgal2o4_inversion/analysis.py`) or cell by cell in VS Code /
# Jupyter via the `# %%` markers.

# %%
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running as notebook cells
    HERE = Path.cwd()
sys.path.insert(0, str(HERE))
import common  # noqa: E402

#: Folder holding the run folders (as set by SWEEP_DIR in setup_sweep.py).
SWEEP_DIR = common.DEFAULT_SWEEP_DIR
SUMMARY = common.RESULTS / "mace_mc_summary.json"
FIGURE = common.RESULTS / "inversion_vs_temperature.png"
#: Equilibration discarded from each run: the larger of these.
DISCARD_STEPS = 500
DISCARD_FRACTION = 0.2
#: Below this temperature Redfern's heating and cooling legs disagree: the
#: inversion is frozen in rather than in equilibrium.
EQUILIBRIUM_T = 1000.0
#: Measured points within this many K of a run's temperature are averaged for comparison.
COMPARE_WINDOW_K = 60.0

HEATING_COLOR = "#2a78d6"
COOLING_COLOR = "#eb6834"
MACE_COLOR = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SHADE = "#efeeea"

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

redfern = common.read_redfern()

# %% [markdown]
# ## Summarize the runs
#
# For each run folder: discard the equilibration steps, then report the mean
# x with a block-averaged error (20 blocks), the drift between the two halves
# of the production run, the acceptance and cache-hit rates, and the number of
# relaxations that hit the step cap. Normal and random starts converging to the
# same x is the equilibration check.


# %%
def block_mean(values: np.ndarray, n_blocks: int = 20) -> tuple[float, float]:
    means = np.array([b.mean() for b in np.array_split(values, n_blocks)])
    return float(values.mean()), float(means.std(ddof=1) / np.sqrt(n_blocks))


def summarize_run(run_dir: Path) -> dict | None:
    steps = common.read_jsonl(run_dir / "steps.jsonl")
    if not steps:
        return None
    metadata = json.loads((run_dir / "metadata.json").read_text())
    discard = max(DISCARD_STEPS, int(DISCARD_FRACTION * len(steps)))
    production = steps[discard:]
    evaluated = [s for s in steps if not s["cached"]]
    row = {
        "run": run_dir.name,
        **{k: metadata[k] for k in ("temperature_K", "initial", "seed", "model", "fmax")},
        "steps": len(steps),
        "discarded": discard,
        "acceptance_rate": float(np.mean([s["accepted"] for s in steps])),
        "cache_hit_rate": float(np.mean([s["cached"] for s in steps])),
        "unconverged_relaxations": int(sum(s["relax_converged"] is False for s in evaluated)),
        "mean_relax_time_s": float(np.mean([s["relax_time_s"] for s in evaluated])) if evaluated else None,
    }
    if len(production) >= 40:
        x = np.array([s["x"] for s in production])
        row["x_mean"], row["x_err"] = block_mean(x)
        half = len(x) // 2
        row["x_drift"] = float(x[half:].mean() - x[:half].mean())
    return row


run_dirs = sorted(d for d in Path(SWEEP_DIR).glob("*") if (d / "metadata.json").exists())
runs = [row for row in map(summarize_run, run_dirs) if row is not None]
for row in runs:
    print(
        f"{row['run']:<24} {row['steps']:>6} steps  x = {row.get('x_mean', float('nan')):.3f} "
        f"+/- {row.get('x_err', float('nan')):.3f}  drift {row.get('x_drift', float('nan')):+.3f}  "
        f"acc {row['acceptance_rate']:.2f}  cache hits {row['cache_hit_rate']:.2f}  "
        f"unconverged {row['unconverged_relaxations']}"
    )
if runs:
    common.write_json_atomic(
        SUMMARY, {"discard_steps": DISCARD_STEPS, "discard_fraction": DISCARD_FRACTION, "runs": runs}
    )
    print(f"Wrote {SUMMARY}")
else:
    print(f"No runs with steps in {SWEEP_DIR}; run the sweep (see README) or point SWEEP_DIR at it.")
    if SUMMARY.exists():
        runs = json.loads(SUMMARY.read_text())["runs"]
        print(f"Using the existing {SUMMARY.name} ({len(runs)} runs).")

# %% [markdown]
# ## Inversion vs. temperature

# %%
fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.axvspan(250, EQUILIBRIUM_T, color=SHADE, zorder=0)
ax.text(270, 0.545, "frozen in (heating and\ncooling disagree)", va="top", fontsize=8, color=INK_SECONDARY)
for leg, color, label in (
    ("heating", HEATING_COLOR, "Redfern et al., heating"),
    ("cooling", COOLING_COLOR, "Redfern et al., cooling"),
):
    rows = [r for r in redfern if r["leg"] == leg]
    ax.plot(
        [r["temperature_K"] for r in rows],
        [r["x"] for r in rows],
        "o",
        ms=6,
        color=color,
        mec=SURFACE,
        mew=1,
        label=label,
        zorder=3,
    )
mc_runs = [r for r in runs if "x_mean" in r]
for initial, face in (("normal", MACE_COLOR), ("random", SURFACE)):
    selected = [r for r in mc_runs if r["initial"] == initial]
    if selected:
        ax.errorbar(
            [r["temperature_K"] for r in selected],
            [r["x_mean"] for r in selected],
            yerr=[r["x_err"] for r in selected],
            fmt="D",
            ms=7,
            mfc=face,
            mec=MACE_COLOR,
            mew=1.5,
            color=MACE_COLOR,
            capsize=3,
            zorder=4,
            label=f"MACE Monte Carlo ({initial} start)",
        )
ax.set_xlim(250, 2000)
ax.set_ylim(-0.01, 0.56)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Inversion x (Al fraction on tetrahedral sites)")
ax.set_title("MgAl2O4 cation inversion", loc="left", color=INK, fontsize=10)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGURE, dpi=200)
print(f"Wrote {FIGURE}")

# %% [markdown]
# ## Monte Carlo vs. experiment
#
# Each run is compared with the mean of the measured equilibrium points
# (T >= 1000 K) within 60 K of its temperature.

# %%
equilibrium = [r for r in redfern if r["temperature_K"] >= EQUILIBRIUM_T]
differences = []
for row in mc_runs:
    nearby = [r["x"] for r in equilibrium if abs(r["temperature_K"] - row["temperature_K"]) <= COMPARE_WINDOW_K]
    if not nearby:
        continue
    measured = float(np.mean(nearby))
    differences.append(row["x_mean"] - measured)
    print(
        f"T = {row['temperature_K']:>6.0f} K {row['initial']:>6} start: MC x = {row['x_mean']:.3f} +/- {row['x_err']:.3f}, "
        f"measured {measured:.3f} (n = {len(nearby)})"
    )
if differences:
    differences = np.array(differences)
    print(f"MC - measured: mean {differences.mean():+.3f}, rms {np.sqrt(np.mean(differences**2)):.3f}")
else:
    print("No Monte Carlo results at equilibrium temperatures to compare yet.")
