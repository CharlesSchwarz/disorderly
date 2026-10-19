# %% [markdown]
# # Demo: Li/Ti short-range order in Li4Ti5O12
#
# Loads the outputs of `run.py` (and, if present, the MACE Monte Carlo sweep
# made by `setup_sweep.py`) and makes the figures. Runs as a plain script
# (`python demos/li4ti5o12_short_range_order/analysis.py`) or cell by cell in
# VS Code / Jupyter via the `# %%` markers.

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
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
import common  # noqa: E402

SHELL_COLORS = ["#2a78d6", "#eb6834"]
EWALD_COLOR = "#2a78d6"
MACE_COLOR = "#1baf7a"
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

sro = json.loads((RESULTS / "sro_summary.json").read_text())
runs = sorted(sro["runs"], key=lambda r: r["temperature_K"])
n_shells = len(sro["shell_radii_A"])

# %% [markdown]
# ## Short-range order vs. temperature (Ewald)

# %%
print(f"{'T (K)':>9} {'acc':>6} " + " ".join(f"{'alpha_' + str(n + 1):>15}" for n in range(n_shells)) + "   drift a1")
for r in runs:
    alphas = " ".join(f"{r[f'alpha_{n + 1}']:+.3f} +/- {r[f'alpha_{n + 1}_err']:.3f}" for n in range(n_shells))
    print(f"{r['temperature_K']:>9.0f} {r['acceptance_rate']:>6.3f} {alphas}   {r['alpha_1_drift']:+.3f}")

#: Runs accepting fewer moves than this are frozen rather than equilibrated.
MIN_ACCEPTANCE = 0.01

T = np.array([r["temperature_K"] for r in runs])
frozen = np.array([r["acceptance_rate"] < MIN_ACCEPTANCE for r in runs])
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.axhline(-0.2, color=REFERENCE, lw=1, ls=(0, (1, 2)))
ax.text(T[0], -0.2, "  no Li-Li neighbors", va="bottom", fontsize=8, color=INK_SECONDARY)
for n in range(n_shells):
    alpha = np.array([r[f"alpha_{n + 1}"] for r in runs])
    ax.axhline(sro["random_baseline_alpha"][n], color=SHELL_COLORS[n], lw=1, ls="--", alpha=0.6)
    ax.errorbar(
        T,
        alpha,
        yerr=[r[f"alpha_{n + 1}_err"] for r in runs],
        fmt="-",
        lw=2,
        color=SHELL_COLORS[n],
        capsize=2,
        label=f"Shell {n + 1}: {sro['shell_radii_A'][n]:.2f} A, {sro['shell_coordination'][n]} neighbors",
    )
    ax.plot(T[~frozen], alpha[~frozen], "o", ms=7, color=SHELL_COLORS[n], mec=SURFACE, mew=1.2, zorder=3)
    ax.plot(T[frozen], alpha[frozen], "o", ms=7, mfc=SURFACE, mec=SHELL_COLORS[n], mew=1.5, zorder=3)
ax.plot([], [], "o", ms=7, mfc=SURFACE, mec=INK_SECONDARY, mew=1.5, label=f"Frozen (acceptance < {MIN_ACCEPTANCE:.0%})")
ax.text(
    T[-1], sro["random_baseline_alpha"][0], "random arrangement ", ha="right", va="bottom", fontsize=8, color=INK_SECONDARY
)
ax.set_xscale("log")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel(r"Warren-Cowley $\alpha$ (Li-Li, 16d)")
ax.set_title("Li4Ti5O12 16d short-range order, Ewald energy", loc="left", color=INK, fontsize=10)
ax.legend(loc="center left")
fig.tight_layout()
fig.savefig(RESULTS / "sro_vs_temperature.png", dpi=200)

# %% [markdown]
# ## MACE check (optional)
#
# Only if `run.py --mace MODEL` has been run. Energies are per 16d Li,
# relative to the mean of the random arrangements.

# %%
mace_path = RESULTS / "mace_check.json"
if mace_path.exists():
    mace = json.loads(mace_path.read_text())
    print(json.dumps({"groups": mace["groups"], "spearman_alpha_1_vs_energy": mace["spearman_alpha_1_vs_energy"]}, indent=2))

    rows = mace["structures"]
    n_li = 8
    alpha_1 = np.array([r["alpha"][0] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, key, color, title in (
        (axes[0], "ewald_eV", EWALD_COLOR, "Ewald"),
        (axes[1], "mace_relaxed_eV", MACE_COLOR, "MACE (relaxed)"),
    ):
        energies = np.array([r[key] for r in rows])
        random_mean = energies[[r["group"] == "random" for r in rows]].mean()
        ax.axhline(0, color=REFERENCE, lw=1)
        ax.scatter(alpha_1, (energies - random_mean) / n_li, s=40, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_xlabel(r"$\alpha_1$ (Li-Li, first 16d shell)")
        ax.set_ylabel("Energy vs. random (eV per 16d Li)")
        rho = mace["spearman_alpha_1_vs_energy"][key]
        ax.set_title(f"{title}   (Spearman with $\\alpha_1$: {rho:+.2f})", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "mace_check.png", dpi=200)
else:
    print("No mace_check.json; run `run.py --mace MODEL` to add the MACE comparison.")

# %% [markdown]
# ## MACE Monte Carlo (optional)
#
# Only if the sweep made by `setup_sweep.py` has steps. For each run: discard
# the first max(500 steps, 20%) as equilibration, then report alpha with a
# block-averaged error (20 blocks), the drift between the two halves of the
# production run, and the acceptance and cache-hit rates. Ordered and random
# starts converging to the same alpha is the equilibration check.

# %%
#: Folder holding the run folders (as set by SWEEP_DIR in setup_sweep.py).
SWEEP_DIR = common.DEFAULT_SWEEP_DIR
DISCARD_STEPS = 500
DISCARD_FRACTION = 0.2
START_STYLES = {"ordered": "filled", "random": "hollow"}


def block_mean(values: np.ndarray, n_blocks: int = 20) -> tuple[float, float]:
    means = np.array([b.mean() for b in np.array_split(values, n_blocks)])
    return float(values.mean()), float(means.std(ddof=1) / np.sqrt(n_blocks))


def summarize_mace_run(run_dir: Path) -> dict | None:
    steps = common.read_jsonl(run_dir / "steps.jsonl")
    if not steps:
        return None
    metadata = json.loads((run_dir / "metadata.json").read_text())
    discard = max(DISCARD_STEPS, int(DISCARD_FRACTION * len(steps)))
    production = steps[discard:]
    evaluated = [s for s in steps if not s["cached"]]
    row = {
        "run": run_dir.name,
        **{k: metadata[k] for k in ("temperature_K", "initial", "seed", "model", "relax", "relax_cell", "fmax")},
        "steps": len(steps),
        "discarded": discard,
        "acceptance_rate": float(np.mean([s["accepted"] for s in steps])),
        "cache_hit_rate": float(np.mean([s["cached"] for s in steps])),
        "unconverged_relaxations": int(sum(s["relax_converged"] is False for s in evaluated)),
        "mean_eval_time_s": float(np.mean([s["eval_time_s"] for s in evaluated])) if evaluated else None,
    }
    if len(production) >= 40:
        for n in range(n_shells):
            series = np.array([s[f"alpha_{n + 1}"] for s in production])
            row[f"alpha_{n + 1}"], row[f"alpha_{n + 1}_err"] = block_mean(series)
            half = len(series) // 2
            row[f"alpha_{n + 1}_drift"] = float(series[half:].mean() - series[:half].mean())
    return row


run_dirs = sorted(d for d in Path(SWEEP_DIR).glob("*") if (d / "metadata.json").exists())
mace_runs = [row for row in map(summarize_mace_run, run_dirs) if row is not None]
if mace_runs:
    for row in mace_runs:
        alphas = "  ".join(
            f"a{n + 1}={row[f'alpha_{n + 1}']:+.3f}+/-{row[f'alpha_{n + 1}_err']:.3f}" if f"alpha_{n + 1}" in row else f"a{n + 1}=  (too short)"
            for n in range(n_shells)
        )
        print(
            f"{row['run']:<24} {row['steps']:>6} steps  {alphas}  acc {row['acceptance_rate']:.2f}  "
            f"cache hits {row['cache_hit_rate']:.2f}  unconverged {row['unconverged_relaxations']}"
        )
    common.write_json_atomic(
        RESULTS / "mace_mc_summary.json",
        {"discard_steps": DISCARD_STEPS, "discard_fraction": DISCARD_FRACTION, "runs": mace_runs},
    )

    finished = [r for r in mace_runs if "alpha_1" in r]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhline(-0.2, color=REFERENCE, lw=1, ls=(0, (1, 2)))
    for n in range(n_shells):
        ax.axhline(sro["random_baseline_alpha"][n], color=SHELL_COLORS[n], lw=1, ls="--", alpha=0.6)
        for initial, style in START_STYLES.items():
            selected = sorted((r for r in finished if r["initial"] == initial), key=lambda r: r["temperature_K"])
            if not selected:
                continue
            ax.errorbar(
                [r["temperature_K"] for r in selected],
                [r[f"alpha_{n + 1}"] for r in selected],
                yerr=[r[f"alpha_{n + 1}_err"] for r in selected],
                fmt="-o",
                ms=6,
                lw=1.5,
                capsize=2,
                color=SHELL_COLORS[n],
                mfc=SHELL_COLORS[n] if style == "filled" else SURFACE,
                mec=SHELL_COLORS[n],
                label=f"Shell {n + 1}, {initial} start",
            )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Warren-Cowley $\alpha$ (Li-Li, 16d)")
    ax.set_title("Li4Ti5O12 16d short-range order, MACE Monte Carlo", loc="left", color=INK, fontsize=10)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "mace_sro_vs_temperature.png", dpi=200)
    print(f"Wrote {RESULTS / 'mace_mc_summary.json'} and {RESULTS / 'mace_sro_vs_temperature.png'}")
else:
    print(f"No MACE Monte Carlo runs with steps in {SWEEP_DIR}; see 'MACE Monte Carlo' in the README.")
