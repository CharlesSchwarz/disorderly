# %% [markdown]
# # Demo: Monte Carlo vs. exact enumeration
#
# Loads the outputs of `run.py` and makes the figures. Runs as a plain script
# (`python demos/lifeo2_exact_enumeration/analysis.py`) or cell by cell in VS Code /
# Jupyter via the `# %%` markers.

# %%
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # running as notebook cells
    HERE = Path.cwd()
RESULTS = HERE / "results"
KB_EV_PER_K = 8.617333262e-5

EXACT_COLOR = "#2a78d6"
MC_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
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

exact = json.loads((RESULTS / "exact.json").read_text())
summary = json.loads((RESULTS / "mc_summary.json").read_text())
e0 = exact["classes"][0]["energy_eV"]

# %% [markdown]
# ## Symmetry checks

# %%
for key, value in exact["checks"].items():
    print(f"{key:>45}: {value}")
print()
print("Lowest-energy symmetry classes (energy relative to ground state, per 16-cation cell):")
for c in exact["classes"][:8]:
    print(f"  {c['energy_eV'] - e0:7.3f} eV   g = {c['degeneracy']:4d}   {c['spacegroup']}")

# %% [markdown]
# ## Mean energy and heat capacity

# %%
T_grid = np.array(exact["temperature_K"])
T_mc = np.array([row["temperature_K"] for row in summary])

fig, (ax_e, ax_c) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
ax_e.plot(T_grid, np.array(exact["mean_energy_eV"]) - e0, color=EXACT_COLOR, lw=2, label="Exact enumeration")
ax_e.errorbar(
    T_mc,
    [row["mc_mean_energy"] - e0 for row in summary],
    yerr=[row["mc_mean_energy_err"] for row in summary],
    fmt="o",
    ms=7,
    color=MC_COLOR,
    mec=SURFACE,
    mew=1.5,
    capsize=3,
    label="Monte Carlo",
)
ax_e.set_ylabel(r"$\langle E\rangle - E_0$ (eV per cell)")
ax_e.legend(loc="upper left")

ax_c.plot(T_grid, np.array(exact["heat_capacity_eV_per_K"]) / KB_EV_PER_K, color=EXACT_COLOR, lw=2)
ax_c.errorbar(
    T_mc,
    [row["mc_heat_capacity"] / KB_EV_PER_K for row in summary],
    yerr=[row["mc_heat_capacity_err"] / KB_EV_PER_K for row in summary],
    fmt="o",
    ms=7,
    color=MC_COLOR,
    mec=SURFACE,
    mew=1.5,
    capsize=3,
)
ax_c.set_ylabel(r"$C\,/\,k_B$ (per cell)")
ax_c.set_xlabel("Temperature (K)")
ax_c.set_xscale("log")
ax_c.set_xlim(3e3, 1e6)
fig.suptitle("LiFeO$_2$ cation ordering, 16-site cell (Ewald energy)", color=INK, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(RESULTS / "energy_heat_capacity.png", dpi=200)

# %% [markdown]
# ## Symmetry-class probabilities
#
# Exact probability of each class is g_i exp(-E_i / kT) / Z; Monte Carlo is
# the fraction of production steps spent in that class.

# %%
n_show = 10
shown = [summary[0], summary[2], summary[-1]]
labels = [f"{c['energy_eV'] - e0:.1f}\n{c['spacegroup']}" for c in exact["classes"][:n_show]]
x = np.arange(n_show)

fig, axes = plt.subplots(len(shown), 1, figsize=(7.2, 7.4), sharex=True)
for ax, row in zip(axes, shown):
    p_exact = np.array(row["class_probability_exact"][:n_show])
    p_mc = np.array(row["class_probability_mc"][:n_show])
    ax.bar(x, p_exact, width=0.6, color=EXACT_COLOR, alpha=0.35, edgecolor=EXACT_COLOR, lw=1, label="Exact")
    ax.plot(x, p_mc, "o", ms=7, color=MC_COLOR, mec=SURFACE, mew=1.5, label="Monte Carlo")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"T = {row['temperature_K']:,.0f} K   (total variation distance {row['class_total_variation_distance']:.3f})",
        loc="left",
        color=INK,
        fontsize=10,
    )
    ax.grid(axis="x", visible=False)
axes[0].legend(loc="upper right")
axes[-1].set_xticks(x, labels, fontsize=8)
axes[-1].set_xlabel(
    f"{n_show} lowest-energy of {len(exact['classes'])} symmetry classes: energy above ground state (eV per cell), space group"
)
fig.tight_layout()
fig.savefig(RESULTS / "class_probabilities.png", dpi=200)

# %% [markdown]
# ## Summary table

# %%
print(f"{'T (K)':>9} {'acc':>5} {'<E> z':>7} {'C z':>7} {'TVD':>6}")
for row in summary:
    ez = "n/a" if row["mean_energy_z"] is None else f"{row['mean_energy_z']:+.2f}"
    cz = "n/a" if row["heat_capacity_z"] is None else f"{row['heat_capacity_z']:+.2f}"
    print(f"{row['temperature_K']:>9.0f} {row['acceptance_rate']:>5.2f} {ez:>7} {cz:>7} {row['class_total_variation_distance']:>6.3f}")

# %% [markdown]
# ## Ewald vs. MACE (optional)
#
# Only if `run.py --mace MODEL` has been run. Both curves are exact (from the
# symmetry-class partition function); no Monte Carlo is involved.

# %%
MACE_COLOR = "#1baf7a"
mace_path = RESULTS / "mace_exact.json"
if mace_path.exists():
    mace = json.loads(mace_path.read_text())
    for key, value in mace["checks"].items():
        print(f"{key:>45}: {value}")

    mace_classes = mace["classes"]
    ewald_rel = np.array([c["energy_eV"] for c in mace_classes]) - e0
    mace_rel = np.array([c["mace_relaxed_eV"] for c in mace_classes])
    mace_rel -= mace_rel.min()
    # What drives each ranking: the number of nearest-neighbor Li-Fe (unlike)
    # cation pairs in each class's representative ordering.
    import sys

    from scipy.stats import spearmanr

    sys.path.insert(0, str(HERE))
    import run

    parent = run.build_parent()
    cations = run.cation_indices(parent)
    cation_coords = parent.frac_coords[cations]
    nn = np.isclose(
        parent.lattice.get_all_distances(cation_coords, cation_coords), run.LATTICE_PARAMETER / np.sqrt(2), atol=0.05
    )
    enumeration = np.load(RESULTS / "enumeration.npz")
    unlike_pairs = []
    for c in mace_classes:
        li = enumeration["occupations"][np.flatnonzero(enumeration["class_id"] == c["class"])[0]].astype(int)
        unlike_pairs.append(int((nn * np.outer(li, 1 - li)).sum()))
    unlike_pairs = np.array(unlike_pairs)
    print()
    print(f"Unlike (Li-Fe) nearest-neighbor pairs per cell: {unlike_pairs.min()}-{unlike_pairs.max()} of {nn.sum() // 2}")
    print(f"  Spearman vs Ewald energy: {spearmanr(unlike_pairs, ewald_rel).statistic:+.2f}")
    print(f"  Spearman vs MACE relaxed energy: {spearmanr(unlike_pairs, mace_rel).statistic:+.2f}")

    print()
    print("Lowest MACE (relaxed) classes:")
    for k in np.argsort(mace_rel)[:8]:
        c = mace_classes[k]
        print(
            f"  MACE {mace_rel[k]:6.3f} eV   Ewald {ewald_rel[k]:6.2f} eV   g = {c['degeneracy']:4d}"
            f"   unlike pairs {unlike_pairs[k]:2d}   {c['spacegroup']}"
        )

    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax_c.plot(T_grid, np.array(exact["heat_capacity_eV_per_K"]) / KB_EV_PER_K, color=EXACT_COLOR, lw=2, label="Ewald")
    ax_c.plot(
        np.array(mace["temperature_K"]),
        np.array(mace["curves"]["relaxed"]["heat_capacity_eV_per_K"]) / KB_EV_PER_K,
        color=MACE_COLOR,
        lw=2,
        label="MACE (relaxed)",
    )
    ax_c.set_xscale("log")
    ax_c.set_xlim(30, 1e6)
    ax_c.set_xlabel("Temperature (K)")
    ax_c.set_ylabel(r"$C\,/\,k_B$ (per cell)")
    ax_c.set_title("Exact heat capacity", loc="left", color=INK, fontsize=10)
    ax_c.legend(loc="upper left")

    ax_s.scatter(ewald_rel, mace_rel, s=40, color=MACE_COLOR, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for k, label, offset, ha in (
        (int(np.argmin(ewald_rel)), "Ewald ground state", (10, -24), "left"),
        (int(np.argmin(mace_rel)), "MACE ground state", (-10, 4), "right"),
    ):
        ax_s.annotate(
            f"{label}\n{mace_classes[k]['spacegroup']}",
            (ewald_rel[k], mace_rel[k]),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax_s.set_xlabel("Ewald energy above Ewald minimum (eV per cell)")
    ax_s.set_ylabel("MACE energy above MACE minimum (eV per cell)")
    ax_s.set_title(
        f"All {len(mace_classes)} symmetry classes (Spearman {mace['checks']['spearman_ewald_vs_mace_relaxed']:+.2f})",
        loc="left",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "ewald_vs_mace.png", dpi=200)
else:
    print("No mace_exact.json; run `run.py --mace MODEL` to add the MACE comparison.")
