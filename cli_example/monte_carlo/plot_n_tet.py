"""Plot the number of tetrahedral Li (``n_tet``) against Monte Carlo step.

Each step's ``properties`` describe the *proposed* configuration, so the
chain's ``n_tet`` is carried forward from the last accepted step.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from disorderly.io import load_trajectory

HERE = Path(__file__).parent
RESULTS = HERE / "cubic_LLZO_parent_T1200K_monte_carlo_results"

trajectory = load_trajectory(RESULTS / "trajectory.json")

steps, n_tet = [], []
current = None
for step in trajectory.steps:
    if (step.accepted or current is None) and "n_tet" in step.properties:
        current = step.properties["n_tet"]
    if current is not None:
        steps.append(step.step)
        n_tet.append(current)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, n_tet, lw=1)
ax.set_xlabel("Monte Carlo step")
ax.set_ylabel("Tetrahedral Li, n_tet")
ax.set_title(f"Cubic LLZO, T = {trajectory.steps[-1].temperature:g} K")
fig.tight_layout()
fig.savefig(RESULTS / "n_tet_vs_step.png", dpi=200)
print(f"Saved {RESULTS / 'n_tet_vs_step.png'}")
