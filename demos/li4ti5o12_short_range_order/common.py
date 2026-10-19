"""Shared helpers for the Li4Ti5O12 short-range-order demo: the Li4Ti5O12 cell, the Warren-Cowley
short-range order, starting arrangements, and file I/O.

Used by ``run.py`` (Ewald Monte Carlo), ``mace_mc.py`` (MACE Monte Carlo),
``setup_sweep.py`` and ``analysis.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from disorderly import (
    PrecomputedEwald,
    disordered_site_indices,
    random_ordered_configuration,
    structure_from_ordering_key,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RAW = RESULTS / "raw"
CIF = HERE.parents[1] / "tests" / "test_structures" / "LTO_COD_1001098.cif"
#: Default location of the MACE run folders made by ``setup_sweep.py``.
DEFAULT_SWEEP_DIR = HERE / "sweep"

#: Supercell of the primitive spinel cell: 48 16d sites (8 Li, 40 Ti).
SUPERCELL = [2, 2, 3]
#: Number of 16d-16d neighbor shells to report. Shell 3 (5.91 A) is exactly
#: half a supercell vector, so periodic images coincide and its coordination
#: is wrong in this cell; shells 1 (6 neighbors) and 2 (12) are unaffected.
N_SHELLS = 2


def build_parent() -> Structure:
    parent = Structure.from_file(CIF).get_primitive_structure()
    parent.make_supercell(SUPERCELL)
    return parent


class ShortRangeOrder:
    """Warren-Cowley parameters for Li-Li pairs on the 16d sublattice.

    For neighbor shell n, ``alpha_n = (P_n(Li | Li) - c) / (1 - c)``, where
    ``P_n(Li | Li)`` is the probability that a shell-n 16d neighbor of a 16d
    Li is also Li and ``c`` is the Li fraction on 16d. ``alpha_n = 0`` for an
    uncorrelated arrangement in an infinite crystal, ``-c / (1 - c) = -0.2``
    when no shell-n neighbor of any Li is Li, and positive for clustering.
    """

    def __init__(self, parent: Structure, model: PrecomputedEwald, n_shells: int = N_SHELLS):
        self.model = model
        self.sites = np.array(disordered_site_indices(parent))
        coords = parent.frac_coords[self.sites]
        dists = parent.lattice.get_all_distances(coords, coords)
        radii = np.unique(np.round(dists[dists > 0.1], 2))[:n_shells]
        self.radii = radii.tolist()
        self.shells = [np.isclose(dists, r, atol=0.02) for r in radii]
        self.coordination = []
        for shell in self.shells:
            counts = shell.sum(axis=1)
            if not (counts == counts[0]).all():
                raise ValueError("16d sites do not all have the same number of neighbors in a shell.")
            self.coordination.append(int(counts[0]))
        self.n_parent = len(parent)

    def li_occupation(self, structure: Structure) -> np.ndarray:
        """Boolean Li occupation of each 16d site (in parent order)."""
        indices = self.model.reference_indices(structure)
        is_li = np.array([site.specie.symbol == "Li" for site in structure])
        occupation = np.zeros(self.n_parent, dtype=bool)
        occupation[indices[is_li]] = True
        return occupation[self.sites]

    def __call__(self, structure: Structure) -> list[float]:
        li = self.li_occupation(structure).astype(float)
        c = li.mean()
        return [
            float(((li @ shell @ li) / (li.sum() * z) - c) / (1 - c)) for shell, z in zip(self.shells, self.coordination)
        ]


def initial_configuration(parent: Structure, initial: str, seed: int) -> tuple[Structure, str]:
    """Starting arrangement for a MACE Monte Carlo run, and how it was made.

    ``"ordered"`` is the lowest-energy arrangement found by the Ewald runs in
    ``results/sro_summary.json`` (no first-shell Li-Li pairs, alpha_1 = -0.2);
    ``"random"`` is a random arrangement drawn with ``seed``.
    """
    if initial == "ordered":
        summary_path = RESULTS / "sro_summary.json"
        if not summary_path.exists():
            raise SystemExit(f"{summary_path} not found; the ordered start comes from the Ewald runs (run.py).")
        best = min(json.loads(summary_path.read_text())["runs"], key=lambda r: r["lowest_energy_eV"])
        source = f"lowest-energy Ewald arrangement ({best['temperature_K']:.0f} K run): {best['lowest_energy_key']}"
        return structure_from_ordering_key(parent, best["lowest_energy_key"]), source
    if initial == "random":
        return random_ordered_configuration(parent, np.random.default_rng(seed)), f"random arrangement, seed {seed}"
    raise ValueError(f"initial must be 'ordered' or 'random', got {initial!r}.")


def run_name(temperature_K: float, initial: str, seed: int) -> str:
    return f"T{temperature_K:.0f}K_{initial}_seed{seed}"


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file, skipping lines corrupted by a killed job."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)
