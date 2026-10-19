"""Shared helpers for the MgAl2O4 inversion demo: the MgAl2O4 cell, configuration keys, and file I/O.

In normal spinel MgAl2O4, Mg occupies the tetrahedral (8a) sites and Al the
octahedral (16d) sites. An ordered configuration is written as a *key*: one
character per cation site, ``"M"`` for Mg or ``"A"`` for Al, tetrahedral sites
first. The inversion x is the fraction of tetrahedral sites holding Al.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RAW = RESULTS / "raw"
#: Default location of the run folders made by ``setup_sweep.py``.
DEFAULT_SWEEP_DIR = HERE / "sweep"
#: Room-temperature structure (Redfern et al. 1999, COD 9002082) used as the starting geometry.
REFERENCE_CIF = DATA / "cifs" / "9002082.cif"
OXIDATION_STATES = {"Mg": 2, "Al": 3, "O": -2}


def load_parent(path: Path) -> Structure:
    parent = Structure.from_file(path)
    parent.add_oxidation_state_by_element(OXIDATION_STATES)
    return parent


def cation_sites(parent: Structure) -> tuple[list[int], list[int]]:
    """Tetrahedral (Mg-majority) and octahedral (Al-majority) site indices."""
    tet, octa = [], []
    for i, site in enumerate(parent):
        amounts = site.species.get_el_amt_dict()
        if "O" in amounts:
            continue
        (tet if amounts.get("Mg", 0) > amounts.get("Al", 0) else octa).append(i)
    return tet, octa


def decorate(parent: Structure, tet: list[int], octa: list[int], al_tet=(), mg_oct=()) -> Structure:
    """Ordered configuration: normal spinel except Al on ``al_tet`` and Mg on ``mg_oct``."""
    structure = parent.copy()
    al_tet, mg_oct = set(al_tet), set(mg_oct)
    for i in tet:
        structure.replace(i, "Al3+" if i in al_tet else "Mg2+")
    for i in octa:
        structure.replace(i, "Mg2+" if i in mg_oct else "Al3+")
    return structure


def build_cell() -> tuple[Structure, list[int], list[int]]:
    """The 112-atom (16 formula unit) cell: a 2x2x2 supercell of the primitive
    reference cell, with its tetrahedral and octahedral site indices."""
    parent = load_parent(REFERENCE_CIF).get_primitive_structure()
    parent.make_supercell([2, 2, 2])
    tet, octa = cation_sites(parent)
    return parent, tet, octa


def configuration_key(structure: Structure, cation_sites: list[int]) -> str:
    """'M'/'A' for Mg/Al on each cation site, tetrahedral sites first."""
    return "".join("M" if structure[i].specie.symbol == "Mg" else "A" for i in cation_sites)


def structure_from_key(parent: Structure, tet: list[int], octa: list[int], key: str) -> Structure:
    sites = tet + octa
    al_tet = [i for i, c in zip(sites, key) if i in tet and c == "A"]
    mg_oct = [i for i, c in zip(sites, key) if i in octa and c == "M"]
    return decorate(parent, tet, octa, al_tet, mg_oct)


def initial_key(tet: list[int], octa: list[int], initial: str, seed: int) -> str:
    """Normal spinel (x = 0), or a random arrangement (x ~ 2/3) drawn with ``seed``."""
    if initial == "normal":
        return "M" * len(tet) + "A" * len(octa)
    rng = np.random.default_rng(seed)
    labels = np.array(["M"] * len(tet) + ["A"] * len(octa))
    return "".join(rng.permutation(labels))


def run_name(temperature_K: float, initial: str, seed: int) -> str:
    return f"T{temperature_K:.0f}K_{initial}_seed{seed}"


def read_redfern() -> list[dict]:
    """Redfern et al. (1999) in situ data from ``data/redfern1999.csv``."""
    with open(DATA / "redfern1999.csv") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    for row in rows:
        row["cod_id"] = int(row["cod_id"])
        row["run"] = int(row["run"])
        for key in ("temperature_K", "a_A", "u", "x"):
            row[key] = float(row[key])
    return rows


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
