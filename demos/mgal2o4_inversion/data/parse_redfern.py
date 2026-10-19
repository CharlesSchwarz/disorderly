"""Extract the Redfern et al. (1999) MgAl2O4 in situ neutron data from COD CIFs.

Source: S. A. T. Redfern, R. J. Harrison, H. St. C. O'Neill and D. R. R.
Wood, "Thermodynamics and kinetics of cation ordering in MgAl2O4 spinel up
to 1600 C from in situ neutron diffraction", American Mineralogist 84,
299-310 (1999). CIFs from the Crystallography Open Database (entries
9002044-9002139, originally from the American Mineralogist Crystal Structure
Database).

Reads every Redfern CIF in a folder of COD CIFs and writes
``redfern1999.csv`` next to this script with, per entry: COD id,
temperature, heating/cooling leg, sample composition, run number, lattice
parameter a, oxygen parameter u (origin choice 2; ideal 0.25), and inversion
x (Al occupancy of the tetrahedral 8a site). It also copies the
room-temperature CIF used as the demo's starting geometry into ``cifs/``.

Usage::

    python demos/mgal2o4_inversion/data/parse_redfern.py path/to/COD
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
#: Room-temperature structure of the stoichiometric heating run: the demo's starting geometry.
REFERENCE_COD_ID = 9002082


def parse_cif(path: Path) -> dict | None:
    text = path.read_text()
    if "Redfern" not in text:
        return None
    title = " ".join(re.search(r"_publ_section_title\s*;(.*?);", text, re.S).group(1).split())
    sites = {
        label: (float(x), float(occupancy))
        for label, x, occupancy in re.findall(
            r"^(Mg\d|Al\d|O)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)", text, re.M
        )
    }
    x_tet_al = next(occ for label, (x, occ) in sites.items() if label.startswith("Al") and abs(x - 0.125) < 1e-3)
    return {
        "cod_id": int(path.stem),
        "temperature_K": float(re.search(r"_diffrn_ambient_temperature\s+([\d.]+)", text).group(1)),
        "leg": "heating" if "heating" in title else "cooling",
        "sample": "Mg0.99Al2O4" if "Mg.99" in title else "MgAl2O4",
        "a_A": float(re.search(r"_cell_length_a\s+([\d.]+)", text).group(1)),
        "u": sites["O"][0],
        "x": x_tet_al,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cod_folder", type=Path, help="Folder of COD CIF files.")
    args = parser.parse_args()

    rows = sorted(filter(None, (parse_cif(p) for p in args.cod_folder.glob("*.cif"))), key=lambda r: r["cod_id"])
    # Runs are consecutive COD ids sharing a sample and leg.
    run = 0
    for k, row in enumerate(rows):
        if k == 0 or (row["sample"], row["leg"]) != (rows[k - 1]["sample"], rows[k - 1]["leg"]) or (
            row["leg"] == "heating" and row["temperature_K"] < rows[k - 1]["temperature_K"] - 200
        ):
            run += 1
        row["run"] = run

    fields = ["cod_id", "run", "sample", "leg", "temperature_K", "a_A", "u", "x"]
    with open(HERE / "redfern1999.csv", "w", newline="") as f:
        f.write(
            "# Redfern, Harrison, O'Neill & Wood, Am. Mineral. 84, 299-310 (1999); in situ neutron diffraction.\n"
            "# Parsed from COD CIFs 9002044-9002139 by parse_redfern.py. u in origin choice 2 (ideal 0.25);\n"
            "# x = Al occupancy of the tetrahedral site.\n"
        )
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)

    (HERE / "cifs").mkdir(exist_ok=True)
    shutil.copy(args.cod_folder / f"{REFERENCE_COD_ID}.cif", HERE / "cifs" / f"{REFERENCE_COD_ID}.cif")
    print(f"Wrote {len(rows)} entries in {run} runs; copied {REFERENCE_COD_ID}.cif.")


if __name__ == "__main__":
    main()
