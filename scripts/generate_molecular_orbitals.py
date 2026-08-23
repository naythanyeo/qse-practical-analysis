#!/usr/bin/env python
"""Generate canonical frontier molecular orbitals as cube files."""

import numpy as np
from pyscf.tools import cubegen

from script_utils import DATA_DIR, MOLECULE_NAMES, load_pyscf_mol


ACTIVE_SPACE = "8e8o"
VISUALISE_DIR = DATA_DIR / "visualise"
GRID_POINTS = 80


def get_frontier_orbitals(mf):
    homo = int(np.flatnonzero(mf.mo_occ > 0)[-1])
    return {
        "HOMO-3": homo - 3,
        "HOMO-2": homo - 2,
        "HOMO-1": homo - 1,
        "HOMO": homo,
        "LUMO": homo + 1,
        "LUMO+1": homo + 2,
        "LUMO+2": homo + 3,
        "LUMO+3": homo + 4,
    }


def generate_molecule_orbitals(molecule_name):
    mf = load_pyscf_mol(molecule_name, ACTIVE_SPACE)
    output_dir = VISUALISE_DIR / molecule_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for orbital_label, mo_index in get_frontier_orbitals(mf).items():
        output_path = output_dir / f"{orbital_label}.cube"

        if output_path.exists():
            print(f"{molecule_name} {orbital_label}: complete, skipping")
            continue

        temporary_path = output_path.with_suffix(".tmp.cube")
        cubegen.orbital(
            mf.mol, str(temporary_path), mf.mo_coeff[:, mo_index],
            nx=GRID_POINTS, ny=GRID_POINTS, nz=GRID_POINTS,
        )
        temporary_path.replace(output_path)
        print(f"{molecule_name} {orbital_label}: saved")


def main():
    for molecule_name in MOLECULE_NAMES:
        generate_molecule_orbitals(molecule_name)


if __name__ == "__main__":
    main()
