#!/usr/bin/env python
"""Generate compact PySCF spin-resolved CASCI reference energies."""

import json
import math
from pathlib import Path

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.fci import spin_op


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOLECULE_DIR = PROJECT_ROOT / "data" / "28_mols"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw_matrices" / "pyscf_spin_energies.jsonl"
N_ROOTS = 10
CANDIDATE_FACTOR = 3
OVERWRITE = False
MOLECULES = None

ACTIVE_SPACES = [
    "2e2o",
    "2e3o",
    "4e3o",
    "4e4o",
    "4e5o",
    "6e5o",
    "6e6o",
    "6e7o",
    "8e8o",
]


def parse_active_space(label):
    electrons, orbitals = label.removesuffix("o").split("e")
    return int(electrons), int(orbitals)


def sector_dimension(num_orbitals, n_alpha, n_beta):
    return math.comb(num_orbitals, n_alpha) * math.comb(num_orbitals, n_beta)


def run_rhf(xyz_path):
    mol = gto.M(atom=str(xyz_path), basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    return mf


def run_casci_roots(mf, num_orbitals, n_alpha, n_beta, n_roots):
    nelecas = (n_alpha, n_beta)
    n_roots = min(n_roots, sector_dimension(num_orbitals, n_alpha, n_beta))

    mc = mcscf.CASCI(mf, num_orbitals, nelecas)
    mc.verbose = 0
    mc.fcisolver.verbose = 0
    mc.fcisolver.nroots = n_roots
    mc.kernel()

    h1eff, ecore = mc.get_h1eff()
    h2eff = mc.get_h2eff()
    ci_roots = np.asarray(mc.ci).reshape((-1, *np.asarray(mc.ci).shape[-2:]))

    roots = []
    for ci_matrix in ci_roots:
        spin_s2, _ = spin_op.spin_square(ci_matrix, num_orbitals, nelecas)
        energy = mc.fcisolver.energy(h1eff, h2eff, ci_matrix, num_orbitals, nelecas)
        roots.append({"energy": float(energy + ecore), "spin_s2": float(spin_s2)})

    return sorted(roots, key=lambda row: row["energy"])


def spin_filtered_energies(roots, target_s2, max_roots, tolerance=1e-5):
    energies = [
        row["energy"]
        for row in roots
        if abs(row["spin_s2"] - target_s2) <= tolerance
    ]
    return energies[:max_roots]


def compute_spin_energies(xyz_path, active_space, max_roots, candidate_factor):
    num_electrons, num_orbitals = parse_active_space(active_space)
    n_pairs = num_electrons // 2
    candidate_roots = max_roots * candidate_factor
    mf = run_rhf(xyz_path)

    singlet_candidates = run_casci_roots(
        mf,
        num_orbitals,
        n_pairs,
        n_pairs,
        candidate_roots,
    )
    triplet_candidates = run_casci_roots(
        mf,
        num_orbitals,
        n_pairs + 1,
        n_pairs - 1,
        candidate_roots,
    )

    return {
        "singlet": spin_filtered_energies(singlet_candidates, 0.0, max_roots),
        "triplet": spin_filtered_energies(triplet_candidates, 2.0, max_roots),
    }


def write_record(handle, molecule, active_space, spin_type, energies):
    json.dump(
        {
            "molecule": molecule,
            "active_space": active_space,
            "spin_type": spin_type,
            "energies": energies,
        },
        handle,
    )
    handle.write("\n")


def main():
    molecule_paths = sorted(MOLECULE_DIR.glob("*.xyz"))
    if MOLECULES:
        requested = set(MOLECULES)
        molecule_paths = [path for path in molecule_paths if path.stem in requested]

    if OUTPUT_PATH.exists() and not OVERWRITE:
        raise FileExistsError(f"{OUTPUT_PATH} exists. Set OVERWRITE = True to replace it.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as handle:
        for xyz_path in molecule_paths:
            molecule = xyz_path.stem

            for active_space in ACTIVE_SPACES:
                print(f"Processing {molecule} {active_space}")
                spin_energies = compute_spin_energies(
                    xyz_path,
                    active_space,
                    max_roots=N_ROOTS,
                    candidate_factor=CANDIDATE_FACTOR,
                )

                for spin_type, energies in spin_energies.items():
                    write_record(handle, molecule, active_space, spin_type, energies)

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
