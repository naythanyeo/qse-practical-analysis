#!/usr/bin/env python
"""Generate compact PySCF spin-resolved CASCI reference energies."""

import json
import math
from pathlib import Path

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf import fci
from pyscf.fci import spin_op


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOLECULE_DIR = PROJECT_ROOT / "data" / "28_mols"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw_matrices" / "pyscf_casci_energies.jsonl"
N_ROOTS = 10
OVERWRITE = False
MOLECULES = None

ACTIVE_SPACES = ["2e2o", "2e3o", "4e3o", "4e4o", "4e5o", 
                 "6e5o", "6e6o", "6e7o", "8e7o", "8e8o"]

def parse_active_space(label):
    electrons, orbitals = label.removesuffix("o").split("e")
    return int(electrons), int(orbitals)


def mat2pvec(ci_matrix, active_space, m):
    """
    Convert a PySCF CI matrix coefficients into a partial vector of coefficients 
    Total terms is the number of valid determinants (within an active space) 
    Ie, the number of electrons must match the active space. However, determinants
    that are not within the spin sector are part of the vector as a 0 entry.

    Sign flips accounted for because the determinant basis used uses interleaved 
    alpha and beta ordering, while PySCF keeps alpha and beta strings separate 
    Pyscf: a0 a1 a2 ... b0 b1 b2 ...
    Interleaved: a0 b0 a1 b1 a2 b2 ...
    """
    num_active_e, num_active_o = active_space
    num_bits = 2 * num_active_o

    n_alpha = num_active_e // 2 + m
    n_beta = num_active_e // 2 - m

    pvec = []
    for index in range(2**num_bits):
        bitstring = format(index, f"0{num_bits}b")
        if bitstring.count("1") != num_active_e:
            continue

        alpha_bits = bitstring[0::2]
        beta_bits = bitstring[1::2]
        # Check for spin sector 
        if alpha_bits.count("1") != n_alpha or beta_bits.count("1") != n_beta:
            pvec.append(0)
            continue

        alpha_index = fci.cistring.str2addr(num_active_o, n_alpha, alpha_bits[::-1])
        beta_index = fci.cistring.str2addr(num_active_o, n_beta, beta_bits[::-1])

        flips = 0
        # Check for sign flips due to ordering of interleaved notation 
        # Counts the number of times each beta orbital must be moved past alpha orbital 
        for beta_orbital, beta_occ in enumerate(beta_bits):
            if beta_occ == "1":
                for alpha_orbital, alpha_occ in enumerate(alpha_bits):
                    if alpha_occ == "1" and alpha_orbital > beta_orbital:
                        flips += 1

        sign = (-1) ** flips
        pvec.append(sign * float(ci_matrix[alpha_index, beta_index]))

    return pvec


def run_rhf(xyz_path):
    mol = gto.M(atom=str(xyz_path), basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    return mf

def sector_dimension(num_orbitals, n_alpha, n_beta):
    # Number of roots in each spin sector
    return math.comb(num_orbitals, n_alpha) * math.comb(num_orbitals, n_beta)

def get_casci_roots(
    mf,
    active_space_tuple,
    n_alpha,
    n_beta,
    n_roots,
    spin_target,
    m,
    spin_threshold=1e-5,
):
    _, num_orbitals = active_space_tuple
    nelecas = (n_alpha, n_beta)
    # Request extra candidates because spin filtering may discard some roots.
    # Maximally search within the sector dimension 
    n_fci_roots = min(3 * n_roots, sector_dimension(num_orbitals, n_alpha, n_beta))
    # Run CASCI
    mc = mcscf.CASCI(mf, num_orbitals, nelecas)
    mc.verbose = 0
    mc.fcisolver.verbose = 0
    mc.fcisolver.nroots = n_fci_roots
    mc.kernel()

    # Reshape ci_roots into constant 3D array (n_roots, n_alpha, n_beta) 
    ci_roots = np.asarray(mc.ci).reshape((-1, *np.asarray(mc.ci).shape[-2:]))

    valid_roots = []
    for ci_matrix in ci_roots:
        # Check that the spin matches spin_target
        spin_s2, _ = spin_op.spin_square(ci_matrix, num_orbitals, nelecas)
        if abs(spin_s2 - spin_target) <= spin_threshold:
            # Calculate the total energy
            h1eff, ecore = mc.get_h1eff()
            h2eff = mc.get_h2eff()
            energy = mc.fcisolver.energy(h1eff, h2eff, ci_matrix, num_orbitals, nelecas) 
            valid_roots.append(
                {
                    "energy": float(ecore + energy),
                    "pvec": mat2pvec(ci_matrix, active_space_tuple, m),
                }
            )
            if len(valid_roots) == n_roots:
                return sorted(valid_roots, key=lambda root_data: root_data["energy"])

    return sorted(valid_roots, key=lambda root_data: root_data["energy"])


def run_casci(xyz_path, active_space, max_roots):
    num_electrons, num_orbitals = parse_active_space(active_space)
    active_space_tuple = (num_electrons, num_orbitals)
    n_pairs = num_electrons // 2
    mf = run_rhf(xyz_path)

    # Singlets first 
    singlet_roots = get_casci_roots(
        mf,
        active_space_tuple,
        n_alpha=n_pairs,
        n_beta=n_pairs,
        n_roots=max_roots + 1, # Add this so 10 excited + 1 ground state 
        spin_target=0,
        m=0,
        spin_threshold=1e-5,
    )

    # Triplets
    triplet_ms0_roots = get_casci_roots(
        mf,
        active_space_tuple,
        n_alpha=n_pairs,
        n_beta=n_pairs,
        n_roots=max_roots,
        spin_target=2,
        m=0,
        spin_threshold=1e-5,
    )
    
    triplet_p1_roots = get_casci_roots(
        mf,
        active_space_tuple,
        n_alpha=n_pairs + 1,
        n_beta=n_pairs - 1,
        n_roots=max_roots,
        spin_target=2,
        m=1,
        spin_threshold=1e-5,
    )
    
    triplet_m1_roots = get_casci_roots(
        mf,
        active_space_tuple,
        n_alpha=n_pairs - 1,
        n_beta=n_pairs + 1,
        n_roots=max_roots,
        spin_target=2,
        m=-1,
        spin_threshold=1e-5,
    )

    singlet_energies = [root["energy"] for root in singlet_roots]
    singlet_vectors = {
        str(i): root["pvec"]
        for i, root in enumerate(singlet_roots)
    }

    triplet_roots_by_sector = {
        "0": triplet_ms0_roots,
        "p1": triplet_p1_roots,
        "m1": triplet_m1_roots
    }
    triplet_root_count = min(len(roots) 
                             for roots in triplet_roots_by_sector.values())

    triplet_energies = [
        {
            sector: triplet_roots_by_sector[sector][i]["energy"]
            for sector in ["0", "p1", "m1"]
        }
        for i in range(triplet_root_count)
    ]

    triplet_vectors = {
        f"{i}_{sector}": triplet_roots_by_sector[sector][i]["pvec"]
        for i in range(triplet_root_count)
        for sector in ["0", "p1", "m1"]
    }
   
    return {
        "singlet": {
            "casci_energies": singlet_energies,
            "casci_pvec": singlet_vectors
        },
        "triplet_all": {
            "casci_energies": triplet_energies,
            "casci_pvec": triplet_vectors
        }
    }


def write_record(handle, molecule, active_space, spin_type, casci_data):
    json.dump(
        {
            "molecule": molecule,
            "active_space": active_space,
            "spin_type": spin_type,
            "casci_energies": casci_data["casci_energies"],
            "casci_pvec": casci_data["casci_pvec"],
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
                casci_data = run_casci(
                    xyz_path,
                    active_space,
                    max_roots=N_ROOTS,
                )

                for spin_type, spin_data in casci_data.items():
                    write_record(handle, molecule, active_space, spin_type, spin_data)

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
