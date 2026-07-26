#!/usr/bin/env python
"""Generate compact PySCF spin-resolved CASCI reference energies."""

import math

import numpy as np
from pyscf import mcscf
from pyscf.fci import spin_op

from script_utils import (
    MOLECULE_NAMES,
    PYSCF_OUTPUT_DIR,
    load_pyscf_mol,
    mat2pvec,
    parse_active_space,
    save_npz,
)

N_ROOTS = 10

ACTIVE_SPACES = ["2e2o", "2e3o", "4e3o", "4e4o", "4e5o", "6e5o", "6e6o"]
"""
,  "6e7o", "8e7o", "8e8o"
"""

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
                    # Convert CI matrix into partial vector in 2nd quantised det basis
                    "pvec": mat2pvec(ci_matrix, active_space_tuple, m),
                }
            )
            if len(valid_roots) == n_roots:
                return sorted(valid_roots, key=lambda root_data: root_data["energy"])

    return sorted(valid_roots, key=lambda root_data: root_data["energy"])


def run_casci(molecule_name, active_space, max_roots):
    num_electrons, num_orbitals = parse_active_space(active_space)
    active_space_tuple = (num_electrons, num_orbitals)
    n_pairs = num_electrons // 2
    mf = load_pyscf_mol(molecule_name, active_space)

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

    # Save the energies and pvec 
    singlet_energies = np.asarray([root["energy"] for root in singlet_roots], dtype=float)
    singlet_vectors = np.asarray([root["pvec"] for root in singlet_roots], dtype=float)

    triplet_roots_by_sector = {
        "0": triplet_ms0_roots,
        "p1": triplet_p1_roots,
        "m1": triplet_m1_roots
    }
    triplet_root_count = min(len(roots) 
                             for roots in triplet_roots_by_sector.values())

    triplet_energies = np.asarray([
        [triplet_roots_by_sector[sector][i]["energy"] for sector in ["0", "p1", "m1"]]
        for i in range(triplet_root_count)
    ], dtype=float)

    triplet_vectors = np.asarray([
        [triplet_roots_by_sector[sector][i]["pvec"] for sector in ["0", "p1", "m1"]]
        for i in range(triplet_root_count)
    ], dtype=float)
   
    return {
        "singlet": {
            "casci_energies": singlet_energies,
            "casci_pvec": singlet_vectors,
            "sectors": np.asarray(["0"]),
        },
        "triplet": {
            "casci_energies": triplet_energies,
            "casci_pvec": triplet_vectors,
            "sectors": np.asarray(["0", "p1", "m1"]),
        },
    }


def main():
    for molecule_name in MOLECULE_NAMES:
        for active_space in ACTIVE_SPACES:
            # Check all possible paths 
            output_paths = {
                spin_type: PYSCF_OUTPUT_DIR / active_space / spin_type / f"{molecule_name}.npz"
                for spin_type in ["singlet", "triplet"]
            }
            # Check done paths
            pending_spin_types = [spin_type for spin_type, path in output_paths.items() if not path.exists()]

            # Dont run the done paths
            if not pending_spin_types:
                print(f"{molecule_name} {active_space}: complete, skipping")
                continue

            print(f"Processing {molecule_name} {active_space}")
            casci_data = run_casci(molecule_name, active_space, max_roots=N_ROOTS)

            # Save the cached paths
            for spin_type in pending_spin_types:
                save_npz(
                    output_paths[spin_type],
                    molecule=molecule_name,
                    active_space=active_space,
                    spin_type=spin_type,
                    **casci_data[spin_type],
                )


if __name__ == "__main__":
    main()
