#!/usr/bin/env python
"""Generate optimized VQE states in the fixed-electron determinant basis."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from qibochem.ansatz.ucc import (
    Ansatz_UCCGSD,
    Ansatz_UCCSD,
    Ansatz_UCCSDSinglet,
    Ansatz_kUpCCGSDSinglet,
)
from qibochem.driver.molecule import Molecule
from qibochem.measurement.utils import get_final_state as circuit_final_state
from qibochem.selected_ci.qse import generate_singlet_singles, generate_triplet_singles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MOLECULE_DIR = DATA_DIR / "28_mols"
PARAMETERS_DIR = DATA_DIR / "parameters"
OUTPUT_PATH = DATA_DIR / "processed_states" / "qse_state_matrices.jsonl"

ACTIVE_SPACES = ["2e2o", "2e3o", "4e3o", "4e4o", "4e5o", 
                 "6e5o", "6e6o", "6e7o", "8e8o"]

ANSATZ_FUNCTIONS = {
    "1UpCCGSDSinglet": lambda molecule, **kwargs: Ansatz_kUpCCGSDSinglet(molecule, k=1, **kwargs),
    "UCCSDSinglet": Ansatz_UCCSDSinglet,
    "UCCSD": Ansatz_UCCSD,
    "UCCGSD": Ansatz_UCCGSD,
}


def parse_active_space(active_space):
    match = re.fullmatch(r"(\d+)e(\d+)o", active_space.lower())
    if match is None:
        raise ValueError("active_space must use the format '<electrons>e<orbitals>o'.")
    return int(match.group(1)), int(match.group(2))


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def build_qibochem_molecule(xyz_path, active_space):
    """Load an XYZ file, run PySCF, and apply HOMO-centered active embedding."""
    num_active_e, num_active_o = parse_active_space(active_space)

    molecule = Molecule(
        xyz_file=str(xyz_path),
        basis="sto-3g",
    )
    molecule.run_pyscf()

    active_mo_start = molecule.nelec // 2 - num_active_e // 2
    active_mos = list(range(active_mo_start, active_mo_start + num_active_o))
    frozen_mos = [mo for mo in range(molecule.nelec // 2) if mo not in active_mos]
    molecule.hf_embedding(active=active_mos, frozen=frozen_mos)

    return molecule


def load_vqe_parameters(molecule_name, active_space, ansatz_name):
    """
    Return the cached VQE record if it exists in any of the 
    parameter JSONL file.
    """
    for path in sorted(PARAMETERS_DIR.glob("VQE_Params*.jsonl")):
        for record in read_jsonl(path):
            if (record["molecule"] == molecule_name
                and record["active_space"] == active_space
                and record["ansatz"] == ansatz_name):
                return record["vqe_params"]
    return None


def get_fixed_electron_basis(active_space):
    """
    Return valid determinant indices and bitstrings for an active space.
    The full statevector has length 2**n_spin_orbitals. This function keeps only
    the indices and occupation bitstrings with the active-space electron count.
    """
    num_active_e, num_active_o = parse_active_space(active_space)
    n_spin_orbitals = 2 * num_active_o

    determinant_indices = []
    determinant_bitstrings = []
    for index in range(2**n_spin_orbitals):
        bitstring = format(index, f"0{n_spin_orbitals}b")
        if bitstring.count("1") == num_active_e:
            determinant_indices.append(index)
            determinant_bitstrings.append(bitstring)

    return np.asarray(determinant_indices, dtype=np.int64), determinant_bitstrings

def operator_on_bitstring(fermionic_operator, bitstring):
    """
    This function applies a fermionic operator onto a bistring to help
    construct the partial permuation matrix used to get qse state matrix

    fermionic_operator: FermionOperator object eg (2, 1), (0, 0)) 
    bitstring: represents a determinant eg 1100 

    Output: a list of (target_determinant, coeff)

    We need a list of this because fermioinic operator can have multiple terms
    for spin adapat qse operators. 
    
    In this function though we assume that there is singles
    excitations that are allowed. (Consistent with whatever we used in QSE)
    """
    def destroy(bitstring, orb):
        if bitstring[orb] == "0":
            return None, 0
        sign = (-1) ** bitstring[:orb].count("1")
        new_bitstring = bitstring[:orb] + "0" + bitstring[orb + 1:]
        return new_bitstring, sign
    
    def create(bitstring, orb):
        if bitstring[orb] == "1":
            return None, 0
        sign = (-1) ** bitstring[:orb].count("1")
        new_bitstring = bitstring[:orb] + "1" + bitstring[orb + 1:]
        return new_bitstring, sign

    # Operators act on kets from right to left.
    results = []
    for operators, coeff in fermionic_operator.terms.items():
        (create_orb, _), (destroy_orb, _) = operators
        # Try to destroy 
        after_destroy, destroy_sign = destroy(bitstring, destroy_orb)
        if after_destroy is None:
            continue
        # Try to create
        target, create_sign = create(after_destroy, create_orb)
        if target is None:
            continue
        results.append((target, coeff*destroy_sign*create_sign))
    
    return results

def operator2partial_permu_mat(fermionic_operator, determinant_bitstrings):
    """
    Build a partial permutation matrix for a fermionic operator given a 
    determinant_bitstring basis. Determinant bitstrings will basically be the
    bitstrings basis that pvec uses [0011, 0101 ...] 
    We take in a fermionic operator and build the permutation matrix for that
    so that when we multiply the ground state ansatz pvec by this permutation ish
    matrix, we get the projected vector. 

    This is constructed by taking the operator and acting it on every bitstring, 
    then updating the coefficinets of the matrix based on what the output target
    determinant lands on. Eg if 0011 becomes 1100, then the corresponding row for
    0011 remains 0, and the corresponding row for 1100 becomes 1 (up to a sign)
    """
    # Create the index of each determinant 
    determinant_index = {
        bitstring: index
        for index, bitstring in enumerate(determinant_bitstrings)
    }
    # First create a matrix of zeros 
    matrix = np.zeros(
        (len(determinant_bitstrings), len(determinant_bitstrings)),
        dtype=float,
    )

    for col, source_det in enumerate(determinant_bitstrings):
        # Act the operator on each bitstring  
        results = operator_on_bitstring(fermionic_operator, source_det)
        for target_det, coeff in results:
            row = determinant_index[target_det]
            matrix[row, col] += coeff

    return matrix

def get_qse_operators(active_space):
    """Return QiboChem QSE FermionOperators for singlet or triplet_all."""
    num_active_e, num_active_o = parse_active_space(active_space)
    excitation_params = {"n_elec": num_active_e,
                         "n_orbs": 2 * num_active_o,
                         "spin_projection": "all"}
    singlet_operators = generate_singlet_singles(excitation_params)
    triplet_operators = generate_triplet_singles(excitation_params)
    return singlet_operators, triplet_operators


def get_final_state(molecule, ansatz_name, vqe_params):
    """
    Rebuild an optimized ansatz from cached VQE parameters and return its
    full 2**n statevector.
    """
    ansatz = ANSATZ_FUNCTIONS[ansatz_name](
        molecule,
        final_params=vqe_params,
    )
    return circuit_final_state(ansatz.final_circuit)

def main():
    for active_space in ACTIVE_SPACES:
        # First get the qse operators 
        singlet_operators, triplet_operators = get_qse_operators(active_space)
        determinant_indices , determinant_bitstrings = get_fixed_electron_basis(active_space)
        # Build the qse_matrices 
        singlet_matrices = [operator2partial_permu_mat(operator, determinant_bitstrings)
                            for operator in singlet_operators]
        triplet_matrices = [operator2partial_permu_mat(operator, determinant_bitstrings)
                            for operator in triplet_operators]
        
        for molecule_path in MOLECULE_DIR.glob("*.xyz"):
            molecule = build_qibochem_molecule(molecule_path, active_space)
            molecule_name = molecule_path.stem
            for ansatz_name in ANSATZ_FUNCTIONS.keys():
                vqe_params = load_vqe_parameters(molecule_name, active_space, ansatz_name)
                if vqe_params is None:
                    print(f"Skipping {molecule_name} {active_space} {ansatz_name}: no VQE params")
                    continue

                final_state = get_final_state(molecule, ansatz_name, vqe_params)
                ground_state_pvec = np.real(final_state[determinant_indices])
                singlet_projections = [partial_perm_mat @ ground_state_pvec 
                                       for partial_perm_mat in singlet_matrices]
                triplet_projections = [partial_perm_mat @ ground_state_pvec 
                                       for partial_perm_mat in triplet_matrices]

                singlet_state_matrix = np.column_stack(singlet_projections).tolist()
                triplet_state_matrix = np.column_stack(triplet_projections).tolist()

                append_jsonl(OUTPUT_PATH,
                             {"molecule": molecule_name,
                              "active_space": active_space,
                              "ansatz": ansatz_name,
                              "expansion": "singlet",
                              "qse_state_matrix": singlet_state_matrix})
                append_jsonl(OUTPUT_PATH,
                             {"molecule": molecule_name,
                              "active_space": active_space,
                              "ansatz": ansatz_name,
                              "expansion": "triplet",
                              "qse_state_matrix": triplet_state_matrix})

                print(f"Saved {molecule_name} {active_space} {ansatz_name}")

if __name__ == "__main__":
    main()
