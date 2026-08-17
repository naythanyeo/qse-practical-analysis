"""Generate optimized VQE states in the fixed-electron determinant basis."""

import numpy as np
from scipy.sparse import csr_matrix

from script_utils import (
    ANSATZ_FUNCTIONS,
    MOLECULE_NAMES,
    QSE_EXPANSIONS,
    QSE_STATE_OUTPUT_DIR,
    get_vqe_circuit,
    load_qibo_mol,
    parse_active_space,
    save_npz,
)

ACTIVE_SPACES = ["2e2o", "2e3o", "4e3o", "4e4o", "4e5o", "6e5o", "6e6o", "6e7o", "8e7o", "8e8o"]
"""
,  "6e7o", "8e7o", "8e8o"
"""
from qibochem.ansatz.ucc import (
    Ansatz_UCCSD
)
ANSATZ_FUNCTIONS = {
    "UCCSD": Ansatz_UCCSD,
}

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
    # Define csr matrix data fields
    row_data, cols_data, matrix_data = [], [], []

    for col, source_det in enumerate(determinant_bitstrings):
        # Act the operator on each bitstring  
        results = operator_on_bitstring(fermionic_operator, source_det)
        for target_det, coeff in results:
            row = determinant_index[target_det]
            row_data.append(row)
            cols_data.append(col)
            matrix_data.append(coeff)

    # Build sparse matrix
    dim = len(determinant_bitstrings)
    matrix = csr_matrix((matrix_data, (row_data, cols_data)), shape=(dim, dim))

    return matrix

def generate_qse_state_matrices(molecule_name, active_space, ansatz_name, determinant_indices, projection_matrices):
    output_paths = {
        expansion: QSE_STATE_OUTPUT_DIR / active_space / expansion / f"{molecule_name}_{ansatz_name}.npz"
        for expansion in QSE_EXPANSIONS
    }
    # Check for done entries
    pending_expansions = [
        expansion for expansion, output_path in output_paths.items()
        if not output_path.exists()
    ]

    if not pending_expansions:
        print(f"{molecule_name} {active_space} {ansatz_name}: complete, skipping")
        return

    # Load in cached canonical molecule and circuit objects
    molecule = load_qibo_mol(molecule_name, active_space)
    final_circuit = get_vqe_circuit(molecule, molecule_name, active_space, ansatz_name)
    # Execute circuit once to get final state
    final_circuit()
    final_state = final_circuit.final_state.state()
    # Only save the correct matching active space terms from the full vec
    ground_state_pvec = np.real(final_state[determinant_indices])

    for expansion in pending_expansions:
        projected_states = [matrix @ ground_state_pvec for matrix in projection_matrices[expansion]]
        qse_state_matrix = np.column_stack(projected_states)

        save_npz(
            output_paths[expansion],
            molecule=molecule_name,
            active_space=active_space,
            ansatz=ansatz_name,
            expansion=expansion,
            qse_state_matrix=qse_state_matrix,
        )

    print(f"Saved {molecule_name} {active_space} {ansatz_name}")


def main():
    for active_space in ACTIVE_SPACES:
        # Save the relevant determinants and bitstrings matching the active space
        determinant_indices, determinant_bitstrings = get_fixed_electron_basis(active_space)
        num_active_e, num_active_o = parse_active_space(active_space)
        projection_matrices = {}

        # Calculate unique projection matrices once
        # Projects the qse vec into 2nd quantised det basis 
        for expansion, (generator, spin_projection) in QSE_EXPANSIONS.items():
            excitation_params = {
                "n_elec": num_active_e,
                "n_orbs": 2 * num_active_o,
                "spin_projection": spin_projection,
            }
            operators = generator(excitation_params)
            projection_matrices[expansion] = [
                operator2partial_permu_mat(operator, determinant_bitstrings)
                for operator in operators
            ]

        # Use the projection matrices 
        for molecule_name in MOLECULE_NAMES:
            for ansatz_name in ANSATZ_FUNCTIONS:
                generate_qse_state_matrices(
                    molecule_name, active_space, ansatz_name,
                    determinant_indices, projection_matrices
                )

if __name__ == "__main__":
    main()
