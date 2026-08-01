"""Chemistry helpers for the publishable QSE analysis notebooks."""

from math import comb
from pathlib import Path

import numpy as np
from pyscf import fci
from pyscf.fci import spin_op
from rdkit import Chem as Ch
import xyz2mol as x2m

from notebook_utils.general import parse_active_space


PUBLISH_ROOT = Path(__file__).resolve().parents[2]
MOLECULE_DIR = PUBLISH_ROOT / "data" / "28_mols"
# Define Chemical Accuracy here in Ha to be imported
CHEMICAL_ACCURACY = 0.00159362

def xyz_2_smiles(fname, CHARGE=0):
    atoms, charge, coord = x2m.read_xyz_file(fname)
    mol = x2m.xyz2mol(atoms, coord, CHARGE)
    return Ch.MolToSmiles(mol[0])

# Define 28 Molecules smiles
smile_28 = {}
for file in MOLECULE_DIR.glob("*.xyz"):
    try:
        smile = xyz_2_smiles(file, CHARGE=0)
        smile_28.update({file.stem: smile})
    except:
        pass  # DS store

molecule_names = smile_28.keys()


def get_vqe_spin_contamination(vqe_state, active_space):
    """
    Helper function that inputs VQE state and ouputs the spin value 
    VQE state here is the FULL vector 2^n NOT pvec
    Builds the CI matrix that PySCF uses
    Conversion must be done between the interleaved orbital notation in qibo to
    the alpha-beta notation that CI matrix uses 
    """
    num_e, num_o = parse_active_space(active_space)
    n_alpha = n_beta = num_e // 2
    ci_matrix = np.zeros((comb(num_o, n_alpha), comb(num_o, n_beta)))

    for index, amplitude in enumerate(np.real(vqe_state)):
        # Based on the index, get the bit string representation
        # Formatted by 2*n_o qubits
        bitstring = format(index, f"0{2 * num_o}b")
        # Alpha and Beta bits are the interleaved bits
        alpha_bits = bitstring[0::2]
        beta_bits = bitstring[1::2]

        # Skip the non spin conserving terms from the FULL vec 
        if alpha_bits.count("1") != n_alpha or beta_bits.count("1") != n_beta:
            continue

        # cistring converts this into the ci matrix index 
        alpha_index = fci.cistring.str2addr(num_o, n_alpha, alpha_bits[::-1])
        beta_index = fci.cistring.str2addr(num_o, n_beta, beta_bits[::-1])
        # Record and account for sign flips due to change in notation
        flips = sum(
            beta_bits[beta_orbital] == "1" and alpha_bits[alpha_orbital] == "1"
            for beta_orbital in range(num_o)
            for alpha_orbital in range(beta_orbital + 1, num_o)
        )
        ci_matrix[alpha_index, beta_index] = (-1) ** flips * amplitude

    spin_value, _ = spin_op.spin_square(ci_matrix, num_o, (n_alpha, n_beta))
    return float(abs(spin_value))
