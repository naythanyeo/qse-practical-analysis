import re
from functools import partial
from pathlib import Path

import numpy as np

from qibochem.ansatz.ucc import (
    Ansatz_UCCGSD,
    Ansatz_UCCSD,
    Ansatz_UCCSDSinglet,
    Ansatz_kUpCCGSDSinglet,
)
from qibochem.ansatz.ucc_util import params2amplitudes
from qibochem.driver.molecule import Molecule
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.selected_ci.qse import generate_singlet_singles, generate_triplet_singles

from pyscf import fci, gto, scf

"""
0. INITIALISATION
"""

# PROJECT ROOTS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MOLECULE_DIR = DATA_DIR / "28_mols"
HF_DATA_DIR = DATA_DIR / "parameters" / "hf_data"
VQE_PARAMS_DIR = DATA_DIR / "parameters" / "vqe_params"
SV_OUTPUT_DIR = DATA_DIR / "raw_matrices" / "SV_hamiltonian"
SV_CACHE_DIR = DATA_DIR / "cache" / "qse_hamiltonian"
SPIN_OUTPUT_DIR = DATA_DIR / "raw_matrices" / "SV_spin"
SPIN_CACHE_DIR = DATA_DIR / "cache" / "qse_spin"
PYSCF_OUTPUT_DIR = DATA_DIR / "raw_matrices" / "pyscf_casci"
SHOTS_OUTPUT_DIR = DATA_DIR / "raw_matrices" / "shots_hamiltonian"
SHOTS_CACHE_DIR = DATA_DIR / "cache" / "qse_hamiltonian"

# VARIABLES & SETTINGS
ANSATZ_FUNCTIONS = {
    "1UpCCGSDSinglet": partial(Ansatz_kUpCCGSDSinglet, k=1),
    "2UpCCGSDSinglet": partial(Ansatz_kUpCCGSDSinglet, k=2),
    "3UpCCGSDSinglet": partial(Ansatz_kUpCCGSDSinglet, k=3),
    "UCCSDSinglet": Ansatz_UCCSDSinglet,
    "UCCSD": Ansatz_UCCSD,
    "UCCGSD": Ansatz_UCCGSD,
}

MOLECULE_NAMES = [
    "Acetamide", "Acetone", "Adenine", "Benzene", "Benzoquinone",
    "Butadiene", "Cyclopentadiene", "Cyclopropene", "Cytosine", "Ethene",
    "Formaldehyde", "Formamide", "Furan", "Hexatriene", "Imidazole",
    "Naphthalene", "Norbornadiene", "Octatetraene", "Propanamide", "Pyrazine",
    "Pyridazine", "Pyridine", "Pyrimidine", "Pyrrole", "Tetrazine",
    "Thymine", "Triazine", "Uracil"
]

FERM_QUBIT_MAP = "jw"
MAP_THRESHOLD = 1e-12

QSE_EXPANSIONS = {
    "singlet": (generate_singlet_singles, 0),
    "triplet": (generate_triplet_singles, "all"),
}


"""
1. FILE UTILITY
"""

def parse_active_space(active_space):
    """
    Help extract active space information
    """
    match = re.fullmatch(r"(\d+)e(\d+)o", active_space.lower())
    if match is None:
        raise ValueError("active_space must use the format '<electrons>e<orbitals>o'.")
    return int(match.group(1)), int(match.group(2))


def load_npz(path):
    """
    Load in NPZ data from file
    """
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_npz(path, **arrays):
    """
    Save NPZ file to a path
    Save the relevant metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(path)


"""
2. HARTREE FOCK HELPERS
"""

def _run_pyscf(xyz_path, num_active_e, num_active_o):
    """
    Run PYSCF helper function
    Returns a qibo molecule object
    """
    molecule = Molecule(xyz_file=str(xyz_path), basis="sto-3g")
    molecule.run_pyscf()

    active_start = molecule.nelec // 2 - num_active_e // 2
    active_orbitals = list(range(active_start, active_start + num_active_o))
    frozen_orbitals = [orbital for orbital in range(molecule.nelec // 2) if orbital not in active_orbitals]
    molecule.hf_embedding(active=active_orbitals, frozen=frozen_orbitals)
    return molecule


def load_qibo_mol(molecule_name, active_space):
    """
    Function used by scripts to get canonical MOs
    """
    num_active_e, num_active_o = parse_active_space(active_space)
    xyz_path = MOLECULE_DIR / f"{molecule_name}.xyz"
    hf_data_path = HF_DATA_DIR / active_space / f"{molecule_name}.npz"

    # If inside, then run own pyscf and save it
    if not hf_data_path.exists():
        molecule = _run_pyscf(xyz_path, num_active_e, num_active_o)
        save_npz(
            hf_data_path,
            mo_coeff=molecule.ca,
            mo_energy=molecule.eps,
            active_oei=molecule.embed_oei,
            active_tei=molecule.embed_tei,
            inactive_energy=molecule.inactive_energy,
            hf_energy=molecule.e_hf,
            nuclear_energy=molecule.e_nuc,
            num_electrons=molecule.nelec,
            num_alpha=molecule.nalpha,
            num_beta=molecule.nbeta,
            active_orbitals=molecule.active,
            frozen_orbitals=molecule.frozen,
        )
        return molecule

    # If not load in a canonical qibo molecule object from the cached parameters
    hf_data = load_npz(hf_data_path)
    molecule = Molecule(xyz_file=str(xyz_path), basis="sto-3g")
    molecule.nelec = int(hf_data["num_electrons"])
    molecule.nalpha = int(hf_data["num_alpha"])
    molecule.nbeta = int(hf_data["num_beta"])
    molecule.e_hf = float(hf_data["hf_energy"])
    molecule.e_nuc = float(hf_data["nuclear_energy"])
    molecule.eps = hf_data["mo_energy"]
    molecule.ca = hf_data["mo_coeff"]
    molecule.norb = molecule.ca.shape[1]
    molecule.nso = 2 * molecule.norb
    molecule.active = hf_data["active_orbitals"].astype(int).tolist()
    molecule.frozen = hf_data["frozen_orbitals"].astype(int).tolist()
    molecule.embed_oei = hf_data["active_oei"]
    molecule.embed_tei = hf_data["active_tei"]
    molecule.inactive_energy = float(hf_data["inactive_energy"])
    molecule.n_active_e = num_active_e
    molecule.n_active_orbs = 2 * num_active_o
    return molecule


def load_pyscf_mol(molecule_name, active_space):
    """
    Load a PySCF RHF reference from the cached data
    Loads qibo mol object first so the caches are consistent 
    """
    molecule = load_qibo_mol(molecule_name, active_space)
    pyscf_molecule = gto.M(
        atom=molecule.geometry, basis=molecule.basis, unit="Angstrom",
        charge=molecule.charge, spin=molecule.multiplicity - 1, # Qibo uses 2S+1
        symmetry="C1", verbose=0,
    )

    # Build the PYSCF RHF object, without running mf.kernel() so the orbitals
    # are manually defined based on cached parameters
    mf = scf.RHF(pyscf_molecule)
    mf.mo_coeff = np.asarray(molecule.ca)
    mf.mo_energy = np.asarray(molecule.eps)
    mf.mo_occ = np.zeros_like(mf.mo_energy)
    mf.mo_occ[:molecule.nalpha] = 2
    mf.e_tot = molecule.e_hf
    mf.converged = True
    return mf

"""
3. VQE HELPER
"""

def get_vqe_circuit(molecule, molecule_name, active_space, ansatz_name):
    ansatz_function = ANSATZ_FUNCTIONS[ansatz_name]
    vqe_params_path = VQE_PARAMS_DIR / active_space / molecule_name / f"{ansatz_name}.npz"

    if vqe_params_path.exists():
        vqe_data = load_npz(vqe_params_path)
        final_params = dict(zip(vqe_data["param_names"].tolist(), vqe_data["vqe_params"].astype(float)))
        return ansatz_function(molecule, final_params=final_params).final_circuit

    guess_amplitudes = None
    ansatz_names = list(ANSATZ_FUNCTIONS)
    ansatz_index = ansatz_names.index(ansatz_name)

    if ansatz_index > 0:
        previous_name = ansatz_names[ansatz_index - 1]
        previous_path = VQE_PARAMS_DIR / active_space / molecule_name / f"{previous_name}.npz"

        if previous_path.exists():
            previous_data = load_npz(previous_path)
            previous_params = dict(zip(previous_data["param_names"].tolist(), previous_data["vqe_params"].astype(float)))
            previous_ansatz = ANSATZ_FUNCTIONS[previous_name](molecule, use_mp2_guess=False)
            guess_amplitudes = params2amplitudes(previous_params, previous_ansatz.param_excitations)

    ansatz = ansatz_function(molecule, guess_amplitudes=guess_amplitudes)
    vqe_energy, vqe_params, final_circuit = ansatz.run_vqe(StateVectorProtocol(), method="L-BFGS-B", fast=True)
    save_npz(
        vqe_params_path,
        param_names=np.asarray(list(vqe_params)),
        vqe_params=np.asarray(list(vqe_params.values()), dtype=float),
        vqe_energy=float(vqe_energy),
    )
    return final_circuit

"""
4. PYSCF MATRIX HELPER
"""

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
