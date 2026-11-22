"""Compare optimised VQE energies and spin contamination across Trotter step counts."""

import csv
from itertools import product
from math import comb
from time import perf_counter

import numpy as np
from pyscf import fci
from pyscf.fci import spin_op
from qibochem.measurement.protocol import StateVectorProtocol

from script_utils import (
    ANSATZ_FUNCTIONS, DATA_DIR, MOLECULE_NAMES, load_qibo_mol, parse_active_space,
)

BASIS = "sto-3g"
ACTIVE_SPACE = "4e4o"
ANSATZ_NAMES = ("UCCSDSinglet", "UCCSD")
TROTTER_STEPS = (1, 2, 4, 8, 16, 32)
OUTPUT_PATH = DATA_DIR / "table_data" / "vqe_trotter_results.csv"
COLUMNS = (
    "molecule", "basis_set", "active_space", "ansatz", "trotter_steps",
    "num_parameters", "circuit_depth", "vqe_energy_Ha", "spin_contamination",
)


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


def load_results(path):
    results = {}
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(COLUMNS):
                raise ValueError(f"Unexpected CSV header in {path}")
            for row in reader:
                key = (row["molecule"], row["basis_set"], row["active_space"], row["ansatz"], int(row["trotter_steps"]))
                if key in results:
                    raise ValueError(f"Duplicate Trotter result: {key}")
                results[key] = row
    return results


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.csv")
    with temporary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results.values())
    temporary_path.replace(path)


def main():
    results = load_results(OUTPUT_PATH)
    for name in MOLECULE_NAMES:
        pending = [
            (ansatz_name, steps) for ansatz_name, steps in product(ANSATZ_NAMES, TROTTER_STEPS)
            if (name, BASIS, ACTIVE_SPACE, ansatz_name, steps) not in results
        ]
        if not pending:
            continue

        molecule = load_qibo_mol(name, ACTIVE_SPACE, BASIS)
        for ansatz_name, steps in pending:
            start = perf_counter()
            ansatz = ANSATZ_FUNCTIONS[ansatz_name](molecule, trotter_steps=steps)
            energy, _, circuit = ansatz.run_vqe(StateVectorProtocol(), method="L-BFGS-B", fast=True)
            spin = get_vqe_spin_contamination(circuit().state(), ACTIVE_SPACE)
            key = (name, BASIS, ACTIVE_SPACE, ansatz_name, steps)
            results[key] = dict(zip(COLUMNS, (*key, len(ansatz.param_names), circuit.depth, float(energy), spin), strict=True))
            save_results(OUTPUT_PATH, results)
            print(f"{name} {ansatz_name}, {steps} steps: saved ({perf_counter() - start:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
