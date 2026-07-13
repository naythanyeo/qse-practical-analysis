import json
import re

import numpy as np

from qibochem.ansatz.ucc_util import params2amplitudes
from qibochem.driver.molecule import Molecule


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as fp:
        return [json.loads(line) for line in fp if line.strip()]


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fp:
        fp.write(json.dumps(record) + "\n")


def parse_active_space(active_space):
    match = re.fullmatch(r"(\d+)e(\d+)o", active_space.lower())
    if match is None:
        raise ValueError("active_space must use the format '<electrons>e<orbitals>o'.")
    return int(match.group(1)), int(match.group(2))


def parameter_file_for_active_space(parameters_dir, active_space):
    _, num_active_o = parse_active_space(active_space)
    path = parameters_dir / f"VQE_Params_{num_active_o}o.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Canonical VQE parameter file not found: {path}")
    return path


def load_molecule(xyz_path, num_active_e, num_active_o):
    mol = Molecule(xyz_file=str(xyz_path), basis="sto-3g")
    mol.run_pyscf()

    active_mo_start = mol.nelec // 2 - num_active_e // 2
    active_mos = list(range(active_mo_start, active_mo_start + num_active_o))
    frozen_mos = [mo for mo in range(mol.nelec // 2) if mo not in active_mos]
    mol.hf_embedding(active=active_mos, frozen=frozen_mos)
    return mol


def get_vqe_circuit(
    mol,
    molecule_name,
    active_space,
    ansatz_name,
    ansatz_function,
    vqe_params_file,
    ferm_qubit_map="jw",
    guess_amplitudes=None,
):
    ansatz = ansatz_function(
        mol,
        ferm_qubit_map=ferm_qubit_map,
        guess_amplitudes=guess_amplitudes,
    )
    param_names = list(ansatz.param_names)
    matching_records = [
        record for record in read_jsonl(vqe_params_file)
        if (
            record["molecule"] == molecule_name
            and record["active_space"] == active_space
            and record["ansatz"] == ansatz_name
        )
    ]

    if len(matching_records) != 1:
        raise ValueError(
            f"Expected one canonical VQE record for "
            f"{molecule_name} {active_space} {ansatz_name} in {vqe_params_file}, "
            f"found {len(matching_records)}."
        )

    record = matching_records[0]
    if record.get("param_names") != param_names:
        raise ValueError(
            f"Canonical parameter names do not match the current ansatz for "
            f"{molecule_name} {active_space} {ansatz_name}."
        )
    if set(record["vqe_params"]) != set(param_names):
        raise ValueError(
            f"Canonical VQE parameter keys do not match param_names for "
            f"{molecule_name} {active_space} {ansatz_name}."
        )

    ansatz._set_params(record["vqe_params"])
    next_guess_amplitudes = params2amplitudes(record["vqe_params"], ansatz.param_excitations)
    return ansatz.circuit.copy(deep=True), next_guess_amplitudes


def save_sv_qse_record(path, molecule, active_space, ansatz, expansion, H, S):
    append_jsonl(
        path,
        {
            "molecule": molecule,
            "active_space": active_space,
            "ansatz": ansatz,
            "expansion": expansion,
            "h_matrix_real": np.real(H).tolist(),
            "h_matrix_imag": np.imag(H).tolist(),
            "s_matrix_real": np.real(S).tolist(),
            "s_matrix_imag": np.imag(S).tolist(),
        },
    )
