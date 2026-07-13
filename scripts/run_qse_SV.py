import gc
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MOLECULE_DIR = DATA_DIR / "28_mols"
PARAMETERS_DIR = DATA_DIR / "parameters"
OUTPUT_DIR = DATA_DIR / "raw_matrices" / "canonical" / "SV_hamiltonian"
CACHE_DIR = DATA_DIR / "cache" / "canonical" / "qse_hamiltonian"

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_dir = DATA_DIR / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)

from qibochem.ansatz.ucc import (
    Ansatz_UCCGSD,
    Ansatz_UCCSD,
    Ansatz_UCCSDSinglet,
    Ansatz_kUpCCGSDSinglet,
)
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.selected_ci.qse import QSE_Computable, generate_triplet_singles

from script_utils import (
    get_vqe_circuit,
    load_molecule,
    parameter_file_for_active_space,
    parse_active_space,
    read_jsonl,
    save_sv_qse_record,
)


ACTIVE_SPACES = ["2e2o", "2e3o", "4e3o", "4e4o", "4e5o", "6e5o", "6e6o"]
MOLECULE_NAMES = [
    "Acetamide", "Acetone", "Adenine", "Benzene", "Benzoquinone",
    "Butadiene", "Cyclopentadiene", "Cyclopropene", "Cytosine", "Ethene",
    "Formaldehyde", "Formamide", "Furan", "Hexatriene", "Imidazole",
    "Naphthalene", "Norbornadiene", "Octatetraene", "Propanamide", "Pyrazine",
    "Pyridazine", "Pyridine", "Pyrimidine", "Pyrrole", "Tetrazine",
    "Thymine", "Triazine", "Uracil",
]

FERM_QUBIT_MAP = "jw"
MAP_THRESHOLD = 1e-12

ANSATZ_FUNCTIONS = {
    "1UpCCGSDSinglet": lambda molecule, **kwargs: Ansatz_kUpCCGSDSinglet(molecule, k=1, **kwargs),
    "UCCSDSinglet": Ansatz_UCCSDSinglet,
    "UCCSD": Ansatz_UCCSD,
    "UCCGSD": Ansatz_UCCGSD,
}


QSE_EXPANSIONS = {"triplet_all": generate_triplet_singles}


def completed_sv_keys(sv_file):
    return {
        (record["molecule"], record["active_space"], record["ansatz"], record["expansion"])
        for record in read_jsonl(sv_file)
    }


def run_molecule_expansion(
    molecule_name,
    active_space,
    num_active_e,
    num_active_o,
    expansion,
    excitation_generator,
    sv_file,
    vqe_params_file,
    sv_done,
):
    molecule_keys = {
        (molecule_name, active_space, ansatz_name, expansion)
        for ansatz_name in ANSATZ_FUNCTIONS
    }
    if molecule_keys.issubset(sv_done[active_space]):
        print(f"{molecule_name} {active_space} {expansion}: all ansatz SV records done, skipping")
        return

    print(f"\n{active_space} {expansion} {molecule_name}")
    protocol = StateVectorProtocol()
    mol = None
    qse = None
    guess_amplitudes = None

    try:
        mol = load_molecule(MOLECULE_DIR / f"{molecule_name}.xyz", num_active_e, num_active_o)
        molecule_cache_dir = CACHE_DIR / active_space / expansion / molecule_name
        qse = QSE_Computable(
            molecule=mol,
            excitation_generator=excitation_generator,
            observable=None,
            spin_projection="all",
            ferm_qubit_map=FERM_QUBIT_MAP,
            map_threshold=MAP_THRESHOLD,
            h_cache_path=str(molecule_cache_dir / "H.pkl"),
            s_cache_path=str(molecule_cache_dir / "S.pkl"),
        )

        for ansatz_name, ansatz_function in ANSATZ_FUNCTIONS.items():
            sv_key = (molecule_name, active_space, ansatz_name, expansion)
            final_circuit, guess_amplitudes = get_vqe_circuit(
                mol, molecule_name, active_space, ansatz_name, ansatz_function,
                vqe_params_file, ferm_qubit_map=FERM_QUBIT_MAP,
                guess_amplitudes=guess_amplitudes,
            )

            if sv_key in sv_done[active_space]:
                print(f"{molecule_name} {active_space} {ansatz_name} {expansion}: SV done, skipping")
                del final_circuit
                gc.collect()
                continue

            h_matrix = None
            s_matrix = None
            try:
                h_matrix, s_matrix = qse.run_qse(final_circuit, protocol)

                if sv_key not in sv_done[active_space]:
                    save_sv_qse_record(
                        sv_file, molecule_name, active_space, ansatz_name,
                        expansion, h_matrix, s_matrix,
                    )
                    sv_done[active_space].add(sv_key)
            finally:
                del final_circuit
                if h_matrix is not None:
                    del h_matrix
                if s_matrix is not None:
                    del s_matrix
                gc.collect()
    finally:
        del qse
        del mol
        del guess_amplitudes
        gc.collect()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sv_done = {
        active_space: completed_sv_keys(OUTPUT_DIR / f"SV_HS_{active_space}.jsonl")
        for active_space in ACTIVE_SPACES
    }

    for active_space in ACTIVE_SPACES:
        all_keys = {
            (molecule_name, active_space, ansatz_name, expansion)
            for molecule_name in MOLECULE_NAMES
            for ansatz_name in ANSATZ_FUNCTIONS
            for expansion in QSE_EXPANSIONS
        }
        if all_keys.issubset(sv_done[active_space]):
            print(f"{active_space}: all SV records done, skipping")
            continue

        num_active_e, num_active_o = parse_active_space(active_space)
        vqe_params_file = parameter_file_for_active_space(PARAMETERS_DIR, active_space)
        sv_file = OUTPUT_DIR / f"SV_HS_{active_space}.jsonl"

        for expansion, excitation_generator in QSE_EXPANSIONS.items():
            for molecule_name in MOLECULE_NAMES:
                molecule_keys = {
                    (molecule_name, active_space, ansatz_name, expansion)
                    for ansatz_name in ANSATZ_FUNCTIONS
                }
                if molecule_keys.issubset(sv_done[active_space]):
                    print(f"{molecule_name} {active_space} {expansion}: all ansatz SV records done, skipping")
                    continue

                run_molecule_expansion(
                    molecule_name, active_space, num_active_e, num_active_o,
                    expansion, excitation_generator, sv_file, vqe_params_file,
                    sv_done,
                )


if __name__ == "__main__":
    main()
