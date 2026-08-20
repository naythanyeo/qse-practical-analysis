import gc
from concurrent.futures import ThreadPoolExecutor

from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.selected_ci.qse import (
    QSE_Computable,
    generate_singlet_singles,
)
from qibochem.ansatz.ucc import (
    Ansatz_UCCSD
)

from script_utils import (
    FERM_QUBIT_MAP,
    MAP_THRESHOLD,
    MOLECULE_NAMES,
    SV_CACHE_DIR,
    SV_OUTPUT_DIR,
    get_vqe_circuit,
    load_qibo_mol,
    save_npz,
)

import openfermion

ANSATZ_FUNCTIONS = {"UCCSD": Ansatz_UCCSD}
ACTIVE_SPACES = ["6e6o"]

MAX_WORKERS = 8



def generate_singlet_singles_and_select_doubles(excitation_params: dict):
    single_operators = generate_singlet_singles(excitation_params)
    n_active_elec = excitation_params["n_elec"]

    homo_orb = n_active_elec // 2 - 1 # Index start from 0
    homo1_orb = homo_orb - 1
    lumo_orb = homo_orb + 1 
    lumo1_orb = lumo_orb + 1

    homo_lumo = openfermion.FermionOperator(f"{2*lumo_orb+1}^ {2*lumo_orb}^ {2*homo_orb+1} {2*homo_orb}") 
    
    homo1_lumo = openfermion.FermionOperator(f"{2*lumo_orb+1}^ {2*lumo_orb}^ {2*homo1_orb+1} {2*homo1_orb}") 

    homo_lumo1 = openfermion.FermionOperator(f"{2*lumo1_orb+1}^ {2*lumo1_orb}^ {2*homo_orb+1} {2*homo_orb}")

    homo1_lumo1_a = (
        # alpha-alpha
        openfermion.FermionOperator(
            f"{2*lumo1_orb}^ {2*homo_orb} "
            f"{2*lumo_orb}^ {2*homo1_orb}"
        )
        + # alpha-beta
        openfermion.FermionOperator(
            f"{2*lumo1_orb+1}^ {2*homo_orb+1} "
            f"{2*lumo_orb}^ {2*homo1_orb}"
        ) 
        + # beta-alpha
        openfermion.FermionOperator(
            f"{2*lumo_orb+1}^ {2*homo1_orb+1} "
            f"{2*lumo1_orb}^ {2*homo_orb}"
        ) 
        + # beta-beta
        openfermion.FermionOperator(
            f"{2*lumo1_orb+1}^ {2*homo_orb+1} "
            f"{2*lumo_orb+1}^ {2*homo1_orb+1}"
        ) 
    )

    homo1_lumo1_b = ( 
        # alpha-alpha
        openfermion.FermionOperator(
            f"{2*lumo_orb}^ {2*homo_orb} "
            f"{2*lumo1_orb}^ {2*homo1_orb}"
        )
        + # alpha-beta
        openfermion.FermionOperator(
            f"{2*lumo_orb+1}^ {2*homo_orb+1} "
            f"{2*lumo1_orb}^ {2*homo1_orb}"
        ) 
        + # beta-alpha
        openfermion.FermionOperator(
            f"{2*lumo1_orb+1}^ {2*homo1_orb+1} "
            f"{2*lumo_orb}^ {2*homo_orb}"
        ) 
        + # beta-beta
        openfermion.FermionOperator(
            f"{2*lumo_orb+1}^ {2*homo_orb+1} "
            f"{2*lumo1_orb+1}^ {2*homo1_orb+1}"
        ) 
    )

    all_operators = single_operators + [homo_lumo, homo1_lumo, homo_lumo1, homo1_lumo1_a, homo1_lumo1_b]

    return all_operators

def run_molecule_expansion(molecule_name, active_space, excitation_generator):
    # First get all the file paths within the molecule and active space
    expansion = "select_doubles"
    output_paths = {
        ansatz_name: SV_OUTPUT_DIR / active_space / expansion / f"{molecule_name}_{ansatz_name}.npz"
        for ansatz_name in ANSATZ_FUNCTIONS
    }
    # Retain only the ones that do not yet exist 
    pending_ansatzes = [
        ansatz_name for ansatz_name, output_path in output_paths.items()
        if not output_path.exists()
    ]
    # If all ran already, print line
    if not pending_ansatzes:
        print(f"{molecule_name} {active_space} {expansion}: complete, skipping")
        return

    # Running sequence 
    print(f"\n{active_space} {expansion} {molecule_name}")

    molecule = load_qibo_mol(molecule_name, active_space)
    cache_dir = SV_CACHE_DIR / active_space / expansion / molecule_name
    protocol = StateVectorProtocol()

    # Build QSE computable
    qse = QSE_Computable(
        molecule=molecule,
        excitation_generator=excitation_generator,
        spin_projection=0,
        ferm_qubit_map=FERM_QUBIT_MAP,
        map_threshold=MAP_THRESHOLD,
        h_cache_path=str(cache_dir / "H.pkl"),
        s_cache_path=str(cache_dir / "S.pkl"),
    )

    # Run VQE then QSE 
    for ansatz_name in pending_ansatzes:
        print(ansatz_name)

        final_circuit = get_vqe_circuit(molecule, molecule_name, active_space, ansatz_name)
        H, S = qse.run_qse(final_circuit, protocol)

        # Save the results, path and then metadata
        save_npz(
            output_paths[ansatz_name], 
            molecule=molecule_name, 
            active_space=active_space,
            ansatz=ansatz_name,
            expansion=expansion,
            H=H,
            S=S
        )

        # Clear cache for RAM preservation
        del final_circuit, H, S
        gc.collect()

    del qse, molecule
    gc.collect()


def main():
    for active_space in ACTIVE_SPACES:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    run_molecule_expansion, molecule_name, active_space,
                    generate_singlet_singles_and_select_doubles
                )
                for molecule_name in MOLECULE_NAMES
            ]

            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
