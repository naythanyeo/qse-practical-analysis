import gc
from concurrent.futures import ThreadPoolExecutor

from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.selected_ci.qse import QSE_Computable

from script_utils import (
    ANSATZ_FUNCTIONS,
    FERM_QUBIT_MAP,
    MAP_THRESHOLD,
    MOLECULE_NAMES,
    QSE_EXPANSIONS,
    SPIN_CACHE_DIR,
    SPIN_OUTPUT_DIR,
    get_vqe_circuit,
    load_qibo_mol,
    save_npz,
)


ACTIVE_SPACES = ["6e7o", "8e7o", "8e8o"]
"""
"2e2o", "2e3o", "4e3o", "4e4o", "4e5o", "6e5o", "6e6o", 
"""
from qibochem.ansatz.ucc import (
    Ansatz_UCCSD
)
ANSATZ_FUNCTIONS = {
    "UCCSD": Ansatz_UCCSD,
}
MAX_WORKERS = 2


def run_molecule_expansion(molecule_name, active_space, expansion, excitation_generator, spin_projection):
    # First get all the file paths within the molecule and active space
    output_paths = {
        ansatz_name: SPIN_OUTPUT_DIR / active_space / expansion / f"{molecule_name}_{ansatz_name}.npz"
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
    cache_dir = SPIN_CACHE_DIR / active_space / expansion / molecule_name
    protocol = StateVectorProtocol()

    # Build QSE computable
    qse = QSE_Computable(
        molecule=molecule,
        excitation_generator=excitation_generator,
        observable=molecule.s2_operator(),
        spin_projection=spin_projection,
        ferm_qubit_map=FERM_QUBIT_MAP,
        map_threshold=MAP_THRESHOLD,
        h_cache_path=str(cache_dir / "Z.pkl"),
        s_cache_path=str(cache_dir / "S.pkl"),
    )

    # Run VQE then QSE 
    for ansatz_name in pending_ansatzes:
        print(ansatz_name)

        final_circuit = get_vqe_circuit(molecule, molecule_name, active_space, ansatz_name)
        Z, S = qse.run_qse(final_circuit, protocol)

        # Save the results, path and then metadata
        save_npz(
            output_paths[ansatz_name], 
            molecule=molecule_name, 
            active_space=active_space,
            ansatz=ansatz_name,
            expansion=expansion,
            Z=Z,
            S=S
        )

        # Clear cache for RAM preservation
        del final_circuit, Z, S
        gc.collect()

    del qse, molecule
    gc.collect()


def main():
    for active_space in ACTIVE_SPACES:
        for expansion, (excitation_generator, spin_projection) in QSE_EXPANSIONS.items():
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        run_molecule_expansion, molecule_name, active_space,
                        expansion, excitation_generator, spin_projection
                    )
                    for molecule_name in MOLECULE_NAMES
                ]

                for future in futures:
                    future.result()


if __name__ == "__main__":
    main()
