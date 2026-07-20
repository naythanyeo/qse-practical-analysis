import gc
from concurrent.futures import ThreadPoolExecutor

from qibochem.measurement.protocol import MultiShotProtocol
from qibochem.selected_ci.qse import QSE_Computable

from script_utils import (
    FERM_QUBIT_MAP,
    MAP_THRESHOLD,
    MOLECULE_NAMES,
    QSE_EXPANSIONS,
    SHOTS_CACHE_DIR,
    SHOTS_OUTPUT_DIR,
    get_vqe_circuit,
    load_qibo_mol,
    save_npz,
)


ACTIVE_SPACES = ["2e2o", "2e3o", "4e3o", "4e4o"]

SHOT_LADDER = (
    1000, 2000, 5000, 10000, 20000,
    50000, 100000, 200000, 500000, 1000000,
)
N_REPEATS = 10

MAX_WORKERS = 4

def run_molecule_expansion(molecule_name, active_space, expansion, excitation_generator, spin_projection):
    output_path = SHOTS_OUTPUT_DIR / active_space / expansion / f"{molecule_name}.npz"
    # Check for done paths 
    if output_path.exists():
        print(f"{molecule_name} {active_space} {expansion}: complete, skipping")
        return

    print(f"\n{active_space} {expansion} {molecule_name}")

    # Load in qibo molecule and VQE circuit from cached / regenerate
    molecule = load_qibo_mol(molecule_name, active_space)
    final_circuit = get_vqe_circuit(molecule, molecule_name, active_space, "UCCSD")
    cache_dir = SHOTS_CACHE_DIR / active_space / expansion / molecule_name

    # Load in QSE computable (from cache if available)
    qse = QSE_Computable(
        molecule=molecule,
        excitation_generator=excitation_generator,
        spin_projection=spin_projection,
        ferm_qubit_map=FERM_QUBIT_MAP,
        map_threshold=MAP_THRESHOLD,
        h_cache_path=str(cache_dir / "H.pkl"),
        s_cache_path=str(cache_dir / "S.pkl"),
    )
    protocol = MultiShotProtocol(sample_sizes=SHOT_LADDER, n_repeats=N_REPEATS)
    H_results, S_results = qse.run_qse(final_circuit, protocol)
    # Check and save number of commuting groups for each group 
    num_commuting_groups = len(protocol.groups)

    # Rename the keys so H and S keys are unique 
    H_data = {f"H_{key}": matrix for key, matrix in H_results.items()}
    S_data = {f"S_{key}": matrix for key, matrix in S_results.items()}

    save_npz(
        output_path,
        molecule=molecule_name,
        active_space=active_space,
        ansatz="UCCSD",
        expansion=expansion,
        num_commuting_groups=num_commuting_groups,
        **H_data,
        **S_data,
    )

    # Clear cache for RAM 
    del qse, molecule, final_circuit, H_results, S_results, H_data, S_data
    gc.collect()


def main(max_workers=MAX_WORKERS):
    for active_space in ACTIVE_SPACES:
        for expansion, (excitation_generator, spin_projection) in QSE_EXPANSIONS.items():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
