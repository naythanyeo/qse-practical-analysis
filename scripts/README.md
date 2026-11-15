# Calculation Scripts

These scripts generate the underlying calculation data. They are optional when analysing downloaded datasets. Follow the [environment setup](../README.md) first and retain the molecular inputs and parameter caches described in the [data guide](../data/README.md).

## Available Scripts

Output paths below are relative to `data/`.

| Script | Calculation | Output |
| --- | --- | --- |
| `generate_pyscf_casci_energies.py` | Exact active-space singlet/triplet energies and determinant vectors | `wavefunctions/pyscf_casci/` |
| `run_qse_SV.py` | Statevector projected Hamiltonian and overlap matrices | `raw_matrices/SV_hamiltonian/` |
| `run_qse_spin_SV.py` | Projected spin matrices | `raw_matrices/SV_spin/` |
| `generate_qse_state_matrices.py` | QSE basis states in the determinant basis | `wavefunctions/qse_state_matrices/` |
| `run_qse_shots.py` | Repeated finite-shot Hamiltonian/overlap estimates | `raw_matrices/shots_hamiltonian/` |
| `run_qse_doubles.py` | Singles plus occupied-to-virtual doubles QSE | `raw_matrices/SV_hamiltonian_select_doubles/ov_doubles/<active_space>/` |
| `generate_molecular_orbitals.py` | Frontier canonical orbital cube files | `visualise/<molecule>/` |

`script_utils.py` supplies shared paths, molecule loading, cached VQE circuit reconstruction, and determinant-basis conversion.

## Execution and Configuration

Run scripts from the repository root, for example:

```bash
python scripts/generate_pyscf_casci_energies.py
```

Settings are defined inside the Python files rather than exposed as command-line options. Review active spaces, ansatz mappings, expansion choices, and worker counts before execution. Molecule names and common settings are defined in `script_utils.py`; some scripts override imported settings locally.

For finite-shot calculations, review `SHOT_LADDER` and `N_REPEATS`. Shot counts are per commuting measurement group, per repeat. Fresh sampling need not reproduce the archived realisation exactly.

Current defaults include partial sweeps, not a single configuration that regenerates the complete paper dataset. Large active spaces and parallel workers can require substantial memory and runtime.

The commented alternative in `run_qse_doubles.py` selects the historical frontier-doubles generator and writes to `raw_matrices/SV_hamiltonian_select_doubles/frontier_doubles/<active_space>/`. The runner currently selects UCCSD only for either pool; the archive also preserves historical UCCGSD frontier-doubles data.

## Caches and Existing Files

The molecule loader reuses cached HF data or runs PySCF if the cache is absent. The VQE helper reconstructs circuits from cached parameters or performs optimisation when parameters are missing. Preserve the supplied caches to retain the reference calculations used in the analysis.

QSE measurement scripts cache constructed observables under `data/cache/`. Generation scripts skip existing outputs, checking each file or pending ansatz/expansion as appropriate. File existence is not validation: changing settings does not force existing results to be recomputed. Use a separate output location when testing a different configuration.

The orbital generator currently uses the `8e8o` reference and writes HOMO-3 through LUMO+3 on an 80-by-80-by-80 grid, skipping existing cube files.
