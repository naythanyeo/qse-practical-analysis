# Data

This directory combines tracked molecular inputs with externally archived calculation results. Complete the [environment setup](../README.md) before processing data.

## Tracked Data

- `28_mols/`: molecule geometries as `.xyz` files.
- `parameters/hf_data/`: cached canonical orbitals, integrals, and Hartree-Fock metadata in NPZ files.
- `parameters/vqe_params/`: optimised VQE parameters and energies in NPZ files.

## External Data

Download the raw matrices and processed dataframes from [Figshare](https://doi.org/10.6084/m9.figshare.33950287). Preserve their subdirectories when extracting into this expected layout:

```text
data/
  raw_matrices/
    SV_hamiltonian/
    SV_spin/
    pyscf_casci/
    qse_state_matrices/
    shots_hamiltonian/
    SV_hamiltonian_select_doubles/
      ov_doubles/
  processed_dataframes/
    sv_qse_data.pkl
    shots_qse_data.pkl
    sv_qse_ov_doubles_data.pkl
    Orbital_Character.csv
```

## File Contents

Raw NPZ files hold calculation metadata and numerical arrays: projected Hamiltonian/overlap matrices, spin matrices, exact CASCI energies and vectors, or QSE basis-state matrices, depending on the directory.

Finite-shot files also contain sampling information and repeated matrix estimates. Processed pickle files combine records into pandas dataframes. Load pickle files only from trusted sources.

Visualise.zip contains cube files of all the molecular orbitals.

## Rebuilding Processed Data

Run [00_data_processing.ipynb](../notebooks/00_data_processing.ipynb) to regenerate the three pickle files from the corresponding raw datasets. This does not regenerate the underlying calculations, QSE state matrices, or molecular metadata.
