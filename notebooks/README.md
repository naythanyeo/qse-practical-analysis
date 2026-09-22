# Analysis Notebooks

These notebooks process calculation data, analyse results, and generate figures for the paper. First complete the [environment setup](../README.md) and [data setup](../data/README.md).

## Running the Notebooks

From the repository root, with the project environment activated:

```bash
cd notebooks
jupyter lab
```

Select a Python kernel from the project environment. Run cells in order within each notebook; local imports expect `notebooks/` as the working directory. `config.py` defines dataset and figure paths relative to the repository.

## Notebook Guide

| Notebook | Purpose | Main inputs |
| --- | --- | --- |
| [00_data_processing](00_data_processing.ipynb) | Build processed analysis dataframes | Raw statevector, spin, CASCI, finite-shot, and OV-doubles datasets |
| [01_subspace_representability](01_subspace_representability.ipynb) | Representability, ansatz comparisons, and spin diagnostics | `sv_qse_data.pkl`, QSE state matrices, and molecular/parameter inputs |
| [02_dimensions_analysis](02_dimensions_analysis.ipynb) | Overlap spectra, eigenvector composition, and truncation comparisons | `sv_qse_data.pkl` |
| [03_shots_data](03_shots_data.ipynb) | Finite-shot accuracy, threshold selection, and noise scaling | `shots_qse_data.pkl` |
| [04_scaling_heatmaps](04_scaling_heatmaps.ipynb) | Matrix structure and scaling relationships | Statevector and finite-shot dataframes |
| [05_spare_calculations](05_spare_calculations.ipynb) | Supplementary ansatz-complexity calculations | Qibochem research fork |
| [06_molecular_orbitals](06_molecular_orbitals.ipynb) | CASCI excitation composition and orbital-window analysis | Raw CASCI files, `Orbital_Character.csv`, and notebook 01's error CSV |
| [07_select_doubles](07_select_doubles.ipynb) | Compare singles-only and OV-doubles QSE | Statevector and OV-doubles dataframes |

Downloaded processed dataframes let you skip notebook 00. The remaining notebooks are not a strictly sequential pipeline. However, run notebook 01's UCCSD error export before notebook 06's orbital-window examples if `uccsd_low_lying_state_errors.csv` is absent.

Notebook 01 also contains circuit reconstruction and a separate VQE Trotter-step study. The optimisation sweep is more expensive than analysing stored matrices and is not required for the earlier representability results. Notebook 03's threshold scans can also take appreciable time and memory. Review these sections before choosing to run all cells.

## Outputs and Helpers

`notebook_utils/` contains shared graphing, matrix-solving, operator, and chemistry helpers. Notebook-local functions prepare data for each analysis.

The shared `save_figure(...)` helper writes PDF figures into analysis-specific subdirectories of `../manuscript/figures/`. Rerunning a figure cell replaces its existing output. Numerical tables appear in notebook outputs; selected results are exported to `../data/processed_dataframes/` as CSV files.