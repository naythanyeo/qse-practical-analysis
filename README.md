# Practical QSE Analysis

This repository contains the publishable notebooks, helper utilities, data layout, and manuscript sources for practical Quantum Subspace Expansion (QSE) analysis with QiboChem. The workflow is organized around processing generated workstation data into paper-ready datasets, then analyzing QSE dimensions, ansatz choices, finite-shot behavior, and scaling trends.

## Repository Layout

- `notebooks/`: numbered analysis notebooks for the paper workflow.
- `notebooks/notebook_utils/`: shared Python helpers used by the notebooks.
- `data/`: molecule files, VQE parameters, and local slots for generated QSE data.
- `manuscript/`: LaTeX manuscript sources.

## Notebook Workflow

Run the notebooks in order when rebuilding the analysis:

1. `00_data_processing.ipynb`: convert raw/generated workstation outputs into paper-ready formats.
2. `01_dimensions_analysis.ipynb`: analyze QSE dimension, overlap spectra, and retained ranks.
3. `02_ansatz_analysis.ipynb`: compare ansatz-dependent QSE behavior.
4. `03_shots_data.ipynb`: analyze finite-shot and noisy QSE datasets.
5. `04_scaling_heatmaps.ipynb`: generate operator-count, matrix-structure, heatmap, and scaling analyses.

## Environment

The working conda environment for this project is currently `qse-qibochem`. A formal `environment.yml` can be added once the dependency set is stable.

For local development, QiboChem can be installed in editable mode from the local source checkout. For publication or reproduction, this should eventually be replaced with a pinned GitHub branch, tag, or commit.

## Data

Large generated matrix files are not intended to be tracked directly in git. Download the archived raw JSONL data from:

`[TODO: DOI / archive link]`

and place the extracted files under `data/matrices/` using the folder structure described in `data/README.md`.

## Citation

Citation and data-availability details will be added once the manuscript and data archive are finalized.
