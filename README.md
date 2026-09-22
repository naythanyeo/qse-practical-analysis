# Practical QSE Analysis

This repository contains the calculation scripts, analysis notebooks, and manuscript sources accompanying our study of practical quantum subspace expansion (QSE). Together with the archived datasets, it provides the workflow for reproducing the numerical analyses and figures presented in the paper. Calculations use our [research fork of Qibochem](https://github.com/naythanyeo/qibochem-qse), a quantum chemistry plugin for Qibo.

## Repository Layout

| Directory | Contents |
| --- | --- |
| [data/](data/README.md) | Molecular inputs, cached parameters, and calculation datasets |
| [notebooks/](notebooks/README.md) | Data processing, numerical analysis, and figure generation |
| [scripts/](scripts/README.md) | Scripts for generating calculation data |
| [manuscript/](manuscript/) | Paper sources, tables, and figures |

Each directory has its own README with detailed instructions.


## Set up 

Clone the repository:

```bash
git clone https://github.com/naythanyeo/qse-practical-analysis.git
cd qse-practical-analysis
```

Set up conda environment for dependencies:

```bash
conda env create -f environment.yml
conda activate qse-qibochem
```

## Data source

Download the raw matrices and processed dataframes from [Figshare](https://doi.org/10.6084/m9.figshare.33950287). 
Unzip the folders and place them in the folder [data/](data/README.md).
