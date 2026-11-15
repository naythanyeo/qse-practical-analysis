# Practical QSE Analysis

This repository contains the calculation scripts, analysis notebooks, and manuscript sources for our study of practical quantum subspace expansion (QSE). The scripts generate the raw calculation data, and notebook 00 converts the raw QSE matrices and associated CASCI references into processed dataframes. Users can reproduce the analyses from the supplied processed datasets or rebuild those dataframes from the archived raw data.

Calculations use our [research fork of Qibochem](https://github.com/naythanyeo/qibochem-qse), a quantum chemistry plugin for Qibo.

## Repository Layout

| Directory | Contents |
| --- | --- |
| [data/](data/README.md) | Molecular inputs, reference parameters, CSV tables, and downloaded datasets |
| [notebooks/](notebooks/README.md) | Data processing, numerical analysis, and figure generation |
| [scripts/](scripts/README.md) | Scripts for generating the underlying calculation data |
| [manuscript/](manuscript/) | Paper sources, tables, and figures |

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/naythanyeo/qse-practical-analysis.git
cd qse-practical-analysis
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate qse-qibochem
```

## Download the Data

The numerical datasets are archived on [Figshare](https://doi.org/10.6084/m9.figshare.33950287). Molecular geometries, cached reference parameters, and small CSV tables are included in this repository.

| Archive | Contents |
| --- | --- |
| `raw_matrices.zip` | Raw QSE matrix datasets generated directly by the calculation scripts |
| `processed_dataframes.zip` | Three analysis-ready pickle files produced by notebook 00 |
| `wavefunctions.zip` | Exact CASCI data and QSE basis-state matrices, required for both routes |
| `visualise.zip` | Optional molecular-orbital cube files |

### Recommended: Use Processed Data

Download `processed_dataframes.zip` and `wavefunctions.zip`, extract both into `data/`, and start with notebook **01**. Notebook **00** can be skipped.

### Optional: Rebuild Processed Data

Download `raw_matrices.zip` and `wavefunctions.zip`, extract both into `data/`, and run notebook **00** before proceeding to notebook **01** onward. The processed-dataframe download is not required for this route.


Download `visualise.zip` separately for orbital visualisation; it is not required to run the analysis notebooks.

See the [data guide](data/README.md) for dataset details and the [notebook guide](notebooks/README.md) for execution instructions.
