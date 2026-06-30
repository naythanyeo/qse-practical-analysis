# Data

This folder contains the small tracked inputs and the expected local layout for larger generated QSE datasets.

## Tracked Data

- `28_mols/`: molecule geometries as `.xyz` files.
- `parameters/`: VQE parameter JSONL files used by the analysis notebooks.

## External Data

The full generated QSE matrix datasets are large and should be downloaded separately from:

`[TODO: DOI / archive link]`

After downloading, place them under:

```text
data/matrices/
├── SV_hamiltonian/
├── SV_spin/
└── shots_hamiltonian/
```

These matrix files are ignored by git and are meant to be local working data.

## File Contents

Statevector QSE files contain molecule and calculation metadata together with real and imaginary components of the Hamiltonian and overlap matrices.

Shot-based QSE files contain the same matrix data, plus shot-count and repeat/sample metadata.

VQE parameter files contain molecule, active-space, ansatz, VQE energy, parameter names, and optimized parameter values.

## Practical Notes

The statevector matrix files are generally small enough to load directly into memory. The shot-based files can be much larger, so notebooks should stream, filter, or summarize them when possible.
