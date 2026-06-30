"""Project-wide paths for the QSE practical analysis repository."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DATA_DIR = PROJECT_ROOT / "data"
MOLECULE_DIR = DATA_DIR / "28_mols"
PARAMETERS_DIR = DATA_DIR / "parameters"
RAW_MATRICES_DIR = DATA_DIR / "raw_matrices"

PROCESSED_DATAFRAMES_DIR = DATA_DIR / "processed_dataframes"
