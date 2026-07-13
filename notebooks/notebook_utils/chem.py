"""Chemistry helpers for the publishable QSE analysis notebooks."""

from pathlib import Path

from rdkit import Chem as Ch
import xyz2mol as x2m


PUBLISH_ROOT = Path(__file__).resolve().parents[2]
MOLECULE_DIR = PUBLISH_ROOT / "data" / "28_mols"


def xyz_2_smiles(fname, CHARGE=0):
    atoms, charge, coord = x2m.read_xyz_file(fname)
    mol = x2m.xyz2mol(atoms, coord, CHARGE)
    return Ch.MolToSmiles(mol[0])


smile_28 = {}
for file in MOLECULE_DIR.glob("*.xyz"):
    try:
        smile = xyz_2_smiles(file, CHARGE=0)
        smile_28.update({file.stem: smile})
    except:
        pass  # DS store

molecule_names = smile_28.keys()

CHEMICAL_ACCURACY = 0.00159362