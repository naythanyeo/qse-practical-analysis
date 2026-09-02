"""General helper functions for notebooks."""

import json

import numpy as np

from config import output_figures_dir, output_tables_dir


def parse_active_space(active_space):
    """Parse an active-space label like '4e5o' into (4, 5)."""
    num_e, num_o = active_space.split("e")
    return int(num_e), int(num_o.removesuffix("o"))


def read_npz(path):
    """
    Read NPZ file and return dictionary
    """
    with np.load(path, allow_pickle=False) as data:
        record = {}

        for key in data.files:
            value = data[key]
            record[key] = value.item() if value.ndim == 0 else value

    return record


def save_table(df, subfolder, title):
    output_path = output_tables_dir / subfolder / f"{title}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return output_path


def save_figure(fig, subfolder, title):
    output_path = output_figures_dir / subfolder / f"{title}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return output_path


def complex_matrix(real, imag):
    return np.array(real) + 1j * np.array(imag)


def get_all_EE(energies):
    ha = 27.211386245981
    all_ee = []
    for i in range(len(energies)-1):
        ee = (energies[i+1] - energies[0])*ha
        all_ee.append(round(ee, 5))
    return (all_ee)
