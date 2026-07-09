"""General helper functions for notebooks."""

import json

import numpy as np


def parse_active_space(active_space):
    """Parse an active-space label like '4e5o' into (4, 5)."""
    num_e, num_o = active_space.split("e")
    return int(num_e), int(num_o.removesuffix("o"))


def read_jsonl(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def complex_matrix(real, imag):
    return np.array(real) + 1j * np.array(imag)


def get_all_EE(energies):
    ha = 27.211386245981
    all_ee = []
    for i in range(len(energies)-1):
        ee = (energies[i+1] - energies[0])*ha
        all_ee.append(round(ee, 5))
    return (all_ee)
