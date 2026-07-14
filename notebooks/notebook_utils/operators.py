"""
General helper functions related to QSE operators.
"""

import numpy as np
from qibochem.selected_ci.qse import generate_singlet_singles, generate_triplet_singles

from notebook_utils.general import parse_active_space


def get_qse_operator_metadata(operator, active_space):
    """Describe the spatial transition represented by a QSE operator."""
    num_electrons, _ = parse_active_space(active_space)
    num_occupied = num_electrons // 2
    term = next(iter(operator.terms))
    (create_orbital, _), (destroy_orbital, _) = term
    excite_from = destroy_orbital // 2 + 1
    excite_to = create_orbital // 2 + 1

    if excite_from == excite_to:
        operator_class = "Number"
    else:
        source_space = "O" if excite_from <= num_occupied else "V"
        destination_space = "O" if excite_to <= num_occupied else "V"
        operator_class = source_space + destination_space

    return {"class": operator_class, "excite_from": excite_from, "excite_to": excite_to}


def get_qse_operators(active_space):
    """Return the singlet and all-spin triplet QSE operator vectors."""
    num_electrons, num_orbitals = parse_active_space(active_space)
    excitation_params = {
        "n_elec": num_electrons,
        "n_orbs": 2 * num_orbitals,
        "spin_projection": "all",
    }
    return (generate_singlet_singles(excitation_params), 
            generate_triplet_singles(excitation_params))


def get_dimension_operator_labels(operators, active_space):
    """
    Helper function for 02_dimension_analysis
    Return Number occupied, OV, or Rest for each operator.
    """
    num_electrons, _ = parse_active_space(active_space)
    num_occupied = num_electrons // 2
    labels = []

    for operator in operators:
        metadata = get_qse_operator_metadata(operator, active_space)
        if metadata["class"] == "Number" and metadata["excite_from"] <= num_occupied:
            label = "Number occupied"
        elif metadata["class"] == "OV":
            label = "OV"
        else:
            label = "Rest"
        labels.append(label)

    return np.asarray(labels)


def get_heatmap_operator_permutation(operators, active_space):
    """Return a stable Number, OV, VO, OO, VV operator permutation."""
    class_order = {"Number": 0, "OV": 1, "VO": 2, "OO": 3, "VV": 4}
    classes = [get_qse_operator_metadata(operator, active_space)["class"] for operator in operators]
    return np.argsort([class_order[operator_class] for operator_class in classes], kind="stable")
