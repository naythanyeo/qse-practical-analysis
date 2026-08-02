"""
General helper functions related to QSE operators.
"""

from qibochem.selected_ci.qse import generate_singlet_singles, generate_triplet_singles

from notebook_utils.general import parse_active_space

def get_qse_operators(active_space):
    """
    Return the singlet and triplet QSE operator vectors given
    an active space. Uses the qibo qse generator function.
    """
    num_electrons, num_orbitals = parse_active_space(active_space)
    excitation_params = {
        "n_elec": num_electrons,
        "n_orbs": 2 * num_orbitals,
        "spin_projection": "all",
    }
    return (generate_singlet_singles(excitation_params), 
            generate_triplet_singles(excitation_params))


def get_operator_labels(operators, active_space):
    """
    For each fermionic operator, returns a category and label 
    Category is used in 02 dimensions analysis to identify the
    eigenvector composition. Also used alongside the labels for 
    sorting in 04 scaling and heatmaps. 
    Categories can be: Number Occupied, OV or Rest 
    Labels are ((source, destination), spin_projection)
    ^ In spatial orbitals 
    """
    num_electrons, _ = parse_active_space(active_space)
    num_occupied = num_electrons // 2
    categories, labels = [], []

    for operator in operators:
        term = next(iter(operator.terms))
        (create_orbital, _), (destroy_orbital, _) = term
        source = destroy_orbital // 2 + 1
        destination = create_orbital // 2 + 1
        spin_projection = 0 if len(operator.terms) > 1 else destroy_orbital % 2 - create_orbital % 2

        if source == destination and source <= num_occupied:
            category = "Number occupied"
        elif source <= num_occupied < destination:
            category = "OV"
        else:
            category = "Rest"

        categories.append(category)
        labels.append(((source, destination), spin_projection))

    return categories, labels