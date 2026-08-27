"""
General helper functions related to QSE operators.
"""

from functools import lru_cache

import numpy as np

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

def get_excitation_percentages(pvec, hf_state):
    """Return reference, single, double, and higher-excitation percentages."""
    num_bits = len(hf_state)
    num_electrons = hf_state.count("1")
    bitstrings = [
        format(index, f"0{num_bits}b")
        for index in range(2**num_bits)
        if format(index, f"0{num_bits}b").count("1") == num_electrons
    ]

    weights = np.abs(np.asarray(pvec, dtype=complex)) ** 2
    total_weight = weights.sum()

    percentages = np.zeros(4)
    for bitstring, weight in zip(bitstrings, weights / total_weight):
        # Group excitations larger than 3 together
        rank = min(sum(bit != hf_bit for bit, hf_bit in zip(bitstring, hf_state)) // 2, 3)
        percentages[rank] += weight

    return tuple((100 * percentages).tolist())


@lru_cache(maxsize=None)
def _get_determinant_transition_labels(hf_state):
    """Return spatial transition labels in the canonical fixed-electron order."""
    num_bits = len(hf_state)
    num_electrons = hf_state.count("1")
    labels = []

    for index in range(2**num_bits):
        bitstring = format(index, f"0{num_bits}b")
        if bitstring.count("1") != num_electrons:
            continue

        holes = tuple(
            orbital // 2 + 1
            for orbital, (bit, hf_bit) in enumerate(zip(bitstring, hf_state))
            if hf_bit == "1" and bit == "0"
        )
        particles = tuple(
            orbital // 2 + 1
            for orbital, (bit, hf_bit) in enumerate(zip(bitstring, hf_state))
            if hf_bit == "0" and bit == "1"
        )
        rank = len(holes)

        if rank == 0:
            labels.append(("HF", "HF"))
        elif rank == 1:
            labels.append(("Single", f"{holes[0]} -> {particles[0]}"))
        elif rank == 2:
            category = "Paired double" if len(set(holes)) == len(set(particles)) == 1 else "Mixed double"
            hole_label = ",".join(map(str, holes))
            particle_label = ",".join(map(str, particles))
            labels.append((category, f"{hole_label} -> {particle_label}"))
        else:
            hole_label = ",".join(map(str, holes))
            particle_label = ",".join(map(str, particles))
            labels.append(("Higher", f"{hole_label} -> {particle_label}"))

    return tuple(labels)


def get_dominant_exact_transitions(pvecs, hf_state, coverage=0.99):
    """Return dominant channels after averaging normalized pvec sector weights."""
    pvecs = np.atleast_2d(np.asarray(pvecs, dtype=complex))
    labels = _get_determinant_transition_labels(hf_state)
    weights = np.abs(pvecs) ** 2
    weights = (weights / weights.sum(axis=1, keepdims=True)).mean(axis=0)

    channel_weights = {}
    for channel, weight in zip(labels, weights):
        channel_weights[channel] = channel_weights.get(channel, 0.0) + float(weight)

    transitions = []
    cumulative_weight = 0.0
    for (category, label), weight in sorted(channel_weights.items(), key=lambda item: (-item[1], item[0])):
        transitions.append((category, label, 100 * weight))
        cumulative_weight += weight
        if cumulative_weight >= coverage:
            break

    return transitions


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
