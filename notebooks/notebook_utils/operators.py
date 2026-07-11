"""Operator expansion helpers for notebook workflows."""

import math

import numpy as np
from pyscf import fci
from qibochem.selected_ci.qse import generate_singlet_singles, generate_triplet_singles
from scipy.optimize import linear_sum_assignment

from notebook_utils.general import parse_active_space

"""Number of states from Active Space"""
def get_max_states(active_space):
    num_e, num_o = active_space
    num_pairs = num_e // 2

    dim_m0 = math.comb(num_o, num_pairs) * math.comb(num_o, num_pairs)
    dim_m1 = math.comb(num_o, num_pairs + 1) * math.comb(num_o, num_pairs - 1)

    if num_pairs >= 2 and num_pairs + 2 <= num_o:
        dim_m2 = math.comb(num_o, num_pairs + 2) * math.comb(num_o, num_pairs - 2)
    else:
        dim_m2 = 0

    max_singlet_states = dim_m0 - dim_m1
    max_triplet_states = 3 * (dim_m1 - dim_m2)

    num_singlets = min(max_singlet_states, num_o**2)
    num_triplets = min(max_triplet_states, 3 * num_o * (num_o - 1))
    return num_singlets, num_triplets


def get_stable_states(active_space):
    num_e, num_o = active_space
    num_pairs = num_e // 2
    num_v = num_o - num_pairs

    num_singlets = 1 + num_pairs * num_v
    num_triplets = 3 * num_pairs * num_v
    return num_singlets, num_triplets



"""Function for best pairing between two vectors V1 and V2 to compare energies with "skipped" states
In this case V1 is the larger vector. Use V1 to reprsent the exact energy states. 
V2 can be less than or equal to length of V1, used to represent the QSE energy states"""
def get_best_pairing(v1, v2):
    v1, v2 = np.asarray(v1), np.asarray(v2) # Convert both into np arrays
    cost = (v2[:, None] - v1[None, :]) ** 2 # V1 into a row, V2 into a column. Take element wise difference and square it
    # Cost is now a rectangular matrix len(v2) by len(v1)
    # Use scipy library to determine the best indexes of the cost matrix that minimises the overall cost 
    row_ind, col_ind = linear_sum_assignment(cost)
    # From the index from the best pairing
    matches = [None] * len(v1)
    for r, c in zip(row_ind, col_ind):
        matches[c] = r
    best_pairs = []                             
    for i, v1_val in enumerate(v1):        
        if matches[i] is not None:    
            pair = (v1_val, v2[matches[i]]) 
        else:                              
            pair = (v1_val, None) # If there is no good match then the best pair will be (v1, none)   
        best_pairs.append(pair)    
    return best_pairs


OPERATOR_CLASSES = ("Number", "OO", "OV", "VO", "VV")

def classify_qse_operator(operator, num_electrons):
    """
    Classify a one-body QSE operator by its spatial-orbital transition.
    Input fermionic operator
    Output string (representing its class)
    """
    num_occupied = num_electrons // 2
    term = next(iter(operator.terms))
    destination = term[0][0] // 2
    source = term[1][0] // 2

    if destination == source:
        return "Number"

    source_space = "O" if source < num_occupied else "V"
    destination_space = "O" if destination < num_occupied else "V"
    return source_space + destination_space


def get_qse_operator_label(active_space):
    """
    Input an active space 
    Output the singlet and triplet excitation class labels 
    """
    num_electrons, num_orbitals = parse_active_space(active_space)
    excitation_params = {"n_elec": num_electrons, 
                         "n_orbs": 2 * num_orbitals, 
                         "spin_projection": "all"}
    singlet_operators = generate_singlet_singles(excitation_params)
    triplet_operators = generate_triplet_singles(excitation_params)
    singlet_labels = [classify_qse_operator(operator, num_electrons) 
                      for operator in singlet_operators]
    triplet_labels = [classify_qse_operator(operator, num_electrons) 
                      for operator in triplet_operators]
    return singlet_labels, triplet_labels