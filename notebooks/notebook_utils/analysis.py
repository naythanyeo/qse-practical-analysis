"""
Helper functions for analysis done in the notebook 
"""

import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_max_states(active_space):
    """
    This is the max stable states for a given active space
    If the HOMO and LUMO +-2 are not present, then the number of available
    physical roots are less than the full QSE dimensions, and will be removed

    For active spaces with the HOMO LUMO +-2 gap present, ie 4e4o and above, then
    the max states is M^2 for singlets and 3*M(M-1) for triplets. For triplets the
    stable dimensions are lesser because for paired ansatz, there is totally no
    support for the number operator. So for consistency, this is used as a stable
    dimension. 

    These dimensions are stable within numerical precision. 

    First part checks all the possible roots within each spin sector
    """
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
    """
    The stable states is defined to be a lot more restrictive 
    Cut down to 1+n*v for singlets, and 3*n*v for triplets 
    Empiracally this corresponds to the number / OV stable transitions with large
    condition numbers. 
    """
    num_e, num_o = active_space
    num_pairs = num_e // 2
    num_v = num_o - num_pairs

    num_singlets = 1 + num_pairs * num_v
    num_triplets = 3 * num_pairs * num_v
    return num_singlets, num_triplets



def get_best_pairing(exact_energies, qse_energies, unmatched_penalty):
    """
    Helper function to define consistently the best match between exact and qse
    energies. This is required because the energies sometimes skip states, so this
    allows for a more fair comparison compared to just matching by indices. 

    This matching allows for terms to remain unmatched (ie we note that some states
    from QSE are skipped), so that remaining terms are not penalised unfairly 

    The unmatched penalty will be defined as chemical accuracy. Physically this 
    means that an unmatched state will be penalised the same if the error is 
    out of the chemically accurate range. 

    First build a column and row vector, then take the difference (builds a 2D
    matrix). Add on the unmatched penalty cost as a matrix equal dimensions to it
    that will "extend" the terms. in the example below, 3 penalty terms because
    at most there can be 3 unmatched roots --> ie this means that its so bad that
    none of the exact roots are matched 
        Q0 Q1 Pen Pen Pen
    E0  
    E1
    E2
    """
    exact_energies = np.asarray(exact_energies)
    qse_energies = np.asarray(qse_energies)

    # Exact column vector, qse row vector 
    energy_costs = np.abs(exact_energies[:, None] - qse_energies[None, :])
    unmatched_costs = np.full((len(exact_energies), len(exact_energies)), unmatched_penalty)
    cost_matrix = np.column_stack([energy_costs, unmatched_costs])

    # Use scipy to optimise this matrix indices picking 
    _, assignments = linear_sum_assignment(cost_matrix)
    # If the indices is more than len(qse), means that penalty was the best match
    # IE that root is unmatched, so return None
    matched_energies = [qse_energies[index] 
                        if index < len(qse_energies) else None 
                        for index in assignments]
    return list(zip(exact_energies, matched_energies))
