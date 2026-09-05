"""Matrix helper functions for notebooks."""

import numpy as np
from numpy import linalg as la
from scipy import linalg as sla


def general_solve(H, S, threshold):
    # Diagonalise S to eigenvalues and eigenvector matrix U
    sigma, U = la.eigh(S)
    # Scale the eigenvector matrix to V but multiplying each eigenvector by 1/sqrt(eigenvalue) 
    scale = np.diag(1/np.sqrt(sigma))
    V = np.matmul(U, scale)
    # Only keep columns with eigenvalues above a certain threshold
    mask = sigma > threshold
    V_reduced = V[:, mask]
    V_reduced_adj = V_reduced.conj().T
    # Get the new transformed H which is also hermitian (V dagger H V)
    H_transformed = np.matmul(np.matmul(V_reduced_adj, H), V_reduced)
    # Solve the regular eigenvalue problem
    solved_energies, wavefunctions = la.eigh(H_transformed)
    return solved_energies, wavefunctions


def threshold_ladder_solve(H, S, thresholds, n_states):
    thresholds = np.asarray(thresholds, dtype=float)
    sigma, U = la.eigh(S)

    base_mask = sigma > thresholds.min()
    sigma = sigma[base_mask]
    V = U[:, base_mask] / np.sqrt(sigma)
    H_transformed = V.conj().T @ H @ V

    energies = np.full((len(thresholds), n_states), np.nan)
    for index, threshold in enumerate(thresholds):
        mask = sigma > threshold
        if np.count_nonzero(mask) < n_states:
            continue

        H_reduced = H_transformed[np.ix_(mask, mask)]
        energies[index] = sla.eigvalsh(H_reduced, subset_by_index=(0, n_states - 1))

    return energies


def dimension_solve(H, S, dim):
    # Diagonalise S to eigenvalues and eigenvector matrix U
    sigma, U = la.eigh(S)
    # Scale the eigenvector matrix to V but multiplying each eigenvector by 1/sqrt(eigenvalue) 
    scale = np.diag(1/np.sqrt(sigma))
    V = np.matmul(U, scale)
    # Only keep columns with eigenvalues above a certain threshold
    V_reduced = V[:, -min(dim, len(V)):]
    V_reduced_adj = V_reduced.conj().T
    # Get the new transformed H which is also hermitian (V dagger H V)
    H_transformed = np.matmul(np.matmul(V_reduced_adj, H), V_reduced)
    # Solve the regular eigenvalue problem
    solved_energies, wavefunctions = la.eigh(H_transformed)
    return solved_energies, wavefunctions


def get_Vr_th(S, threshold):
    sigma, U = la.eigh(S)
    scale = np.diag(1 / np.sqrt(sigma))
    V = np.matmul(U, scale)
    mask = sigma > threshold
    V_r = V[:, mask]
    return V_r


def get_Vr_dim(S, dim):
    sigma, U = la.eigh(S)
    scale = np.diag(1 / np.sqrt(sigma))
    V = np.matmul(U, scale)
    V_r = V[:, -min(dim, len(V)):]
    return V_r


# Solve the eigenvalue problem with threshold and get spin values for roots
def solve_energy_spin_th(H, S, Z, threshold):
    sigma, U = la.eigh(S)
    scale = np.diag(1/np.sqrt(sigma))
    V = U @ scale
    mask = sigma > threshold
    V_r = V[:, mask]
    V_r_adj = V_r.conj().T
    H_transformed = V_r_adj @ H @ V_r 
    energies, orth_vectors = la.eigh(H_transformed)
    # orth_vectors are the vectors in orthogonal basis 
    spins = []
    for i in range(orth_vectors.shape[1]):
        w = orth_vectors[:, i]
        spin_expectation = w.conj().T @ V_r_adj @ Z @ V_r @ w
        denom = w.conj().T @ w
        spin_value = (spin_expectation / denom).real
        spins.append(spin_value)
    # qse_wavefunctions are in the original qse basis
    qse_wavefunctions = V_r @ orth_vectors
    return energies, qse_wavefunctions, spins


# Returns condition number of overlap matrix based on filtered dimensions
def condition_number_dim(S, dimension):
    eigenvalues = np.sort(la.eigvalsh(np.asarray(S, dtype=complex)).real)[::-1]
    eigenvalues = eigenvalues[:dimension]
    return np.max(eigenvalues) / np.min(eigenvalues)
