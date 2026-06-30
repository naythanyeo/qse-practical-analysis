"""Matrix helper functions for notebooks."""

import numpy as np
from numpy import linalg as la


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

# Spin calculations 
def calc_spin(S, Z, W, threshold):
    sigma, U = la.eigh(S)
    scale = np.diag(1/np.sqrt(sigma))
    V = np.matmul(U, scale)
    mask = sigma > threshold
    V_r = V[:, mask]
    V_r_adj = V_r.conj().T
    
    spins = []
    for i in range(W.shape[1]):
        w = W[:, i]
        spin_expectation = w.conj().T @ V_r_adj @ Z @ V_r @ w
        denom = w.conj().T @ w
        spin_value = (spin_expectation / denom).real
        spins.append(spin_value)
    
    return spins

def condition_number_dim(S, dimension):
    eigenvalues = np.sort(la.eigvalsh(np.asarray(S, dtype=complex)).real)[::-1]
    eigenvalues = eigenvalues[:dimension]
    return np.max(eigenvalues) / np.min(eigenvalues)
