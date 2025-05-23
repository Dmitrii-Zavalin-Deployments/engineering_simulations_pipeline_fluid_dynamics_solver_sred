import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres

def divergence(U, face):
    """ Computes divergence for pressure correction. """
    return np.dot(U[face["owner"]], face["normal"]) * face["area"]  # ✅ Ensure correct dictionary access

def construct_poisson_system(P, U, mesh):
    """ Constructs the Poisson system for pressure correction. """
    A = lil_matrix((len(mesh.faces), len(mesh.faces)))  # ✅ References `faces`
    b = np.zeros(len(mesh.faces))  # ✅ References `faces`

    for face_id, face in mesh.faces.items():  # ✅ Correctly iterating through dictionary
        A[face["owner"], face["owner"]] += 1.0  # ✅ Removed diffusion coefficient
        b[face["owner"]] -= divergence(U, face)

    return A.tocsr(), b

def solve_pressure_correction(P, U, mesh):
    """ Solves the pressure correction equation using GMRES. """
    A, b = construct_poisson_system(P, U, mesh)
    P_prime, _ = gmres(A, b)  # Multi-grid preconditioning can be applied
    return P_prime



