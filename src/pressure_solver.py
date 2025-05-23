import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres

def compute_pressure_coefficient(face):
    """ Computes the pressure coefficient for the given face. """
    return face.get("diffusion_coefficient", 0)  # Default to 0 if missing

def divergence(U, face):
    """ Computes the divergence term for the pressure equation. """
    velocity = U.get(face["id"], [0.0, 0.0, 0.0])  # Default velocity if missing
    return np.dot(velocity, face.get("normal", [0.0, 0.0, 0.0]))  # Divergence based on velocity direction

def construct_poisson_system(P, mesh):
    """ Constructs the Poisson system for pressure correction. """
    A = lil_matrix((len(mesh.faces), len(mesh.faces)))
    b = np.zeros(len(mesh.faces))

    for face_id, face in mesh.faces.items():
        coeff = compute_pressure_coefficient(face)
        A[face_id, face_id] += coeff
        b[face_id] -= divergence(P, face)

    return A.tocsr(), b

def solve_pressure_correction(P, U, mesh):
    """ Solves the pressure correction equation using GMRES. """
    A, b = construct_poisson_system(P, mesh)
    P_prime, _ = gmres(A, b)  # Multi-grid preconditioning can be applied
    return P_prime



