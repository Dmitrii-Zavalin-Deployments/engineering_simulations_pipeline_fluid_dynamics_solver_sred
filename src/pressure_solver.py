import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres

def compute_pressure_coefficient(face):
    """ Computes the pressure coefficient for a given face. """
    return face.diffusion_coefficient  # Ensure face attribute exists

def divergence(U, face):
    """ Computes divergence for pressure correction. """
    return np.dot(U[face.owner], face.normal) * face.area  # Basic flux computation

def construct_poisson_system(P, U, mesh):
    """ Constructs the Poisson system for pressure correction using faces instead of cells. """
    A = lil_matrix((len(mesh.faces), len(mesh.faces)))  # ✅ Fix: Updated to `faces`
    b = np.zeros(len(mesh.faces))  # ✅ Fix: Updated to `faces`

    for face in mesh.faces:
        coeff = compute_pressure_coefficient(face)
        A[face.owner, face.owner] += coeff
        b[face.owner] -= divergence(U, face)

    return A.tocsr(), b

def solve_pressure_correction(P, U, mesh):
    """ Solves the pressure correction equation using GMRES. """
    A, b = construct_poisson_system(P, U, mesh)
    P_prime, _ = gmres(A, b)  # Multi-grid preconditioning can be applied
    return P_prime



