import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres

def compute_pressure_coefficient(face):
    """ Computes the pressure coefficient for the given face. """
    return face.get("diffusion_coefficient", 0)  # Default to 0 if missing

def divergence_face_contribution(U_star_data, face):
    """ Computes the correct divergence contribution from a single face. """
    velocity_vector = np.array(U_star_data.get(face["id"], [0.0, 0.0, 0.0]))  # Ensure velocity is always an array
    normal_vector = np.array(face.get("normal", [0.0, 0.0, 0.0]))  # Convert normal to NumPy array
    face_area = face.get("area", 1.0)  # Retrieve face area

    # Correct divergence calculation: dot product of velocity and normal, scaled by face area
    flux = np.dot(velocity_vector, normal_vector) * face_area

    return flux  # Face-level contribution to divergence

def construct_poisson_system(U_star, mesh):
    """ Constructs the Poisson system for pressure correction using divergence of tentative velocity field. """
    A = lil_matrix((len(mesh.faces), len(mesh.faces)))
    b = np.zeros(len(mesh.faces))

    for face_id, face in mesh.faces.items():
        coeff = compute_pressure_coefficient(face)
        A[face_id, face_id] += coeff
        b[face_id] -= divergence_face_contribution(U_star, face)  # ✅ Fix: Now correctly uses velocity divergence

    return A.tocsr(), b

def solve_pressure_correction(U_star, mesh):
    """ Solves the pressure correction equation using GMRES. """
    A, b = construct_poisson_system(U_star, mesh)
    P_prime, _ = gmres(A, b)  # Multi-grid preconditioning can be applied
    return P_prime



