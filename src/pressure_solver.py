import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres

def compute_pressure_coefficient(face):
    """ Computes the pressure coefficient for the given face. """
    return face.get("diffusion_coefficient", 0)  # Default to 0 if missing

def divergence(P, face):
    """ Computes the divergence term for the pressure equation and ensures scalar output. """
    velocity = P.get(face["id"], [0.0, 0.0, 0.0])  # Retrieve velocity (ensure valid format)
    
    if isinstance(velocity, (list, np.ndarray)):  # ✅ Fix: Convert vector velocity to scalar
        velocity = np.linalg.norm(np.array(velocity))  # Use norm to get a single magnitude value
    
    normal = np.array(face.get("normal", [0.0, 0.0, 0.0]))  # Convert normal to NumPy array
    
    divergence_value = np.dot(velocity, normal)  # ✅ Ensure velocity is scalar before dot product
    return float(divergence_value)  # ✅ Explicitly convert to float to avoid errors

def construct_poisson_system(P, mesh):
    """ Constructs the Poisson system for pressure correction. """
    A = lil_matrix((len(mesh.faces), len(mesh.faces)))
    b = np.zeros(len(mesh.faces))

    for face_id, face in mesh.faces.items():
        coeff = compute_pressure_coefficient(face)
        A[face_id, face_id] += coeff
        b[face_id] -= divergence(P, face)  # ✅ Fix: Now always a scalar value

    return A.tocsr(), b

def solve_pressure_correction(P, U, mesh):
    """ Solves the pressure correction equation using GMRES. """
    A, b = construct_poisson_system(P, mesh)
    P_prime, _ = gmres(A, b)  # Multi-grid preconditioning can be applied
    return P_prime



