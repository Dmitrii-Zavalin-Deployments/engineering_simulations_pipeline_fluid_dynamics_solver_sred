import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres

def compute_pressure_coefficient(face):
    """ Computes the pressure coefficient for the given face. """
    return face.get("diffusion_coefficient", 0)  # Default to 0 if missing

def divergence_face_contribution(U_star, face):
    """ Computes the correct divergence contribution from a single face. """
    owner_id = face["owner_cell"]
    neighbor_id = face.get("neighbor_cell", None)  # Handle missing neighbor

    # Ensure valid owner cell index
    if owner_id < 0 or owner_id >= len(U_star):
        return 0.0  # Invalid index (shouldn't happen if mesh is correctly structured)

    # Default velocity assignment for boundary faces
    U_face = U_star[owner_id]  # Owner cell velocity

    # Apply interpolation for internal faces (neighbor exists)
    if neighbor_id is not None and 0 <= neighbor_id < len(U_star):
        U_face = 0.5 * (U_star[owner_id] + U_star[neighbor_id])  # Linear interpolation

    normal_vector = np.array(face.get("normal", [0.0, 0.0, 0.0]))  # Convert normal to NumPy array
    face_area = face.get("area", 1.0)  # Retrieve face area

    # Compute flux: dot product of velocity and normal, scaled by face area
    flux = np.dot(U_face, normal_vector) * face_area

    return flux  # Correct face-level contribution to divergence

def construct_poisson_system(U_star, mesh):
    """ Constructs the Poisson system for pressure correction using divergence of tentative velocity field. """
    A = lil_matrix((len(mesh.faces), len(mesh.faces)))
    b = np.zeros(len(mesh.faces))

    for face_id, face in mesh.faces.items():
        coeff = compute_pressure_coefficient(face)
        A[face_id, face_id] += coeff
        b[face_id] -= divergence_face_contribution(U_star, face)  # ✅ Fix: Correct velocity interpolation

    return A.tocsr(), b

def solve_pressure_correction(P, U_star, mesh):
    """ Solves the pressure correction equation using GMRES. """
    A, b = construct_poisson_system(U_star, mesh)
    P_prime, _ = gmres(A, b)  # Multi-grid preconditioning can be applied
    return P_prime



