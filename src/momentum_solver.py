import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres

def compute_face_flux(U, face):
    """ Computes convective and diffusive fluxes across a face. """
    face_id = face.get("id", None)  # Ensure face has an ID
    if face_id is None:
        print(f"❌ Warning: Face missing 'id' key. Skipping flux calculation.")
        return {"ap": 0, "anb": 0}

    velocity = U.get(face_id, [0.0, 0.0, 0.0])  # Default velocity if missing
    convective_flux = np.dot(velocity, face.get("normal", [0.0, 0.0, 0.0]))  # Dot product with face normal
    diffusive_flux = face.get("diffusion_coefficient", 0) * face.get("gradient", 0)

    return {
        "ap": max(convective_flux, diffusive_flux),  # Ensure numerical stability
        "anb": min(convective_flux, diffusive_flux)
    }

def assemble_momentum_matrix(U, P, mesh):
    """ Constructs sparse momentum matrix for implicit solving. """
    A = lil_matrix((len(mesh.faces), len(mesh.faces)))
    b = np.zeros(len(mesh.faces))

    for face_id, face in mesh.faces.items():
        flux = compute_face_flux(U, face)
        A[face_id, face_id] += flux["ap"]
        A[face_id, max(0, face_id - 1)] -= flux["anb"]  # Prevent negative indices
        b[face_id] -= P.get(face_id, 0)

    return A.tocsr(), b

def solve_momentum_equation(U, P, mesh):
    """ Solves the momentum equation using GMRES. """
    A, b = assemble_momentum_matrix(U, P, mesh)
    U_star, _ = gmres(A, b)
    return U_star



