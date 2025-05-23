import numpy as np  # ✅ Add this line
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres

def assemble_momentum_matrix(U, P, mesh):
    A = lil_matrix((len(mesh.cells), len(mesh.cells)))
    b = np.zeros(len(mesh.cells))  # ✅ Now `np` is defined

    for face_id, face in mesh.faces.items():
        flux = compute_face_flux(U, face)
        A[face_id, face_id] += flux["ap"]
        A[face_id, face_id - 1] -= flux["anb"]
        b[face_id] -= gradient(P, face)

    return A.tocsr(), b

def solve_momentum_equation(U, P, mesh):
    A, b = assemble_momentum_matrix(U, P, mesh)
    U_star, _ = gmres(A, b)
    return U_star



