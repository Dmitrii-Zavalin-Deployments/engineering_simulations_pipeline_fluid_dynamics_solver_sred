from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import gmres

def assemble_momentum_matrix(U, P, mesh):
    A = lil_matrix((len(mesh.cells), len(mesh.cells)))
    b = np.zeros(len(mesh.cells))

    for face in mesh.faces:
        flux = compute_face_flux(U, face)
        A[face.owner, face.owner] += flux.ap
        A[face.owner, face.neighbor] -= flux.anb
        b[face.owner] -= gradient(P, face)

    return A.tocsr(), b

def solve_momentum_equation(U, P, mesh):
    A, b = assemble_momentum_matrix(U, P, mesh)
    U_star, _ = gmres(A, b)
    return U_star



