def construct_poisson_system(P, mesh):
    A = lil_matrix((len(mesh.cells), len(mesh.cells)))
    b = np.zeros(len(mesh.cells))

    for face in mesh.faces:
        coeff = compute_pressure_coefficient(face)
        A[face.owner, face.owner] += coeff
        b[face.owner] -= divergence(U, face)

    return A.tocsr(), b

def solve_pressure_correction(P, U, mesh):
    A, b = construct_poisson_system(P, mesh)
    P_prime, _ = gmres(A, b)  # Multi-grid preconditioning can be applied
    return P_prime


