# src/numerical_methods/matrix_assembly.py

import numpy as np
from scipy.sparse import lil_matrix

def assemble_poisson_matrix(nx, ny, nz, dx, dy, dz, bc):
    """
    Assembles the sparse matrix A for the discrete Poisson equation ∇²φ = b
    with boundary conditions and periodic support.

    Returns:
        A (scipy.sparse.csr_matrix): System matrix A in CSR format.
    """
    N = nx * ny * nz
    A = lil_matrix((N, N))

    C0 = -2.0 / dx**2 - 2.0 / dy**2 - 2.0 / dz**2
    Cx, Cy, Cz = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2

    def idx(i, j, k):
        return i + j * nx + k * nx * ny

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                center = idx(i, j, k)
                A[center, center] = C0

                # X neighbors
                if i > 0:
                    A[center, idx(i - 1, j, k)] = Cx
                elif bc.get("periodic_x"):
                    A[center, idx(nx - 1, j, k)] = Cx
                elif bc["x_min"]["type"] == "neumann":
                    A[center, center] += Cx

                if i < nx - 1:
                    A[center, idx(i + 1, j, k)] = Cx
                elif bc.get("periodic_x"):
                    A[center, idx(0, j, k)] = Cx
                elif bc["x_max"]["type"] == "neumann":
                    A[center, center] += Cx

                # Y neighbors
                if j > 0:
                    A[center, idx(i, j - 1, k)] = Cy
                elif bc.get("periodic_y"):
                    A[center, idx(i, ny - 1, k)] = Cy
                elif bc["y_min"]["type"] == "neumann":
                    A[center, center] += Cy

                if j < ny - 1:
                    A[center, idx(i, j + 1, k)] = Cy
                elif bc.get("periodic_y"):
                    A[center, idx(i, 0, k)] = Cy
                elif bc["y_max"]["type"] == "neumann":
                    A[center, center] += Cy

                # Z neighbors
                if k > 0:
                    A[center, idx(i, j, k - 1)] = Cz
                elif bc.get("periodic_z"):
                    A[center, idx(i, j, nz - 1)] = Cz
                elif bc["z_min"]["type"] == "neumann":
                    A[center, center] += Cz

                if k < nz - 1:
                    A[center, idx(i, j, k + 1)] = Cz
                elif bc.get("periodic_z"):
                    A[center, idx(i, j, 0)] = Cz
                elif bc["z_max"]["type"] == "neumann":
                    A[center, center] += Cz

    return A.tocsr()


def apply_poisson_rhs_bcs(rhs, nx, ny, nz, dx, dy, dz, bc):
    """
    Modifies the RHS vector b based on Dirichlet boundary conditions.

    Returns:
        Modified rhs vector as NumPy array.
    """
    Cx, Cy, Cz = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2

    def idx(i, j, k):
        return i + j * nx + k * nx * ny

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                center = idx(i, j, k)

                if i == 0 and not bc.get("periodic_x") and bc["x_min"]["type"] == "dirichlet":
                    rhs[center] -= Cx * bc["x_min"]["value"]
                if i == nx - 1 and not bc.get("periodic_x") and bc["x_max"]["type"] == "dirichlet":
                    rhs[center] -= Cx * bc["x_max"]["value"]

                if j == 0 and not bc.get("periodic_y") and bc["y_min"]["type"] == "dirichlet":
                    rhs[center] -= Cy * bc["y_min"]["value"]
                if j == ny - 1 and not bc.get("periodic_y") and bc["y_max"]["type"] == "dirichlet":
                    rhs[center] -= Cy * bc["y_max"]["value"]

                if k == 0 and not bc.get("periodic_z") and bc["z_min"]["type"] == "dirichlet":
                    rhs[center] -= Cz * bc["z_min"]["value"]
                if k == nz - 1 and not bc.get("periodic_z") and bc["z_max"]["type"] == "dirichlet":
                    rhs[center] -= Cz * bc["z_max"]["value"]

    return rhs



