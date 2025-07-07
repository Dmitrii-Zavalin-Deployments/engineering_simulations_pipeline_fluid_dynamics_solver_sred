# src/numerical_methods/matrix_assembly.py

import numpy as np
from scipy.sparse import lil_matrix

def assemble_poisson_matrix(nx, ny, nz, dx, dy, dz, bc):
    """
    Assembles sparse matrix A for discrete Poisson equation ∇²φ = b.
    Supports stencil weights and periodic/neumann boundary encoding.

    Returns:
        A (scipy.sparse.csr_matrix): System matrix A in CSR format.
    """
    N = nx * ny * nz
    A = lil_matrix((N, N))

    Cx, Cy, Cz = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2
    C0 = -2.0 * (Cx + Cy + Cz)

    def idx(i, j, k):
        return i + j * nx + k * nx * ny

    def get_weighted_value(bc_dict, key, base_value):
        entry = bc_dict.get(key, {})
        return entry.get("weight", 1.0) * base_value

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
                    A[center, center] += get_weighted_value(bc, "x_min", Cx)

                if i < nx - 1:
                    A[center, idx(i + 1, j, k)] = Cx
                elif bc.get("periodic_x"):
                    A[center, idx(0, j, k)] = Cx
                elif bc["x_max"]["type"] == "neumann":
                    A[center, center] += get_weighted_value(bc, "x_max", Cx)

                # Y neighbors
                if j > 0:
                    A[center, idx(i, j - 1, k)] = Cy
                elif bc.get("periodic_y"):
                    A[center, idx(i, ny - 1, k)] = Cy
                elif bc["y_min"]["type"] == "neumann":
                    A[center, center] += get_weighted_value(bc, "y_min", Cy)

                if j < ny - 1:
                    A[center, idx(i, j + 1, k)] = Cy
                elif bc.get("periodic_y"):
                    A[center, idx(i, 0, k)] = Cy
                elif bc["y_max"]["type"] == "neumann":
                    A[center, center] += get_weighted_value(bc, "y_max", Cy)

                # Z neighbors
                if k > 0:
                    A[center, idx(i, j, k - 1)] = Cz
                elif bc.get("periodic_z"):
                    A[center, idx(i, j, nz - 1)] = Cz
                elif bc["z_min"]["type"] == "neumann":
                    A[center, center] += get_weighted_value(bc, "z_min", Cz)

                if k < nz - 1:
                    A[center, idx(i, j, k + 1)] = Cz
                elif bc.get("periodic_z"):
                    A[center, idx(i, j, 0)] = Cz
                elif bc["z_max"]["type"] == "neumann":
                    A[center, center] += get_weighted_value(bc, "z_max", Cz)

    return A.tocsr()


def apply_poisson_rhs_bcs(rhs, nx, ny, nz, dx, dy, dz, bc):
    """
    Modifies RHS vector b for Dirichlet boundary conditions.

    Returns:
        np.ndarray: Modified rhs vector
    """
    Cx, Cy, Cz = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2

    def idx(i, j, k):
        return i + j * nx + k * nx * ny

    def get_weighted_value(bc_dict, key, base_value):
        entry = bc_dict.get(key, {})
        return entry.get("weight", 1.0) * base_value

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                center = idx(i, j, k)

                if i == 0 and not bc.get("periodic_x") and bc["x_min"]["type"] == "dirichlet":
                    rhs[center] -= get_weighted_value(bc, "x_min", Cx) * bc["x_min"]["value"]
                if i == nx - 1 and not bc.get("periodic_x") and bc["x_max"]["type"] == "dirichlet":
                    rhs[center] -= get_weighted_value(bc, "x_max", Cx) * bc["x_max"]["value"]

                if j == 0 and not bc.get("periodic_y") and bc["y_min"]["type"] == "dirichlet":
                    rhs[center] -= get_weighted_value(bc, "y_min", Cy) * bc["y_min"]["value"]
                if j == ny - 1 and not bc.get("periodic_y") and bc["y_max"]["type"] == "dirichlet":
                    rhs[center] -= get_weighted_value(bc, "y_max", Cy) * bc["y_max"]["value"]

                if k == 0 and not bc.get("periodic_z") and bc["z_min"]["type"] == "dirichlet":
                    rhs[center] -= get_weighted_value(bc, "z_min", Cz) * bc["z_min"]["value"]
                if k == nz - 1 and not bc.get("periodic_z") and bc["z_max"]["type"] == "dirichlet":
                    rhs[center] -= get_weighted_value(bc, "z_max", Cz) * bc["z_max"]["value"]

    return rhs



