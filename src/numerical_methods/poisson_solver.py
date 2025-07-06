# src/numerical_methods/poisson_solver.py

import numpy as np
from scipy.sparse import lil_matrix, identity
from scipy.sparse.linalg import spsolve, bicgstab, LinearOperator, spilu
import sys

# Solver options
SOLVER_BACKEND_DIRECT = "direct"
SOLVER_BACKEND_BICGSTAB = "bicgstab"

PRECONDITIONER_NONE = "none"
PRECONDITIONER_ILU = "ilu"


def _assemble_poisson_matrix(nx, ny, nz, dx, dy, dz, bc):
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


def _apply_poisson_rhs_bcs(b, nx, ny, nz, dx, dy, dz, bc):
    Cx, Cy, Cz = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2

    def idx(i, j, k):
        return i + j * nx + k * nx * ny

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                center = idx(i, j, k)
                if i == 0 and not bc.get("periodic_x") and bc["x_min"]["type"] == "dirichlet":
                    b[center] -= Cx * bc["x_min"]["value"]
                if i == nx - 1 and not bc.get("periodic_x") and bc["x_max"]["type"] == "dirichlet":
                    b[center] -= Cx * bc["x_max"]["value"]
                if j == 0 and not bc.get("periodic_y") and bc["y_min"]["type"] == "dirichlet":
                    b[center] -= Cy * bc["y_min"]["value"]
                if j == ny - 1 and not bc.get("periodic_y") and bc["y_max"]["type"] == "dirichlet":
                    b[center] -= Cy * bc["y_max"]["value"]
                if k == 0 and not bc.get("periodic_z") and bc["z_min"]["type"] == "dirichlet":
                    b[center] -= Cz * bc["z_min"]["value"]
                if k == nz - 1 and not bc.get("periodic_z") and bc["z_max"]["type"] == "dirichlet":
                    b[center] -= Cz * bc["z_max"]["value"]
    return b


def solve_poisson_for_phi(
    divergence_field,
    mesh_info,
    dt,
    tolerance=1e-6,
    max_iterations=1000,
    backend=SOLVER_BACKEND_BICGSTAB,
    preconditioner_type=PRECONDITIONER_ILU,
    return_residual=False
):
    nx_total, ny_total, nz_total = divergence_field.shape
    nx, ny, nz = nx_total - 2, ny_total - 2, nz_total - 2
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))

    rhs = (divergence_field[interior] / dt).flatten()

    default_bc = {"type": "neumann", "value": 0.0}
    bc_raw = mesh_info.get("boundary_conditions", {})
    bc = {
        "x_min": bc_raw.get("x_min", default_bc),
        "x_max": bc_raw.get("x_max", default_bc),
        "y_min": bc_raw.get("y_min", default_bc),
        "y_max": bc_raw.get("y_max", default_bc),
        "z_min": bc_raw.get("z_min", default_bc),
        "z_max": bc_raw.get("z_max", default_bc),
        "periodic_x": bc_raw.get("periodic_x", False),
        "periodic_y": bc_raw.get("periodic_y", False),
        "periodic_z": bc_raw.get("periodic_z", False),
    }

    A = _assemble_poisson_matrix(nx, ny, nz, dx, dy, dz, bc)
    rhs = _apply_poisson_rhs_bcs(rhs, nx, ny, nz, dx, dy, dz, bc)

    phi_flat, residual = None, np.nan

    if backend == SOLVER_BACKEND_DIRECT:
        phi_flat = spsolve(A, rhs)
        residual = np.linalg.norm(A @ phi_flat - rhs)

    elif backend == SOLVER_BACKEND_BICGSTAB:
        M = None
        if preconditioner_type == PRECONDITIONER_ILU:
            try:
                ilu = spilu(A.tocsc(), drop_tol=1e-5, fill_factor=20)
                M = LinearOperator(A.shape, ilu.solve)
            except Exception:
                M = None

        phi_flat, info = bicgstab(A, rhs, rtol=tolerance, maxiter=max_iterations, M=M)
        if info != 0:
            raise RuntimeError(f"BiCGSTAB did not converge. Info={info}")
        residual = np.linalg.norm(A @ phi_flat - rhs)

    else:
        raise ValueError(f"Unknown backend: {backend}")

    phi_interior = phi_flat.reshape((nx, ny, nz))
    phi = np.zeros_like(divergence_field)
    phi[interior] = phi_interior

    if not bc["periodic_x"]:
        if bc["x_min"]["type"] == "neumann":
            phi[0, :, :] = phi[1, :, :]
        if bc["x_max"]["type"] == "neumann":
            phi[-1, :, :] = phi[-2, :, :]
    if not bc["periodic_y"]:
        if bc["y_min"]["type"] == "neumann":
            phi[:, 0, :] = phi[:, 1, :]
        if bc["y_max"]["type"] == "neumann":
            phi[:, -1, :] = phi[:, -2, :]
    if not bc["periodic_z"]:
        if bc["z_min"]["type"] == "neumann":
            phi[:, :, 0] = phi[:, :, 1]
        if bc["z_max"]["type"] == "neumann":
            phi[:, :, -1] = phi[:, :, -2]

    return (phi, residual) if return_residual else phi



