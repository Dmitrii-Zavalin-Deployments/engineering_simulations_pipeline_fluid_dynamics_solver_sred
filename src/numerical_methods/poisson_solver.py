# src/numerical_methods/poisson_solver.py

import numpy as np
from numba import jit, float64


@jit(
    float64[:, :, :](
        float64[:, :, :],  # phi
        float64[:, :, :],  # b
        float64, float64, float64,  # dx, dy, dz
        float64,  # omega
        float64   # max_iterations as float for Numba compatibility
    ),
    nopython=True,
    parallel=False,
    cache=False  # Disable cache to avoid module import errors during testing
)
def _sor_kernel(phi, b, dx, dy, dz, omega, max_iterations):
    nx, ny, nz = phi.shape
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    dz2_inv = 1.0 / (dz * dz)
    denom = 2.0 * (dx2_inv + dy2_inv + dz2_inv)

    for it in range(int(max_iterations)):
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if (
                        i == 0 or i == nx - 1 or
                        j == 0 or j == ny - 1 or
                        k == 0 or k == nz - 1
                    ):
                        # Enforcing zero Dirichlet boundary condition
                        phi[i, j, k] = 0.0
                    else:
                        # Apply SOR update using central differencing
                        term_x = (phi[i + 1, j, k] + phi[i - 1, j, k]) * dx2_inv
                        term_y = (phi[i, j + 1, k] + phi[i, j - 1, k]) * dy2_inv
                        term_z = (phi[i, j, k + 1] + phi[i, j, k - 1]) * dz2_inv
                        rhs = b[i, j, k]
                        phi_jacobi = (term_x + term_y + term_z - rhs) / denom
                        phi[i, j, k] = (1.0 - omega) * phi[i, j, k] + omega * phi_jacobi
    return phi


def solve_poisson_for_phi(divergence_u_star, mesh_info, time_step,
                          omega=1.7, max_iterations=1000, tolerance=1e-6):
    """
    Solves the Poisson equation for pressure correction (phi) using the SOR method.

    Args:
        divergence_u_star (np.ndarray): Source term (typically ∇·u*), shape (nx, ny, nz).
        mesh_info (dict): Contains:
            - 'grid_shape': tuple (nx, ny, nz)
            - 'dx', 'dy', 'dz': Grid spacing along x, y, z
        time_step (float): Timestep size (dt)
        omega (float): Relaxation factor (suggested 1.7 for 3D)
        max_iterations (int): Max SOR iterations
        tolerance (float): Not currently implemented (placeholder)

    Returns:
        np.ndarray: Pressure correction field phi, shape (nx, ny, nz)
    """
    nx, ny, nz = mesh_info["grid_shape"]
    dx = mesh_info["dx"]
    dy = mesh_info["dy"]
    dz = mesh_info["dz"]

    phi = np.zeros((nx, ny, nz), dtype=np.float64)
    b_source = divergence_u_star / time_step

    phi = _sor_kernel(phi, b_source, dx, dy, dz, omega, float(max_iterations))
    return phi



