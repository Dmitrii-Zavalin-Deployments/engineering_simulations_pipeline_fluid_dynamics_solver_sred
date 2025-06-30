# src/numerical_methods/poisson_solver.py

import numpy as np
from numba import jit, float64


@jit(
    float64[:, :, :](
        float64[:, :, :],  # phi
        float64[:, :, :],  # b
        float64, float64, float64,  # dx, dy, dz
        float64,  # omega
        float64,  # max_iterations
        float64,  # tolerance
        float64[:]  # output_residual[0]
    ),
    nopython=True,
    parallel=False,
    cache=False
)
def _sor_kernel_with_residual(phi, b, dx, dy, dz, omega, max_iterations, tolerance, output_residual):
    nx, ny, nz = phi.shape
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    dz2_inv = 1.0 / (dz * dz)
    denom = 2.0 * (dx2_inv + dy2_inv + dz2_inv)

    for it in range(int(max_iterations)):
        max_residual = 0.0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
                        phi[i, j, k] = 0.0
                    else:
                        term_x = (phi[i + 1, j, k] + phi[i - 1, j, k]) * dx2_inv
                        term_y = (phi[i, j + 1, k] + phi[i, j - 1, k]) * dy2_inv
                        term_z = (phi[i, j, k + 1] + phi[i, j, k - 1]) * dz2_inv
                        rhs = b[i, j, k]
                        phi_jacobi = (term_x + term_y + term_z - rhs) / denom
                        delta = phi_jacobi - phi[i, j, k]
                        phi[i, j, k] += omega * delta
                        max_residual = max(max_residual, abs(delta))
        if max_residual < tolerance:
            break
    output_residual[0] = max_residual
    return phi


def solve_poisson_for_phi(divergence, mesh_info, time_step,
                          omega=1.7, max_iterations=1000, tolerance=1e-6,
                          return_residual=False, backend="sor"):
    """
    Solves the Poisson equation for pressure correction using a selected backend.

    Args:
        divergence (np.ndarray): Source term (∇·u*), shape (nx, ny, nz)
        mesh_info (dict): Dict with grid_shape and spacings ('dx', 'dy', 'dz')
        time_step (float): Timestep (dt)
        omega (float): Relaxation factor for SOR (default 1.7)
        max_iterations (int): Max SOR iterations
        tolerance (float): Residual threshold for convergence
        return_residual (bool): If True, return final residual as second output
        backend (str): Solver backend to use. Only 'sor' is currently supported.

    Returns:
        np.ndarray: Pressure correction field φ
        float (optional): Final residual (if return_residual=True)
    """
    if backend != "sor":
        raise ValueError(f"Unsupported backend '{backend}'. Only 'sor' is implemented.")

    nx, ny, nz = mesh_info["grid_shape"]
    dx = mesh_info["dx"]
    dy = mesh_info["dy"]
    dz = mesh_info["dz"]

    phi = np.zeros((nx, ny, nz), dtype=np.float64)
    rhs = divergence / time_step
    residual_container = np.zeros(1, dtype=np.float64)

    phi = _sor_kernel_with_residual(
        phi, rhs, dx, dy, dz, omega,
        float(max_iterations), float(tolerance), residual_container
    )

    return (phi, residual_container[0]) if return_residual else phi



