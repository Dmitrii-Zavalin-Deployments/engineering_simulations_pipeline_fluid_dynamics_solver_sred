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
    cache=True
)
def _sor_kernel_with_residual(phi, b, dx, dy, dz, omega, max_iterations, tolerance, output_residual):
    nx, ny, nz = phi.shape
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    dz2_inv = 1.0 / (dz * dz)
    denom = 2.0 * (dx2_inv + dy2_inv + dz2_inv)

    for it in range(int(max_iterations)):
        max_residual = 0.0
        for k in range(1, nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
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
    if backend != "sor":
        raise ValueError(f"Unsupported backend '{backend}'.")

    nx_total, ny_total, nz_total = divergence.shape
    dx = mesh_info["dx"]
    dy = mesh_info["dy"]
    dz = mesh_info["dz"]

    phi = np.zeros((nx_total, ny_total, nz_total), dtype=np.float64)
    processed_bcs = mesh_info.get("boundary_conditions", {})

    print("[Poisson Solver] Initializing phi field with Dirichlet BCs...")
    for bc_name, bc in processed_bcs.items():
        bc_type = bc.get("type")
        apply_to_fields = bc.get("apply_to", [])

        if bc_type == "dirichlet" and "pressure" in apply_to_fields:
            ghost_indices = np.array(bc.get("ghost_indices", []), dtype=int)
            target_pressure = bc.get("pressure", 0.0)

            if ghost_indices.size > 0:
                # Defensive mask to avoid IndexError
                valid_mask = (
                    (ghost_indices[:, 0] >= 0) & (ghost_indices[:, 0] < nx_total) &
                    (ghost_indices[:, 1] >= 0) & (ghost_indices[:, 1] < ny_total) &
                    (ghost_indices[:, 2] >= 0) & (ghost_indices[:, 2] < nz_total)
                )
                safe_indices = ghost_indices[valid_mask]
                phi[safe_indices[:, 0], safe_indices[:, 1], safe_indices[:, 2]] = target_pressure
                print(f"   -> Applied pressure {target_pressure} to phi field for BC '{bc_name}'.")
            else:
                print(f"   -> WARNING: No ghost indices found for pressure BC '{bc_name}'.")

    rhs = divergence / time_step
    residual_container = np.zeros(1, dtype=np.float64)

    print(f"[Poisson Solver] Starting SOR solver with {max_iterations} iterations and tolerance {tolerance}.")
    phi = _sor_kernel_with_residual(phi, rhs, dx, dy, dz, omega,
                                    float(max_iterations), float(tolerance), residual_container)
    print(f"[Poisson Solver] Solver finished. Final residual: {residual_container[0]:.6e}")

    return (phi, residual_container[0]) if return_residual else phi



