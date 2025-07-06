# src/numerical_methods/poisson_solver.py

import numpy as np
from scipy.sparse.linalg import spsolve, bicgstab, LinearOperator, spilu
from .matrix_assembly import assemble_poisson_matrix, apply_poisson_rhs_bcs

# Solver options
SOLVER_BACKEND_DIRECT = "direct"
SOLVER_BACKEND_BICGSTAB = "bicgstab"

PRECONDITIONER_NONE = "none"
PRECONDITIONER_ILU = "ilu"


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
    """
    Solves ∇²φ = ∇·u/dt using linear system and returns phi (ghost-padded).

    Args:
        divergence_field (np.ndarray): Full field [nx+2, ny+2, nz+2] including ghost zones.
        mesh_info (dict): Contains dx, dy, dz and boundary_conditions.
        dt (float): Time step size.
        tolerance (float): Solver convergence tolerance.
        max_iterations (int): Max iterations for iterative solvers.
        backend (str): "direct" or "bicgstab".
        preconditioner_type (str): "none" or "ilu".
        return_residual (bool): If True, return ||Ax - b|| after solving.

    Returns:
        phi (np.ndarray): Pressure potential (ghost-padded).
        residual (float): Norm of linear solve residual (if requested).
    """
    nx_total, ny_total, nz_total = divergence_field.shape
    nx, ny, nz = nx_total - 2, ny_total - 2, nz_total - 2
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))

    rhs = (divergence_field[interior] / dt).flatten()

    # Prepare BC config
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

    # Assemble matrix and RHS
    A = assemble_poisson_matrix(nx, ny, nz, dx, dy, dz, bc)
    rhs = apply_poisson_rhs_bcs(rhs, nx, ny, nz, dx, dy, dz, bc)

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

    # Reshape and ghost-pad result
    phi_interior = phi_flat.reshape((nx, ny, nz))
    phi = np.zeros_like(divergence_field)
    phi[interior] = phi_interior

    # Apply Neumann ghost extrapolation
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



