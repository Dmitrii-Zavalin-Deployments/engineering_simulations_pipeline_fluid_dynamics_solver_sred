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
    return_residual=False,
    fallback_on_failure=True
):
    """
    Solves ∇²φ = ∇·u/dt using sparse linear system.

    Args:
        divergence_field (np.ndarray): Full [nx+2, ny+2, nz+2] field incl. ghost zones
        mesh_info (dict): Grid params and BCs
        dt (float): Time step
        tolerance (float): Solver convergence threshold
        max_iterations (int): Max iterative solver cycles
        backend (str): Solver type (direct | bicgstab)
        preconditioner_type (str): Preconditioner type (ilu | none)
        return_residual (bool): Return residual norm if True
        fallback_on_failure (bool): Try spsolve if BiCGSTAB fails

    Returns:
        phi (np.ndarray): Ghost-padded solution field
        residual (float): Optional ||Ax - b|| error norm
    """
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

    A = assemble_poisson_matrix(nx, ny, nz, dx, dy, dz, bc)
    rhs = apply_poisson_rhs_bcs(rhs, nx, ny, nz, dx, dy, dz, bc)

    phi_flat, residual = None, np.nan

    try:
        if backend == SOLVER_BACKEND_DIRECT:
            phi_flat = spsolve(A, rhs)
            residual = np.linalg.norm(A @ phi_flat - rhs)
            print(f"✅ Direct solve residual: {residual:.4e}")

        elif backend == SOLVER_BACKEND_BICGSTAB:
            M = None
            if preconditioner_type == PRECONDITIONER_ILU:
                try:
                    ilu = spilu(A.tocsc(), drop_tol=1e-5, fill_factor=20)
                    M = LinearOperator(A.shape, ilu.solve)
                except Exception as e:
                    print(f"⚠️ ILU preconditioner setup failed: {e}")
                    M = None

            phi_flat, info = bicgstab(A, rhs, rtol=tolerance, maxiter=max_iterations, M=M)
            if info != 0:
                print(f"⚠️ BiCGSTAB did not converge (info={info})")
                if fallback_on_failure:
                    print("🔄 Fallback: switching to direct solve...")
                    phi_flat = spsolve(A, rhs)
                    residual = np.linalg.norm(A @ phi_flat - rhs)
                    print(f"✅ Direct solve residual (fallback): {residual:.4e}")
                else:
                    raise RuntimeError(f"BiCGSTAB failed with info={info}")
            else:
                residual = np.linalg.norm(A @ phi_flat - rhs)
                print(f"✅ BiCGSTAB residual: {residual:.4e}")
        else:
            raise ValueError(f"Unknown solver backend: '{backend}'")

    except Exception as e:
        print(f"❌ Linear solve error: {e}")
        if fallback_on_failure:
            print("🔄 Recovery attempt: direct solve...")
            phi_flat = spsolve(A, rhs)
            residual = np.linalg.norm(A @ phi_flat - rhs)
            print(f"✅ Direct solve residual (recovery): {residual:.4e}")
        else:
            raise

    phi_interior = phi_flat.reshape((nx, ny, nz))
    phi = np.zeros_like(divergence_field)
    phi[interior] = phi_interior

    # Ghost zone padding for Neumann BCs
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



