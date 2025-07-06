# src/numerical_methods/diffusion.py

import numpy as np

def compute_diffusion_term(field, viscosity, mesh_info):
    """
    Computes the diffusion term (viscosity * ∇²(field)) using central differences.

    Args:
        field (np.ndarray): Scalar field (nx, ny, nz) or vector field (nx, ny, nz, 3) with ghost cells.
                            The ghost cells are assumed to be correctly populated by boundary conditions.
        viscosity (float): The fluid's kinematic or dynamic viscosity (ν or μ).
        mesh_info (dict): Grid metadata including:
            - 'grid_shape': (nx, ny, nz)
            - 'dx', 'dy', 'dz': spacing in x, y, z directions.

    Returns:
        np.ndarray: Diffusion term (same shape as input field).
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Clamp input to prevent NaN/Inf propagation
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    # Initialize directional second derivatives
    d2x = np.zeros_like(field)
    d2y = np.zeros_like(field)
    d2z = np.zeros_like(field)

    # d²/dx²
    if field.shape[0] > 2:
        d2x[1:-1, :, :] = (field[2:, :, :] - 2 * field[1:-1, :, :] + field[:-2, :, :]) / dx**2
        d2x[0, :, :] = d2x[1, :, :]
        d2x[-1, :, :] = d2x[-2, :, :]

    # d²/dy²
    if field.shape[1] > 2:
        d2y[:, 1:-1, :] = (field[:, 2:, :] - 2 * field[:, 1:-1, :] + field[:, :-2, :]) / dy**2
        d2y[:, 0, :] = d2y[:, 1, :]
        d2y[:, -1, :] = d2y[:, -2, :]

    # d²/dz²
    if field.shape[2] > 2:
        d2z[:, :, 1:-1] = (field[:, :, 2:] - 2 * field[:, :, 1:-1] + field[:, :, :-2]) / dz**2
        d2z[:, :, 0] = d2z[:, :, 1]
        d2z[:, :, -1] = d2z[:, :, -2]

    # Compute final diffusion term
    diffusion = viscosity * (d2x + d2y + d2z)

    # Clamp output and report instability if detected
    if np.isnan(diffusion).any() or np.isinf(diffusion).any():
        print("❌ Warning: Invalid values in diffusion term — clamping to zero.")
        diffusion = np.nan_to_num(diffusion, nan=0.0, posinf=0.0, neginf=0.0)

    # Diagnostic: Log maximum diffusion magnitude
    max_diff = np.max(np.abs(diffusion))
    print(f"📈 Max diffusion magnitude: {max_diff:.4e}")

    return diffusion


# The following function 'apply_diffusion_step' is not currently used by ExplicitSolver.step.
# ExplicitSolver.step directly computes the diffusion term and adds it to u_star.
# It's commented out to avoid confusion and keep the focus on the primary execution path.
# def apply_diffusion_step(field, diffusion_coefficient, mesh_info, dt):
#     """
#     Performs one explicit diffusion update using:
#     u_new = u + dt * ν ∇²(u)
#     """
#     diff = compute_diffusion_term(field, diffusion_coefficient, mesh_info)
#     return field + dt * diff



