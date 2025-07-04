# src/numerical_methods/pressure_divergence.py

import numpy as np

def compute_pressure_divergence(u_field, mesh_info, should_log_verbose=False):
    """
    Computes the divergence of the velocity field (∇·u) using central differencing.
    This function calculates the divergence for the interior cells of the domain.

    Args:
        u_field (np.ndarray): Velocity field with ghost cells, shape (nx+2, ny+2, nz+2, 3).
                              Assumed to have correct boundary conditions applied to ghost cells.
        mesh_info (dict): Dictionary containing grid spacing: 'dx', 'dy', 'dz'.
        should_log_verbose (bool): If True, print detailed debug logs.

    Returns:
        np.ndarray: Divergence field of shape (nx, ny, nz) (interior cells only).
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Extract individual velocity components and defensively clamp any NaNs/Infs
    u = np.nan_to_num(u_field[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(u_field[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(u_field[..., 2], nan=0.0, posinf=0.0, neginf=0.0)

    # Compute partial derivatives using central differences for the interior domain.
    # The slicing [1:-1, 1:-1, 1:-1] implicitly selects the interior cells for which
    # the divergence is calculated. The [2:] and [:-2] access the i+1 and i-1 neighbors.
    du_dx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
    dv_dy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
    dw_dz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)

    # Sum the partial derivatives to get the divergence (∇·u)
    divergence = du_dx + dv_dy + dw_dz

    # Final check for NaNs/Infs in the computed divergence and clamp if necessary
    # These are critical warnings and should always be printed.
    if np.isnan(divergence).any() or np.isinf(divergence).any():
        print("❌ Warning: Invalid values in divergence field — clamping to zero.")
    divergence = np.nan_to_num(divergence, nan=0.0, posinf=0.0, neginf=0.0)

    # No verbose logging specific to this function's calculation, as it's a direct computation.
    # Verbose logging of max divergence is handled in explicit_solver.py.

    return divergence


def compute_pressure_gradient(p_field, mesh_info, should_log_verbose=False):
    """
    Computes the gradient ∇p of a scalar pressure field using central differencing.
    This function calculates the gradient for the interior cells of the domain.

    Args:
        p_field (np.ndarray): Scalar pressure field with ghost cells, shape (nx+2, ny+2, nz+2).
                              Assumed to have correct boundary conditions applied to ghost cells.
        mesh_info (dict): Dictionary containing grid spacing: 'dx', 'dy', 'dz'.
        should_log_verbose (bool): If True, print detailed debug logs. (Added for consistency)

    Returns:
        np.ndarray: Pressure gradient field of shape (nx, ny, nz, 3) (interior cells only).
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    # Defensively clamp any NaNs/Infs in the input pressure field
    p_field = np.nan_to_num(p_field, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute partial derivatives of pressure using central differences for the interior domain.
    # Similar to divergence, slicing implicitly selects the interior cells.
    grad_x = (p_field[2:, 1:-1, 1:-1] - p_field[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = (p_field[1:-1, 2:, 1:-1] - p_field[1:-1, :-2, 1:-1]) / (2 * dy)
    grad_z = (p_field[1:-1, 1:-1, 2:] - p_field[1:-1, 1:-1, :-2]) / (2 * dz)

    # Stack the individual gradient components to form a vector field
    grad = np.stack([grad_x, grad_y, grad_z], axis=-1)

    # Final check for NaNs/Infs in the computed gradient and clamp if necessary
    # These are critical warnings and should always be printed.
    if np.isnan(grad).any() or np.isinf(grad).any():
        print("❌ Warning: Invalid values in pressure gradient — clamping to zero.")
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    # No verbose logging specific to this function's calculation, as it's a direct computation.
    # Verbose logging of max/mean gradient is handled in apply_pressure_correction.

    return grad



