# src/numerical_methods/pressure_divergence.py

import numpy as np

def compute_pressure_divergence(u_field, mesh_info):
    """
    Computes the divergence of the velocity field (∇·u) using central differencing.

    Args:
        u_field (np.ndarray): Velocity field with ghost cells, shape (nx+2, ny+2, nz+2, 3).
        mesh_info (dict): Includes 'dx', 'dy', 'dz'.

    Returns:
        np.ndarray: Divergence field of shape (nx, ny, nz).
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    u = np.nan_to_num(u_field[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(u_field[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(u_field[..., 2], nan=0.0, posinf=0.0, neginf=0.0)

    du_dx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
    dv_dy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
    dw_dz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)

    divergence = du_dx + dv_dy + dw_dz

    if np.isnan(divergence).any() or np.isinf(divergence).any():
        print("❌ Warning: Invalid values in divergence field — clamping to zero.")
    divergence = np.nan_to_num(divergence, nan=0.0, posinf=0.0, neginf=0.0)

    return divergence


def compute_pressure_gradient(p_field, mesh_info):
    """
    Computes the gradient ∇p of a scalar pressure field using central differencing.

    Args:
        p_field (np.ndarray): Scalar pressure field with ghost cells, shape (nx+2, ny+2, nz+2).
        mesh_info (dict): Includes 'dx', 'dy', 'dz'.

    Returns:
        np.ndarray: Pressure gradient field of shape (nx, ny, nz, 3).
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    p_field = np.nan_to_num(p_field, nan=0.0, posinf=0.0, neginf=0.0)

    grad_x = (p_field[2:, 1:-1, 1:-1] - p_field[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = (p_field[1:-1, 2:, 1:-1] - p_field[1:-1, :-2, 1:-1]) / (2 * dy)
    grad_z = (p_field[1:-1, 1:-1, 2:] - p_field[1:-1, 1:-1, :-2]) / (2 * dz)

    grad = np.stack([grad_x, grad_y, grad_z], axis=-1)

    if np.isnan(grad).any() or np.isinf(grad).any():
        print("❌ Warning: Invalid values in pressure gradient — clamping to zero.")
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    return grad



