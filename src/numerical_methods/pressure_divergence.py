# src/numerical_methods/pressure_divergence.py

import numpy as np

def compute_pressure_divergence(u_field, mesh_info):
    """
    Computes the divergence of the velocity field (∇·u) using central differencing.

    Args:
        u_field (np.ndarray): Velocity field with ghost cells, shape (nx+2, ny+2, nz+2, 3).
        mesh_info (dict): Includes:
            - 'grid_shape': (nx+2, ny+2, nz+2)
            - 'dx', 'dy', 'dz': grid spacing in x, y, z directions.

    Returns:
        np.ndarray: Divergence field of shape (nx, ny, nz).
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    u = u_field[..., 0]
    v = u_field[..., 1]
    w = u_field[..., 2]

    du_dx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
    dv_dy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
    dw_dz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)

    divergence = du_dx + dv_dy + dw_dz
    return divergence

def compute_pressure_gradient(p_field, mesh_info):
    """
    Computes the gradient ∇p of a scalar pressure field using central differencing.

    Args:
        p_field (np.ndarray): Scalar pressure field with ghost cells, shape (nx+2, ny+2, nz+2).
        mesh_info (dict): Includes:
            - 'dx', 'dy', 'dz': grid spacing in x, y, z directions.

    Returns:
        np.ndarray: Pressure gradient field of shape (nx, ny, nz, 3).
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    grad_x = (p_field[2:, 1:-1, 1:-1] - p_field[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = (p_field[1:-1, 2:, 1:-1] - p_field[1:-1, :-2, 1:-1]) / (2 * dy)
    grad_z = (p_field[1:-1, 1:-1, 2:] - p_field[1:-1, 1:-1, :-2]) / (2 * dz)

    grad = np.stack([grad_x, grad_y, grad_z], axis=-1)
    return grad



