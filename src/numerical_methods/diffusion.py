# src/numerical_methods/diffusion.py

import numpy as np

def compute_diffusion_term(field, viscosity, mesh_info):
    """
    Computes the diffusion term (viscosity * ∇²(field)) using central differences.

    Args:
        field (np.ndarray): Scalar field (nx, ny, nz) or vector field (nx, ny, nz, 3) with ghost cells.
        viscosity (float): The fluid's kinematic or dynamic viscosity (ν or μ).
        mesh_info (dict): Grid metadata including:
            - 'grid_shape': (nx, ny, nz)
            - 'dx', 'dy', 'dz': spacing in x, y, z directions.

    Returns:
        np.ndarray: Diffusion term (same shape as input field).
    """
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    d2x = np.zeros_like(field)
    d2y = np.zeros_like(field)
    d2z = np.zeros_like(field)

    if nx > 2:
        d2x[1:-1, :, :] = (field[2:, :, :] - 2 * field[1:-1, :, :] + field[:-2, :, :]) / dx**2
    if ny > 2:
        d2y[:, 1:-1, :] = (field[:, 2:, :] - 2 * field[:, 1:-1, :] + field[:, :-2, :]) / dy**2
    if nz > 2:
        d2z[:, :, 1:-1] = (field[:, :, 2:] - 2 * field[:, :, 1:-1] + field[:, :, :-2]) / dz**2

    return viscosity * (d2x + d2y + d2z)

def apply_diffusion_step(field, diffusion_coefficient, mesh_info, dt):
    """
    Performs one explicit diffusion update using:
        u_new = u + dt * ν ∇²(u)

    Args:
        field (np.ndarray): Field with ghost padding (3D scalar or 4D vector).
        diffusion_coefficient (float): Kinematic viscosity.
        mesh_info (dict): Must include 'grid_shape', 'dx', 'dy', 'dz'.
        dt (float): Time step.

    Returns:
        np.ndarray: Updated field (same shape).
    """
    diff = compute_diffusion_term(field, diffusion_coefficient, mesh_info)
    return field + dt * diff



