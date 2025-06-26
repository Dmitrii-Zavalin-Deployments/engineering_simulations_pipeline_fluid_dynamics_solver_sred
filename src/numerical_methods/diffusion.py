# src/numerical_methods/diffusion.py

import numpy as np

def compute_diffusion_term(field, viscosity, mesh_info):
    """
    Computes the diffusion term (viscosity * nabla^2(field)) using central differences.

    Args:
        field (np.ndarray): Scalar (nx, ny, nz) or vector (nx, ny, nz, 3) field.
        viscosity (float): The fluid's dynamic viscosity (mu).
        mesh_info (dict): Grid metadata with 'grid_shape', 'dx', 'dy', 'dz'.

    Returns:
        np.ndarray: Diffusion term, same shape as input field.
    """
    nx, ny, nz = mesh_info['grid_shape']
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    d2_field_dx2 = np.zeros_like(field)
    if nx > 2:
        d2_field_dx2[1:-1, :, :] = (
            field[2:, :, :] - 2 * field[1:-1, :, :] + field[:-2, :, :]
        ) / dx**2

    d2_field_dy2 = np.zeros_like(field)
    if ny > 2:
        d2_field_dy2[:, 1:-1, :] = (
            field[:, 2:, :] - 2 * field[:, 1:-1, :] + field[:, :-2, :]
        ) / dy**2

    d2_field_dz2 = np.zeros_like(field)
    if nz > 2:
        d2_field_dz2[:, :, 1:-1] = (
            field[:, :, 2:] - 2 * field[:, :, 1:-1] + field[:, :, :-2]
        ) / dz**2

    laplacian = d2_field_dx2 + d2_field_dy2 + d2_field_dz2
    return viscosity * laplacian

def apply_diffusion_step(field, diffusion_coefficient, mesh_info, dt):
    """
    Advances the field forward by one explicit diffusion step:
        u_new = u + dt * ν ∇²(u)

    Args:
        field (np.ndarray): 3D array (with ghost cells) to be diffused.
        diffusion_coefficient (float): ν (e.g., kinematic viscosity)
        mesh_info (dict): Must include 'grid_shape', 'dx', 'dy', 'dz'
        dt (float): Time step

    Returns:
        np.ndarray: New field after one diffusion update.
    """
    diff_term = compute_diffusion_term(field, diffusion_coefficient, mesh_info)
    return field + dt * diff_term



