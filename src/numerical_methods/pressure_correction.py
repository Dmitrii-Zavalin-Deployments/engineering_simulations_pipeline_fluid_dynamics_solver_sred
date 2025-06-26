# src/numerical_methods/pressure_correction.py

import numpy as np

def calculate_gradient(field, h, axis, padding_mode="edge"):
    """
    Calculates the central difference gradient of a 3D scalar field.

    Pads the input field using the specified mode to enable gradient estimation 
    at the boundaries. Central differencing is used along the given axis.

    Args:
        field (np.ndarray): Input scalar field of shape (nx, ny, nz).
        h (float): Grid spacing along the specified axis.
        axis (int): Axis to compute gradient along (0 = x, 1 = y, 2 = z).
        padding_mode (str): Padding mode passed to np.pad (default is 'edge').

            Common options:
                - 'edge': replicates boundary values (1st-order accurate)
                - 'reflect': mirror-symmetric boundaries (2nd-order)
                - 'constant': zero-padding
                - 'wrap': periodic boundaries

    Returns:
        np.ndarray: Gradient field of the same shape as input.

    Raises:
        ValueError: If axis is not 0, 1, or 2.
    """
    padded = np.pad(field, ((1, 1), (1, 1), (1, 1)), mode=padding_mode)

    if axis == 0:
        return (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) / (2 * h)
    elif axis == 1:
        return (padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]) / (2 * h)
    elif axis == 2:
        return (padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]) / (2 * h)
    else:
        raise ValueError("Axis must be 0, 1, or 2.")


def apply_pressure_correction(velocity_next, p_field, phi, mesh_info, time_step, density):
    """
    Applies pressure correction to enforce divergence-free velocity 
    using the pressure potential φ from the Poisson solve.

    Args:
        velocity_next (np.ndarray): Tentative velocity field (shape: nx+2, ny+2, nz+2, 3).
        p_field (np.ndarray): Pressure field with ghost cells (shape: nx+2, ny+2, nz+2).
        phi (np.ndarray): Pressure correction potential φ (shape: nx, ny, nz).
        mesh_info (dict): Dictionary with keys 'dx', 'dy', 'dz'.
        time_step (float): Time step Δt.
        density (float): Fluid density ρ.

    Returns:
        tuple: Corrected velocity field and updated pressure field.
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    updated_pressure = p_field.copy()
    updated_pressure[1:-1, 1:-1, 1:-1] += phi

    grad_phi_x = calculate_gradient(phi, dx, axis=0, padding_mode="edge")
    grad_phi_y = calculate_gradient(phi, dy, axis=1, padding_mode="edge")
    grad_phi_z = calculate_gradient(phi, dz, axis=2, padding_mode="edge")

    corrected_velocity = velocity_next.copy()
    corrected_velocity[1:-1, 1:-1, 1:-1, 0] -= time_step * grad_phi_x / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 1] -= time_step * grad_phi_y / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 2] -= time_step * grad_phi_z / density

    return corrected_velocity, updated_pressure



