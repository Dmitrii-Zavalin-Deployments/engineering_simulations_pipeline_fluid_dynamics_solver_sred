# src/numerical_methods/pressure_correction.py

import numpy as np

def calculate_gradient(field, h, axis, padding_mode="edge"):
    """
    Calculates the central difference gradient of a 3D scalar field.

    Pads the input field using the specified mode to enable gradient estimation 
    near boundaries. Central differencing is used along the specified axis.

    Args:
        field (np.ndarray): Scalar field (shape: [nx, ny, nz]).
        h (float): Grid spacing along the axis.
        axis (int): Axis index (0 = x, 1 = y, 2 = z).
        padding_mode (str): Padding strategy passed to np.pad.

    Returns:
        np.ndarray: Gradient field matching input shape.
    """
    padded = np.pad(field, ((1, 1), (1, 1), (1, 1)), mode=padding_mode)

    if axis == 0:
        grad = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) / (2 * h)
    elif axis == 1:
        grad = (padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]) / (2 * h)
    elif axis == 2:
        grad = (padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]) / (2 * h)
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    if np.isnan(grad).any() or np.isinf(grad).any():
        print(f"❌ Warning: Invalid values in gradient axis {axis} — clamping to zero.")
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    return grad


def apply_pressure_correction(velocity_next, p_field, phi, mesh_info, time_step, density):
    """
    Applies pressure correction to enforce incompressibility.

    Args:
        velocity_next (np.ndarray): Tentative velocity field [nx+2, ny+2, nz+2, 3].
        p_field (np.ndarray): Pressure field [nx+2, ny+2, nz+2].
        phi (np.ndarray): Pressure correction potential φ [nx, ny, nz].
        mesh_info (dict): Grid spacing dictionary: {'dx', 'dy', 'dz'}.
        time_step (float): Time step Δt.
        density (float): Fluid density ρ.

    Returns:
        (np.ndarray, np.ndarray): Updated velocity and pressure fields.
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    # Optional: limit extreme phi corrections
    if np.abs(phi).max() > 1e4:
        print("⚠️ Pressure correction potential (phi) exceeds threshold — clipping applied.")
        phi = np.clip(phi, -1e3, 1e3)

    updated_pressure = p_field.copy()
    updated_pressure[1:-1, 1:-1, 1:-1] += phi

    grad_phi_x = calculate_gradient(phi, dx, axis=0, padding_mode="edge")
    grad_phi_y = calculate_gradient(phi, dy, axis=1, padding_mode="edge")
    grad_phi_z = calculate_gradient(phi, dz, axis=2, padding_mode="edge")

    corrected_velocity = velocity_next.copy()
    corrected_velocity[1:-1, 1:-1, 1:-1, 0] -= time_step * grad_phi_x / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 1] -= time_step * grad_phi_y / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 2] -= time_step * grad_phi_z / density

    # Final safety clamp
    if np.isnan(corrected_velocity).any() or np.isinf(corrected_velocity).any():
        print("❌ Warning: Invalid values in corrected velocity — clamping to zero.")
        corrected_velocity = np.nan_to_num(corrected_velocity, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(updated_pressure).any() or np.isinf(updated_pressure).any():
        print("❌ Warning: Invalid values in updated pressure — clamping to zero.")
        updated_pressure = np.nan_to_num(updated_pressure, nan=0.0, posinf=0.0, neginf=0.0)

    return corrected_velocity, updated_pressure



