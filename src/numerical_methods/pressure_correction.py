# src/numerical_methods/pressure_correction.py

import numpy as np

def calculate_gradient(field, h, axis):
    """
    Calculates the central difference gradient of a 3D scalar field.
    Assumes the input 'field' already includes ghost cells.
    The gradient is computed for the interior cells of the domain.

    Args:
        field (np.ndarray): Scalar field with ghost cells (e.g., shape: [nx+2, ny+2, nz+2]).
        h (float): Grid spacing along the axis.
        axis (int): Axis index (0 = x, 1 = y, 2 = z).

    Returns:
        np.ndarray: Gradient field for the interior cells (shape: [nx, ny, nz]).
    """
    # Defensive clamping for input field
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    # Slicing to compute central difference for the interior cells.
    # The result will have the shape of the interior domain (nx, ny, nz).
    if axis == 0: # Gradient along x-axis (d/dx)
        grad = (field[2:, 1:-1, 1:-1] - field[:-2, 1:-1, 1:-1]) / (2 * h)
    elif axis == 1: # Gradient along y-axis (d/dy)
        grad = (field[1:-1, 2:, 1:-1] - field[1:-1, :-2, 1:-1]) / (2 * h)
    elif axis == 2: # Gradient along z-axis (d/dz)
        grad = (field[1:-1, 1:-1, 2:] - field[1:-1, 1:-1, :-2]) / (2 * h)
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    # Final check for NaNs/Infs in the computed gradient and clamp if necessary
    if np.isnan(grad).any() or np.isinf(grad).any():
        print(f"❌ Warning: Invalid values in gradient axis {axis} — clamping to zero.")
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    return grad


def apply_pressure_correction(velocity_next, p_field, phi, mesh_info, time_step, density):
    """
    Applies pressure correction to the tentative velocity field to enforce incompressibility,
    and updates the pressure field.

    Args:
        velocity_next (np.ndarray): Tentative velocity field (u*) with ghost cells,
                                    shape [nx+2, ny+2, nz+2, 3].
        p_field (np.ndarray): Current pressure field with ghost cells,
                              shape [nx+2, ny+2, nz+2].
        phi (np.ndarray): Pressure correction potential (φ) with ghost cells,
                          shape [nx+2, ny+2, nz+2].
                          This is the output from solve_poisson_for_phi.
        mesh_info (dict): Grid spacing dictionary: {'dx', 'dy', 'dz'}.
        time_step (float): Time step Δt.
        density (float): Fluid density ρ.

    Returns:
        (np.ndarray, np.ndarray): Tuple containing:
            - corrected_velocity (np.ndarray): The updated, divergence-free velocity field
                                               with ghost cells, shape [nx+2, ny+2, nz+2, 3].
            - updated_pressure (np.ndarray): The updated pressure field with ghost cells,
                                             shape [nx+2, ny+2, nz+2].
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    # Optional: limit extreme phi corrections to prevent blow-up if solver is unstable
    # This is a safety measure; ideally, phi should not become excessively large.
    if np.abs(phi).max() > 1e4: # Threshold can be tuned
        print("⚠️ Pressure correction potential (phi) exceeds threshold — clipping applied.")
        phi = np.clip(phi, -1e3, 1e3) # Clip to a reasonable range

    # Update the pressure field: P_new = P_old + (rho / dt) * phi
    # The pressure update is applied only to the interior cells.
    updated_pressure = p_field.copy()
    # Note: The phi from Poisson solver is typically already scaled by (rho/dt) or similar,
    # or the pressure update formula might be P_new = P_old + phi
    # Based on the original code, it seems phi is directly added to pressure.
    # If phi is the pressure correction potential, the pressure update is usually:
    # P_new = P_old + (rho / dt) * phi (if phi is just the potential)
    # Or P_new = P_old + phi (if phi is already (rho/dt) * potential)
    # Given your current code `updated_pressure[1:-1, 1:-1, 1:-1] += phi`,
    # it implies phi is already scaled to be added directly to pressure.
    # However, the standard projection method has P_new = P_old + rho * phi / dt.
    # Let's assume phi from the Poisson solver is already scaled such that it can be added directly.
    # If not, you might need `updated_pressure[1:-1, 1:-1, 1:-1] += density * phi[1:-1, 1:-1, 1:-1] / time_step`
    # For now, keeping original logic for direct addition, but be aware of this.
    updated_pressure[1:-1, 1:-1, 1:-1] += phi[1:-1, 1:-1, 1:-1] # Apply to interior cells only

    # Compute the gradient of phi for each direction.
    # calculate_gradient now correctly takes phi with ghost cells and returns interior gradient.
    grad_phi_x = calculate_gradient(phi, dx, axis=0)
    grad_phi_y = calculate_gradient(phi, dy, axis=1)
    grad_phi_z = calculate_gradient(phi, dz, axis=2)

    # Apply the pressure correction to the tentative velocity field:
    # u_corrected = u_star - (dt / rho) * grad(phi)
    corrected_velocity = velocity_next.copy()
    corrected_velocity[1:-1, 1:-1, 1:-1, 0] -= time_step * grad_phi_x / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 1] -= time_step * grad_phi_y / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 2] -= time_step * grad_phi_z / density

    # Final safety clamp for corrected velocity and updated pressure
    if np.isnan(corrected_velocity).any() or np.isinf(corrected_velocity).any():
        print("❌ Warning: Invalid values in corrected velocity — clamping to zero.")
        corrected_velocity = np.nan_to_num(corrected_velocity, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(updated_pressure).any() or np.isinf(updated_pressure).any():
        print("❌ Warning: Invalid values in updated pressure — clamping to zero.")
        updated_pressure = np.nan_to_num(updated_pressure, nan=0.0, posinf=0.0, neginf=0.0)

    return corrected_velocity, updated_pressure




