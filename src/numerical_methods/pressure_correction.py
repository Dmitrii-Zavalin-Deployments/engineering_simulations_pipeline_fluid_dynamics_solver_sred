# src/numerical_methods/pressure_correction.py

import numpy as np
# Assuming compute_pressure_gradient is in the same directory or accessible
from .pressure_divergence import compute_pressure_gradient

def apply_pressure_correction(
    u_star: np.ndarray,
    pressure_field: np.ndarray,
    phi: np.ndarray,
    mesh_info: dict,
    dt: float,
    density: float,
    should_log_verbose: bool = False # Added verbose logging flag
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the pressure correction to the tentative velocity field (u*) and updates
    the pressure field, ensuring the corrected velocity field is divergence-free.

    Args:
        u_star (np.ndarray): Tentative velocity field (u*) on the full grid
                             including ghost cells (nx+2, ny+2, nz+2, 3).
        pressure_field (np.ndarray): Previous pressure field on the full grid
                                     including ghost cells (nx+2, ny+2, nz+2).
        phi (np.ndarray): Scalar potential field from the Poisson solver,
                          on the full grid including ghost cells (nx+2, ny+2, nz+2).
        mesh_info (dict): Dictionary containing grid spacing: 'dx', 'dy', 'dz'.
        dt (float): Time step size.
        density (float): Fluid density.
        should_log_verbose (bool): If True, print detailed debug logs.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - updated_velocity (np.ndarray): The divergence-free velocity field
                                             on the full grid (nx+2, ny+2, nz+2, 3).
            - updated_pressure (np.ndarray): The updated pressure field
                                              on the full grid (nx+2, ny+2, nz+2).
    """
    # Initialize updated fields by copying the input
    updated_velocity = np.copy(u_star)
    updated_pressure = np.copy(pressure_field)

    # Compute the gradient of phi (∇phi) for the interior cells
    # The compute_pressure_gradient function expects a field with ghost cells and returns interior gradient
    gradient_phi = compute_pressure_gradient(phi, mesh_info, should_log_verbose=should_log_verbose)

    # Apply pressure correction to the interior velocity field
    # u_new = u* - (dt / density) * ∇phi
    # Note: u_star and updated_velocity have ghost cells, gradient_phi does not.
    # We apply correction to interior cells: updated_velocity[1:-1, 1:-1, 1:-1, :]
    updated_velocity[1:-1, 1:-1, 1:-1, 0] -= (dt / density) * gradient_phi[..., 0]
    updated_velocity[1:-1, 1:-1, 1:-1, 1] -= (dt / density) * gradient_phi[..., 1]
    updated_velocity[1:-1, 1:-1, 1:-1, 2] -= (dt / density) * gradient_phi[..., 2]

    # Update pressure field
    # p_new = p_old + density * phi / dt
    # Apply to interior cells. Phi is defined such that its values correspond to cell centers.
    updated_pressure[1:-1, 1:-1, 1:-1] += density * phi[1:-1, 1:-1, 1:-1] / dt

    # Check for NaNs or Infs and handle them in the corrected fields
    if np.isnan(updated_velocity).any() or np.isinf(updated_velocity).any():
        print("❌ Warning: Invalid values in updated velocity during pressure correction. Clamping to zero.")
        updated_velocity = np.nan_to_num(updated_velocity, nan=0.0, posinf=0.0, neginf=0.0)

    if np.isnan(updated_pressure).any() or np.isinf(updated_pressure).any():
        print("❌ Warning: Invalid values in updated pressure during pressure correction. Clamping to zero.")
        updated_pressure = np.nan_to_num(updated_pressure, nan=0.0, posinf=0.0, neginf=0.0)

    if should_log_verbose:
        # These are debug prints, only show if verbose logging is on
        print(f"    - Pressure Correction: Max Abs Velocity Correction: {np.max(np.abs((dt / density) * gradient_phi)):.4e}")
        print(f"    - Pressure Correction: Max Abs Pressure Update: {np.max(np.abs(density * phi[1:-1, 1:-1, 1:-1] / dt)):.4e}")

    return updated_velocity, updated_pressure



