# src/numerical_methods/pressure_correction.py

import numpy as np
import sys

def _check_nan_inf(field, name, step_number, output_frequency_steps):
    """Helper to check for NaN/Inf and print debug info conditionally."""
    if step_number % output_frequency_steps == 0:
        has_nan = np.isnan(field).any()
        has_inf = np.isinf(field).any()
        min_val = np.min(field) if not has_nan and not has_inf else float('nan')
        max_val = np.max(field) if not has_nan and not has_inf else float('nan')
        print(f"  [PressureCorr DEBUG] {name} stats BEFORE clamp: min={min_val:.2e}, max={max_val:.2e}, has_nan={has_nan}, has_inf={has_inf}")

    # Always clamp to prevent propagation, regardless of output frequency
    if np.isnan(field).any() or np.isinf(field).any():
        if step_number % output_frequency_steps == 0:
            print(f"  ❌ Warning: Invalid values in {name} — clamping to zero.")
        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    if step_number % output_frequency_steps == 0:
        print(f"  [PressureCorr DEBUG] {name} stats AFTER clamp: min={np.min(field):.2e}, max={np.max(field):.2e}")
    return field


def calculate_gradient(field, h, axis, step_number, output_frequency_steps):
    """
    Calculates the central difference gradient of a 3D scalar field.
    Assumes the input 'field' already includes ghost cells.
    The gradient is computed for the interior cells of the domain.

    Args:
        field (np.ndarray): Scalar field with ghost cells (e.g., shape: [nx+2, ny+2, nz+2]).
        h (float): Grid spacing along the axis.
        axis (int): Axis index (0 = x, 1 = y, 2 = z).
        step_number (int): Current simulation step number, used for conditional logging.
        output_frequency_steps (int): Frequency for printing debug output, used for conditional logging.

    Returns:
        np.ndarray: Gradient field for the interior cells (shape: [nx, ny, nz]).
    """
    # Defensive clamping for input field
    field = _check_nan_inf(field, f"gradient input (axis {axis})", step_number, output_frequency_steps)

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
        if step_number % output_frequency_steps == 0:
            print(f"❌ Warning: Invalid values in gradient axis {axis} — clamping to zero.")
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    if step_number % output_frequency_steps == 0:
        print(f"  [PressureCorr DEBUG] grad_phi_{'x' if axis==0 else 'y' if axis==1 else 'z'} stats: min={np.min(grad):.2e}, max={np.max(grad):.2e}, has_nan={np.isnan(grad).any()}, has_inf={np.isinf(grad).any()}")

    return grad


def apply_pressure_correction(
    velocity_next: np.ndarray,
    p_field: np.ndarray,
    phi: np.ndarray,
    mesh_info: dict,
    time_step: float,
    density: float,
    step_number: int,          # Added for conditional logging
    output_frequency_steps: int # Added for conditional logging
) -> Tuple[np.ndarray, np.ndarray]:
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
                          This is the output from solve_poisson_for_phi, where
                          ∇²φ = (∇·u*) / Δt.
        mesh_info (dict): Grid spacing dictionary: {'dx', 'dy', 'dz'}.
        time_step (float): Time step Δt.
        density (float): Fluid density ρ.
        step_number (int): Current simulation step number, used for conditional logging.
        output_frequency_steps (int): Frequency for printing debug output, used for conditional logging.

    Returns:
        (np.ndarray, np.ndarray): Tuple containing:
            - corrected_velocity (np.ndarray): The updated, divergence-free velocity field
                                               with ghost cells, shape [nx+2, ny+2, nz+2, 3].
            - updated_pressure (np.ndarray): The updated pressure field with ghost cells,
                                             shape [nx+2, ny+2, nz+2].
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    # DEBUG: Inspect phi coming into this function
    if step_number % output_frequency_steps == 0:
        print(f"  [PressureCorr DEBUG] phi input stats: min={np.min(phi):.2e}, max={np.max(phi):.2e}, has_nan={np.isnan(phi).any()}, has_inf={np.isinf(phi).any()}")

    # Optional: limit extreme phi corrections to prevent blow-up if solver is unstable
    # This clipping is always applied for safety, but the warning is conditional
    if np.abs(phi).max() > 1e4: # Threshold can be tuned
        if step_number % output_frequency_steps == 0:
            print("⚠️ Pressure correction potential (phi) exceeds threshold — clipping applied.")
        phi = np.clip(phi, -1e3, 1e3) # Clip to a reasonable range
        if step_number % output_frequency_steps == 0:
            print(f"  [PressureCorr DEBUG] phi after clipping: min={np.min(phi):.2e}, max={np.max(phi):.2e}")

    # Initialize updated_pressure with a copy of p_field
    updated_pressure = p_field.copy()
    # DEBUG: Inspect p_field before update
    if step_number % output_frequency_steps == 0:
        print(f"  [PressureCorr DEBUG] p_field BEFORE update: min={np.min(p_field):.2e}, max={np.max(p_field):.2e}, has_nan={np.isnan(p_field).any()}, has_inf={np.isinf(p_field).any()}")


    # Update the pressure field: P_new = P_old + ρ * φ
    # Since Poisson solver solves ∇²φ = (∇·u*) / Δt, φ already incorporates 1/Δt.
    # Thus, the pressure update is P_new = P_old + ρ * φ
    updated_pressure[1:-1, 1:-1, 1:-1] += density * phi[1:-1, 1:-1, 1:-1]

    # DEBUG: Inspect updated_pressure after update, before clamping
    # Added explicit checks for NaN/Inf for more reliable debugging
    nan_check_pressure_after_update = np.isnan(updated_pressure).any()
    inf_check_pressure_after_update = np.isinf(updated_pressure).any()
    if step_number % output_frequency_steps == 0:
        print(f"  [PressureCorr DEBUG] updated_pressure AFTER update (before clamp): min={np.min(updated_pressure):.2e}, max={np.max(updated_pressure):.2e}, has_nan={nan_check_pressure_after_update}, has_inf={inf_check_pressure_after_update}")


    # Final safety clamp for updated_pressure
    if nan_check_pressure_after_update or inf_check_pressure_after_update:
        if step_number % output_frequency_steps == 0:
            print("❌ Warning: Invalid values in updated pressure — clamping to zero.")
    updated_pressure = np.nan_to_num(updated_pressure, nan=0.0, posinf=0.0, neginf=0.0)
    # DEBUG: Inspect updated_pressure after clamping
    if step_number % output_frequency_steps == 0:
        print(f"  [PressureCorr DEBUG] updated_pressure AFTER clamp: min={np.min(updated_pressure):.2e}, max={np.max(updated_pressure):.2e}")


    # Compute the gradient of phi for each direction.
    # calculate_gradient now correctly takes phi with ghost cells and returns interior gradient.
    grad_phi_x = calculate_gradient(phi, dx, axis=0, step_number=step_number, output_frequency_steps=output_frequency_steps)
    grad_phi_y = calculate_gradient(phi, dy, axis=1, step_number=step_number, output_frequency_steps=output_frequency_steps)
    grad_phi_z = calculate_gradient(phi, dz, axis=2, step_number=step_number, output_frequency_steps=output_frequency_steps)

    # Apply the pressure correction to the tentative velocity field:
    # u_corrected = u_star - (dt / rho) * grad(phi)
    # This formula is correct as grad(phi) is based on the potential phi.
    corrected_velocity = velocity_next.copy()
    corrected_velocity[1:-1, 1:-1, 1:-1, 0] -= time_step * grad_phi_x / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 1] -= time_step * grad_phi_y / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 2] -= time_step * grad_phi_z / density

    # DEBUG: Inspect corrected_velocity after update, before clamping
    nan_check_velocity_after_update = np.isnan(corrected_velocity).any()
    inf_check_velocity_after_update = np.isinf(corrected_velocity).any()
    if step_number % output_frequency_steps == 0:
        print(f"  [PressureCorr DEBUG] corrected_velocity AFTER update (before clamp): min={np.min(corrected_velocity):.2e}, max={np.max(corrected_velocity):.2e}, has_nan={nan_check_velocity_after_update}, has_inf={inf_check_velocity_after_update}")


    # Final safety clamp for corrected velocity
    if nan_check_velocity_after_update or inf_check_velocity_after_update:
        if step_number % output_frequency_steps == 0:
            print("❌ Warning: Invalid values in corrected velocity — clamping to zero.")
    corrected_velocity = np.nan_to_num(corrected_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    # DEBUG: Inspect corrected_velocity after clamping
    if step_number % output_frequency_steps == 0:
        print(f"  [PressureCorr DEBUG] corrected_velocity AFTER clamp: min={np.min(corrected_velocity):.2e}, max={np.max(corrected_velocity):.2e}")


    return corrected_velocity, updated_pressure



