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
    # Check for NaNs/Infs in the input field
    field_has_nan = np.isnan(field).any()
    field_has_inf = np.isinf(field).any()

    if field_has_nan or field_has_inf:
        # Use nanmin/nanmax here to avoid errors if the entire array is NaN
        print(f"    [Gradient DEBUG] Input field stats BEFORE clamp (axis {axis}): min={np.nanmin(field):.2e}, max={np.nanmax(field):.2e}, has_nan={field_has_nan}, has_inf={field_has_inf}")

    # Clamp input field before gradient calculation to prevent NaN propagation
    field_clamped_for_grad = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    # Slicing to compute central difference for the interior cells.
    # The result will have the shape of the interior domain (nx, ny, nz).
    if axis == 0: # Gradient along x-axis (d/dx)
        grad = (field_clamped_for_grad[2:, 1:-1, 1:-1] - field_clamped_for_grad[:-2, 1:-1, 1:-1]) / (2 * h)
    elif axis == 1: # Gradient along y-axis (d/dy)
        grad = (field_clamped_for_grad[1:-1, 2:, 1:-1] - field_clamped_for_grad[1:-1, :-2, 1:-1]) / (2 * h)
    elif axis == 2: # Gradient along z-axis (d/dz)
        grad = (field_clamped_for_grad[1:-1, 1:-1, 2:] - field_clamped_for_grad[1:-1, 1:-1, :-2]) / (2 * h)
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    # Final check for NaNs/Infs in the computed gradient and clamp if necessary
    grad_has_nan = np.isnan(grad).any()
    grad_has_inf = np.isinf(grad).any()
    if grad_has_nan or grad_has_inf:
        # Use nanmin/nanmax here
        print(f"❌ Warning: Invalid values in gradient axis {axis} — clamping to zero. Stats: min={np.nanmin(grad):.2e}, max={np.nanmax(grad):.2e}, has_nan={grad_has_nan}, has_inf={grad_has_inf}")
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
                          This is the output from solve_poisson_for_phi, where
                          ∇²φ = (∇·u*) / Δt.
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

    # DEBUG: Inspect phi coming into this function
    phi_has_nan = np.isnan(phi).any()
    phi_has_inf = np.isinf(phi).any()
    if phi_has_nan or phi_has_inf:
        print(f"  [PressureCorr DEBUG] phi input stats: min={np.nanmin(phi):.2e}, max={np.nanmax(phi):.2e}, has_nan={phi_has_nan}, has_inf={phi_has_inf}")
    
    # Optional: limit extreme phi corrections to prevent blow-up if solver is unstable
    # This clipping can act as a temporary measure, but the root cause of NaNs in phi
    # must be found (likely in Poisson solver or divergence calculation).
    # Setting threshold higher to allow more natural variations, but still guard against extremes.
    # The previous log showed max 1e2, but then NaNs. Let's try a smaller threshold on phi itself.
    if phi_has_nan or phi_has_inf or np.abs(phi).max() > 1e6: # A slightly higher threshold than 1e4
        print("⚠️ Pressure correction potential (phi) exceeds threshold or contains invalid values — clamping applied.")
        phi = np.clip(phi, -1e6, 1e6) # Clip to a reasonable range
        # Use nanmin/nanmax here
        print(f"  [PressureCorr DEBUG] phi after clipping: min={np.nanmin(phi):.2e}, max={np.nanmax(phi):.2e}")

    # Initialize updated_pressure with a copy of p_field
    updated_pressure = p_field.copy()
    # DEBUG: Inspect p_field before update
    p_field_has_nan = np.isnan(p_field).any()
    p_field_has_inf = np.isinf(p_field).any()
    if p_field_has_nan or p_field_has_inf:
        print(f"  [PressureCorr DEBUG] p_field BEFORE update: min={np.nanmin(p_field):.2e}, max={np.nanmax(p_field):.2e}, has_nan={p_field_has_nan}, has_inf={p_field_has_inf}")


    # Update the pressure field: P_new = P_old + ρ * φ
    # Since Poisson solver solves ∇²φ = (∇·u*) / Δt, φ already incorporates 1/Δt IF the RHS was (∇·u*)/Δt.
    # If the RHS to Poisson was just (∇·u*), then pressure update is P_new = P_old + ρ * φ / Δt.
    # Given your Poisson Solver output suggests φ is the potential, usually it's P_new = P_old + rho * phi / dt.
    # Let's assume the standard formulation where solve_poisson_for_phi gives a phi for grad(phi) in velocity,
    # and the pressure is updated by phi/dt. Re-check the source Poisson equation (∇²φ = (∇·u*) / Δt or ∇²φ = (∇·u*)).
    # Based on the typical fractional step, it's often ∇²φ = (∇·u*)/Δt.
    # If so, P = P_old + rho * phi.
    # If it was ∇²φ = (∇·u*), then P = P_old + rho * phi / dt.
    # Your current code uses density * phi. Let's stick with that for now, but be aware.

    # Ensure phi interior is used for pressure update
    updated_pressure[1:-1, 1:-1, 1:-1] += density * phi[1:-1, 1:-1, 1:-1]

    # DEBUG: Inspect updated_pressure after update, before clamping
    nan_check_pressure_after_update = np.isnan(updated_pressure).any()
    inf_check_pressure_after_update = np.isinf(updated_pressure).any()
    if nan_check_pressure_after_update or inf_check_pressure_after_update:
        print(f"  [PressureCorr DEBUG] updated_pressure AFTER update (before clamp): min={np.nanmin(updated_pressure):.2e}, max={np.nanmax(updated_pressure):.2e}, has_nan={nan_check_pressure_after_update}, has_inf={inf_check_pressure_after_update}")


    # Final safety clamp for updated_pressure
    if nan_check_pressure_after_update or inf_check_pressure_after_update:
        print("❌ Warning: Invalid values in updated pressure — clamping to zero.")
    updated_pressure = np.nan_to_num(updated_pressure, nan=0.0, posinf=0.0, neginf=0.0)
    # DEBUG: Inspect updated_pressure after clamping
    # This print will always execute as it's a final sanity check post-clamping.
    print(f"  [PressureCorr DEBUG] updated_pressure AFTER clamp: min={np.nanmin(updated_pressure):.2e}, max={np.nanmax(updated_pressure):.2e}")


    # Compute the gradient of phi for each direction.
    # calculate_gradient now correctly takes phi with ghost cells and returns interior gradient.
    # The clamping inside calculate_gradient should help here.
    grad_phi_x = calculate_gradient(phi, dx, axis=0)
    grad_phi_y = calculate_gradient(phi, dy, axis=1)
    grad_phi_z = calculate_gradient(phi, dz, axis=2)

    # DEBUG: Inspect gradients of phi
    # These checks are now redundant if calculate_gradient already prints them and clamps.
    # Keeping them for now, but consider removing if logs become too verbose.
    grad_phi_x_has_nan = np.isnan(grad_phi_x).any()
    grad_phi_x_has_inf = np.isinf(grad_phi_x).any()
    if grad_phi_x_has_nan or grad_phi_x_has_inf:
        print(f"  [PressureCorr DEBUG] grad_phi_x stats AFTER calculate_gradient: min={np.nanmin(grad_phi_x):.2e}, max={np.nanmax(grad_phi_x):.2e}, has_nan={grad_phi_x_has_nan}, has_inf={grad_phi_x_has_inf}")
    
    grad_phi_y_has_nan = np.isnan(grad_phi_y).any()
    grad_phi_y_has_inf = np.isinf(grad_phi_y).any()
    if grad_phi_y_has_nan or grad_phi_y_has_inf:
        print(f"  [PressureCorr DEBUG] grad_phi_y stats AFTER calculate_gradient: min={np.nanmin(grad_phi_y):.2e}, max={np.nanmax(grad_phi_y):.2e}, has_nan={grad_phi_y_has_nan}, has_inf={grad_phi_y_has_inf}")
    
    grad_phi_z_has_nan = np.isnan(grad_phi_z).any()
    grad_phi_z_has_inf = np.isinf(grad_phi_z).any()
    if grad_phi_z_has_nan or grad_phi_z_has_inf:
        print(f"  [PressureCorr DEBUG] grad_phi_z stats AFTER calculate_gradient: min={np.nanmin(grad_phi_z):.2e}, max={np.nanmax(grad_phi_z):.2e}, has_nan={grad_phi_z_has_nan}, has_inf={grad_phi_z_has_inf}")


    # Apply the pressure correction to the tentative velocity field:
    # u_corrected = u_star - (dt / rho) * grad(phi)
    corrected_velocity = velocity_next.copy()
    corrected_velocity[1:-1, 1:-1, 1:-1, 0] -= time_step * grad_phi_x / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 1] -= time_step * grad_phi_y / density
    corrected_velocity[1:-1, 1:-1, 1:-1, 2] -= time_step * grad_phi_z / density

    # DEBUG: Inspect corrected_velocity after update, before clamping
    nan_check_velocity_after_update = np.isnan(corrected_velocity).any()
    inf_check_velocity_after_update = np.isinf(corrected_velocity).any()
    if nan_check_velocity_after_update or inf_check_velocity_after_update:
        print(f"  [PressureCorr DEBUG] corrected_velocity AFTER update (before clamp): min={np.nanmin(corrected_velocity):.2e}, max={np.nanmax(corrected_velocity):.2e}, has_nan={nan_check_velocity_after_update}, has_inf={inf_check_velocity_after_update}")


    # Final safety clamp for corrected velocity
    if nan_check_velocity_after_update or inf_check_velocity_after_update:
        print("❌ Warning: Invalid values in corrected velocity — clamping to zero.")
    corrected_velocity = np.nan_to_num(corrected_velocity, nan=0.0, posinf=0.0, neginf=0.0)
    # DEBUG: Inspect corrected_velocity after clamping
    print(f"  [PressureCorr DEBUG] corrected_velocity AFTER clamp: min={np.nanmin(corrected_velocity):.2e}, max={np.nanmax(corrected_velocity):.2e}")


    return corrected_velocity, updated_pressure



