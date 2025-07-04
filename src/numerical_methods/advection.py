# src/numerical_methods/advection.py

import numpy as np
import sys

def _check_nan_inf(field, name, step_number, output_frequency_steps):
    """Helper to check for NaN/Inf and print debug info conditionally."""
    # Only print debug info if the current step is an output step
    if step_number % output_frequency_steps == 0:
        has_nan = np.isnan(field).any()
        has_inf = np.isinf(field).any()
        min_val = np.min(field) if not has_nan and not has_inf else float('nan')
        max_val = np.max(field) if not has_nan and not has_inf else float('nan')
        print(f"  [Advection DEBUG] {name} stats BEFORE clamp: min={min_val:.2e}, max={max_val:.2e}, has_nan={has_nan}, has_inf={has_inf}")

    # Always clamp to prevent propagation, regardless of output frequency
    if np.isnan(field).any() or np.isinf(field).any():
        if step_number % output_frequency_steps == 0: # Only print warning if it's an output step
            print(f"  ❌ Warning: Invalid values in {name} — clamping to zero.")
        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    if step_number % output_frequency_steps == 0: # Only print debug info if it's an output step
        print(f"  [Advection DEBUG] {name} stats AFTER clamp: min={np.min(field):.2e}, max={np.max(field):.2e}")
    return field


def _upwind_derivative(phi, vel_comp, h, axis, step_number, output_frequency_steps):
    """
    Computes the upwind finite difference derivative for a given axis.
    Assumes phi and vel_comp already include ghost cells.
    Returns the derivative for the interior cells.
    """
    # Determine slices for interior, previous, and next cells along the given axis
    interior_slice = [slice(1, -1)] * 3
    prev_slice = [slice(1, -1)] * 3
    next_slice = [slice(1, -1)] * 3

    interior_slice[axis] = slice(1, -1) # e.g., [1:-1, 1:-1, 1:-1]
    prev_slice[axis] = slice(0, -2)     # e.g., [0:-2, 1:-1, 1:-1]
    next_slice[axis] = slice(2, None)   # e.g., [2:, 1:-1, 1:-1]

    # Convert to tuples for indexing
    interior_slice = tuple(interior_slice)
    prev_slice = tuple(prev_slice)
    next_slice = tuple(next_slice)

    # Debugging slices - only print if it's an output step
    if step_number % output_frequency_steps == 0:
        print(f"    [Upwind DEBUG] Axis {axis}, phi[interior_slice] min={np.min(phi[interior_slice]):.2e}, max={np.max(phi[interior_slice]):.2e}")
        print(f"    [Upwind DEBUG] Axis {axis}, phi[prev_slice] min={np.min(phi[prev_slice]):.2e}, max={np.max(phi[prev_slice]):.2e}")
        print(f"    [Upwind DEBUG] Axis {axis}, phi[next_slice] min={np.min(phi[next_slice]):.2e}, max={np.max(phi[next_slice]):.2e}")
        print(f"    [Upwind DEBUG] Axis {axis}, vel_comp[interior_slice] min={np.min(vel_comp[interior_slice]):.2e}, max={np.max(vel_comp[interior_slice]):.2e}")


    # Calculate forward and backward differences for interior cells
    backward_diff = (phi[interior_slice] - phi[prev_slice]) / h
    forward_diff = (phi[next_slice] - phi[interior_slice]) / h

    if step_number % output_frequency_steps == 0: # Only print debug info if it's an output step
        has_nan_b = np.isnan(backward_diff).any()
        has_inf_b = np.isinf(backward_diff).any()
        min_b = np.min(backward_diff) if not has_nan_b and not has_inf_b else float('nan')
        max_b = np.max(backward_diff) if not has_nan_b and not has_inf_b else float('nan')
        print(f"    [Upwind DEBUG] Axis {axis}, backward_diff stats: min={min_b:.2e}, max={max_b:.2e}, has_nan={has_nan_b}, has_inf={has_inf_b}")

        has_nan_f = np.isnan(forward_diff).any()
        has_inf_f = np.isinf(forward_diff).any()
        min_f = np.min(forward_diff) if not has_nan_f and not has_inf_f else float('nan')
        max_f = np.max(forward_diff) if not has_nan_f and not has_inf_f else float('nan')
        print(f"    [Upwind DEBUG] Axis {axis}, forward_diff stats: min={min_f:.2e}, max={max_f:.2e}, has_nan={has_nan_f}, has_inf={has_inf_f}")

    # Initialize derivative array for interior cells
    deriv = np.zeros_like(phi[interior_slice])

    # Apply upwind scheme based on the sign of the velocity component
    # Where velocity is positive, use backward difference (phi_i - phi_{i-1}) / h
    pos_vel_mask = vel_comp[interior_slice] >= 0
    deriv[pos_vel_mask] = backward_diff[pos_vel_mask]

    # Where velocity is negative, use forward difference (phi_{i+1} - phi_i) / h
    neg_vel_mask = vel_comp[interior_slice] < 0
    deriv[neg_vel_mask] = forward_diff[neg_vel_mask]

    if step_number % output_frequency_steps == 0: # Only print debug info if it's an output step
        has_nan_d = np.isnan(deriv).any()
        has_inf_d = np.isinf(deriv).any()
        min_d = np.min(deriv) if not has_nan_d and not has_inf_d else float('nan')
        max_d = np.max(deriv) if not has_nan_d and not has_inf_d else float('nan')
        print(f"    [Upwind DEBUG] Axis {axis}, deriv (interior) after upwind: min={min_d:.2e}, max={max_d:.2e}, has_nan={has_nan_d}, has_inf={has_inf_d}")

    # Clamp the derivative for safety before returning
    if np.isnan(deriv).any() or np.isinf(deriv).any():
        if step_number % output_frequency_steps == 0: # Only print warning if it's an output step
            print(f"    ❌ Warning: Invalid values in upwind derivative axis {axis} — clamping to zero.")
        deriv = np.nan_to_num(deriv, nan=0.0, posinf=0.0, neginf=0.0)

    return deriv


def compute_advection_diffusion(velocity_field, nu, time_step, mesh_info, step_number, output_frequency_steps):
    """
    Computes the advection and diffusion terms for the Navier-Stokes equations.
    This function calculates the explicit part of the velocity update (u*).

    Args:
        velocity_field (np.ndarray): Current velocity field (u, v, w components) with ghost cells.
                                     Shape: (nx+2, ny+2, nz+2, 3).
        nu (float): Kinematic viscosity.
        time_step (float): Time step Δt.
        mesh_info (dict): Dictionary containing grid spacing 'dx', 'dy', 'dz'.
        step_number (int): Current simulation step number.
        output_frequency_steps (int): Frequency for printing debug output.

    Returns:
        np.ndarray: Tentative velocity field (u*) with ghost cells.
                    Shape: (nx+2, ny+2, nz+2, 3).
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    # Create a copy for the tentative velocity field (u*)
    # We will update the interior cells of this copy.
    velocity_star = velocity_field.copy()

    if step_number % output_frequency_steps == 0:
        print("  1. Computing advection and diffusion terms...")

    # Iterate over each velocity component (u, v, w)
    for i in range(3): # 0 for u, 1 for v, 2 for w
        u_field = velocity_field[:, :, :, i] # The component being advected/diffused

        # Check and clamp input field before calculations
        u_field = _check_nan_inf(u_field, f"u_field (component {i}) input", step_number, output_frequency_steps)
        # velocity_field (all components) is also checked to ensure consistency
        # No need to re-check velocity_field here as it's the same object and already checked by _check_nan_inf for u_field
        # _ = _check_nan_inf(velocity_field, "velocity_field input", step_number, output_frequency_steps)


        # Compute advection terms for the current component (u_field)
        # Advection term: u * du/dx + v * du/dy + w * du/dz
        # Use upwind scheme for stability
        advection_x = velocity_field[1:-1, 1:-1, 1:-1, 0] * _upwind_derivative(u_field, velocity_field[:, :, :, 0], dx, 0, step_number, output_frequency_steps)
        advection_y = velocity_field[1:-1, 1:-1, 1:-1, 1] * _upwind_derivative(u_field, velocity_field[:, :, :, 1], dy, 1, step_number, output_frequency_steps)
        advection_z = velocity_field[1:-1, 1:-1, 1:-1, 2] * _upwind_derivative(u_field, velocity_field[:, :, :, 2], dz, 2, step_number, output_frequency_steps)

        advection_term = advection_x + advection_y + advection_z

        # Clamp advection_term for safety
        if np.isnan(advection_term).any() or np.isinf(advection_term).any():
            if step_number % output_frequency_steps == 0: # Only print warning if it's an output step
                print(f"  ❌ Warning: Invalid values in advection_term (component {i}) — clamping to zero.")
            advection_term = np.nan_to_num(advection_term, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute diffusion terms for the current component (u_field)
        # Diffusion term: nu * (d^2u/dx^2 + d^2u/dy^2 + d^2u/dz^2)
        # This part is typically handled by a separate diffusion function.
        # For now, keeping it here for completeness as per previous code.
        laplacian_x = (u_field[2:, 1:-1, 1:-1] - 2 * u_field[1:-1, 1:-1, 1:-1] + u_field[:-2, 1:-1, 1:-1]) / (dx**2)
        laplacian_y = (u_field[1:-1, 2:, 1:-1] - 2 * u_field[1:-1, 1:-1, 1:-1] + u_field[1:-1, :-2, 1:-1]) / (dy**2)
        laplacian_z = (u_field[1:-1, 1:-1, 2:] - 2 * u_field[1:-1, 1:-1, 1:-1] + u_field[1:-1, 1:-1, :-2]) / (dz**2)

        diffusion_term = nu * (laplacian_x + laplacian_y + laplacian_z)

        # Clamp diffusion_term for safety
        if np.isnan(diffusion_term).any() or np.isinf(diffusion_term).any():
            if step_number % output_frequency_steps == 0: # Only print warning if it's an output step
                print(f"  ❌ Warning: Invalid values in diffusion_term (component {i}) — clamping to zero.")
            diffusion_term = np.nan_to_num(diffusion_term, nan=0.0, posinf=0.0, neginf=0.0)

        # Update the tentative velocity field (u*) for interior cells
        # u* = u^n + dt * (- (u . grad)u + nu * grad^2 u)
        velocity_star[1:-1, 1:-1, 1:-1, i] = \
            u_field[1:-1, 1:-1, 1:-1] + time_step * (-advection_term + diffusion_term)

        # Clamp velocity_star component after update
        velocity_star[:, :, :, i] = _check_nan_inf(velocity_star[:, :, :, i], f"velocity_star (component {i})", step_number, output_frequency_steps)

    return velocity_star



