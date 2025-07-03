# src/numerical_methods/advection.py

import numpy as np

def compute_advection_term(u_field, velocity_field, mesh_info):
    """
    Computes the advection term - (u · ∇)u using a first-order upwind scheme.

    Args:
        u_field (np.ndarray): Scalar (nx, ny, nz) or vector (nx, ny, nz, 3) field to advect.
                              This is the field being advected (e.g., u, v, or w component).
        velocity_field (np.ndarray): The full velocity vector field of shape (nx, ny, nz, 3).
                                     Used to determine the advecting velocity components (u_x, u_y, u_z).
        mesh_info (dict): Dictionary with:
            - 'grid_shape': (nx, ny, nz)
            - 'dx', 'dy', 'dz': Grid spacing in each direction.

    Returns:
        np.ndarray: Advection term, same shape as u_field.
    """
    # Defensive clamping at the start to handle any incoming NaNs/Infs from previous steps
    u_field = np.nan_to_num(u_field, nan=0.0, posinf=0.0, neginf=0.0)
    velocity_field = np.nan_to_num(velocity_field, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.all(u_field.shape == velocity_field[..., 0].shape):
        print("⚠️ Shape mismatch between scalar/vector field and velocity components.")

    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    is_scalar = u_field.ndim == 3
    advection = np.zeros_like(u_field)

    # Extract advecting velocity components
    vel_x = velocity_field[..., 0] # u_x component for advection in x-direction
    vel_y = velocity_field[..., 1] # u_y component for advection in y-direction
    vel_z = velocity_field[..., 2] # u_z component for advection in z-direction

    def upwind_derivative(phi, vel_comp, axis, spacing):
        """
        Computes the first-order upwind derivative of phi along a given axis,
        based on the sign of vel_comp.

        Args:
            phi (np.ndarray): The scalar field (e.g., u, v, or w component) for which to compute the derivative.
                              Assumed to include ghost cells.
            vel_comp (np.ndarray): The velocity component along the 'axis' direction.
                                   Used to determine the upwind direction.
            axis (int): The axis along which to compute the derivative (0 for x, 1 for y, 2 for z).
            spacing (float): The grid spacing (dx, dy, or dz) for the given axis.

        Returns:
            np.ndarray: The upwind derivative of phi, same shape as phi.
        """
        deriv = np.zeros_like(phi)

        # Slices for interior cells where the derivative is computed (index 'i')
        interior_slice = [slice(None)] * phi.ndim
        interior_slice[axis] = slice(1, -1) # Represents index 'i'

        # Slices for neighbor cells
        prev_slice = [slice(None)] * phi.ndim
        prev_slice[axis] = slice(0, -2) # Represents index 'i-1'

        next_slice = [slice(None)] * phi.ndim
        next_slice[axis] = slice(2, None) # Represents index 'i+1'

        # Calculate backward difference: (phi_i - phi_{i-1}) / spacing
        # Used when vel_comp >= 0 (flow from i-1 to i)
        backward_diff = (phi[tuple(interior_slice)] - phi[tuple(prev_slice)]) / spacing

        # Calculate forward difference: (phi_{i+1} - phi_i) / spacing
        # Used when vel_comp < 0 (flow from i+1 to i)
        forward_diff = (phi[tuple(next_slice)] - phi[tuple(interior_slice)]) / spacing

        # Determine the upwind direction based on the velocity component's sign
        # Use >= 0 for positive velocity, meaning information comes from 'previous' cell
        pos_vel_mask = vel_comp[tuple(interior_slice)] >= 0

        # Apply upwinding:
        # If velocity is positive (or zero), use backward difference.
        # Else (if velocity is negative), use forward difference.
        deriv[tuple(interior_slice)] = np.where(
            pos_vel_mask,
            backward_diff,
            forward_diff
        )
        
        # --- Boundary Handling for the 'deriv' array itself ---
        # The above calculation only fills the interior of 'deriv'.
        # The outermost layers of 'deriv' (corresponding to ghost cells of 'phi')
        # need to be handled. For simplicity, we extend the nearest interior value.
        # This assumes that the 'phi' input already has valid ghost cells from
        # apply_boundary_conditions.

        # Fill the first slice with the value from the second slice along the axis
        if phi.shape[axis] > 1: # Ensure there's more than one cell to copy from
            first_deriv_slice = [slice(None)] * phi.ndim
            first_deriv_slice[axis] = 0
            second_deriv_slice = [slice(None)] * phi.ndim
            second_deriv_slice[axis] = 1
            deriv[tuple(first_deriv_slice)] = deriv[tuple(second_deriv_slice)]
        
        # Fill the last slice with the value from the second-to-last slice along the axis
        if phi.shape[axis] > 1:
            last_deriv_slice = [slice(None)] * phi.ndim
            last_deriv_slice[axis] = -1
            second_last_deriv_slice = [slice(None)] * phi.ndim
            second_last_deriv_slice[axis] = -2
            deriv[tuple(last_deriv_slice)] = deriv[tuple(second_last_deriv_slice)]

        return deriv
    # --- END CORRECTED UPWIND DERIVATIVE FUNCTION ---

    # Calculate the advection term (u · ∇)u
    # This involves summing the contributions from each direction (x, y, z)
    # for each component of the advected field (u_x, u_y, u_z for momentum equations).
    if is_scalar:
        # For advecting a scalar field (e.g., temperature, concentration)
        # (u_x * d(u_field)/dx) + (u_y * d(u_field)/dy) + (u_z * d(u_field)/dz)
        advection += vel_x * upwind_derivative(u_field, vel_x, axis=0, spacing=dx)
        advection += vel_y * upwind_derivative(u_field, vel_y, axis=1, spacing=dy)
        advection += vel_z * upwind_derivative(u_field, vel_z, axis=2, spacing=dz)
    else:
        # For advecting a vector field (momentum equations for u, v, w components)
        # For each component 'uc' (u_x, u_y, or u_z):
        # (u_x * d(uc)/dx) + (u_y * d(uc)/dy) + (u_z * d(uc)/dz)
        for comp in range(3):
            uc = u_field[..., comp] # Get the specific velocity component (u_x, u_y, or u_z) being advected
            
            # Advection contribution from x-direction velocity (vel_x)
            advection[..., comp] += vel_x * upwind_derivative(uc, vel_x, axis=0, spacing=dx)
            
            # Advection contribution from y-direction velocity (vel_y)
            advection[..., comp] += vel_y * upwind_derivative(uc, vel_y, axis=1, spacing=dy)
            
            # Advection contribution from z-direction velocity (vel_z)
            advection[..., comp] += vel_z * upwind_derivative(uc, vel_z, axis=2, spacing=dz)

    # Final check for NaNs/Infs before returning, and clamp if necessary
    if np.isnan(advection).any() or np.isinf(advection).any():
        print(f"   Advection stats before clamp: min={np.min(advection):.2e}, max={np.max(advection):.2e}")
        print("❌ Warning: NaNs or infs detected in advection term — clamping to zero.")
    advection = np.nan_to_num(advection, nan=0.0, posinf=0.0, neginf=0.0)

    # In Navier-Stokes, the advection term is typically subtracted from the velocity update.
    # So, if 'advection' computes (u . grad)u, we return -(u . grad)u.
    return -advection


# The following function 'advect_velocity' is not currently used by ExplicitSolver.step
# and appears to be for a different grid setup (staggered MAC grid).
# It's commented out to avoid confusion and keep the focus on the primary execution path.
# If you intend to use a MAC grid or this function later, it will need to be re-evaluated
# and potentially adapted to the new upwind_derivative logic and your overall solver structure.
# def advect_velocity(u, v, w, dx, dy, dz, dt):
#     """
#     Performs forward Euler advection of velocity components using a staggered MAC grid.
#     """
#     interior = (slice(1, -1), slice(1, -1), slice(1, -1))
#     velocity = np.stack([u, v, w], axis=-1)

#     cfl_x = np.max(np.abs(velocity[..., 0])) * dt / dx
#     cfl_y = np.max(np.abs(velocity[..., 1])) * dt / dy
#     cfl_z = np.max(np.abs(velocity[..., 2])) * dt / dz
#     print(f"   CFL conditions: X={cfl_x:.2f}, Y={cfl_y:.2f}, Z={cfl_z:.2f}")

#     mesh_info = {
#         "grid_shape": u.shape,
#         "dx": dx,
#         "dy": dy,
#         "dz": dz,
#     }

#     u_star = u[interior] + dt * compute_advection_term(u, velocity, mesh_info)[interior]
#     v_star = v[interior] + dt * compute_advection_term(v, velocity, mesh_info)[interior]
#     w_star = w[interior] + dt * compute_advection_term(w, velocity, mesh_info)[interior]

#     return u_star, v_star, w_star




