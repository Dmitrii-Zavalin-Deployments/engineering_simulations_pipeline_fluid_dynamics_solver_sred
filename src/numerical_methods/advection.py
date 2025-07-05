# src/numerical_methods/advection.py

import numpy as np

def compute_advection_term(u_field, velocity_field, mesh_info):
    """
    Computes the advection term - (u · ∇)u using a first-order upwind scheme.

    Args:
        u_field (np.ndarray): The specific scalar field (e.g., u_x, u_y, or u_z component)
                              being advected, including ghost cells. Shape (nx, ny, nz).
        velocity_field (np.ndarray): The full velocity vector field of shape (nx, ny, nz, 3),
                                     including ghost cells. Used to determine the advecting
                                     velocity components (u_x, u_y, u_z).
        mesh_info (dict): Dictionary with:
            - 'grid_shape': (nx, ny, nz)
            - 'dx', 'dy', 'dz': Grid spacing in each direction.

    Returns:
        np.ndarray: The advection term, same shape as the interior of u_field (nx-2, ny-2, nz-2).
                    This represents -(u · ∇)u.
    """
    # Defensive clamping at the start to handle any incoming NaNs/Infs from previous steps.
    # While we aim to eliminate NaNs at their source, this acts as a safeguard.
    u_field_has_nan = np.isnan(u_field).any()
    u_field_has_inf = np.isinf(u_field).any()
    velocity_field_has_nan = np.isnan(velocity_field).any()
    velocity_field_has_inf = np.isinf(velocity_field).any()

    if u_field_has_nan or u_field_has_inf or velocity_field_has_nan or velocity_field_has_inf:
        print(f"    [Advection DEBUG] u_field input stats BEFORE clamp: min={np.nanmin(u_field):.2e}, max={np.nanmax(u_field):.2e}, has_nan={u_field_has_nan}, has_inf={u_field_has_inf}")
        print(f"    [Advection DEBUG] velocity_field input stats BEFORE clamp: min={np.nanmin(velocity_field):.2e}, max={np.nanmax(velocity_field):.2e}, has_nan={velocity_field_has_nan}, has_inf={velocity_field_has_inf}")

    u_field = np.nan_to_num(u_field, nan=0.0, posinf=0.0, neginf=0.0)
    velocity_field = np.nan_to_num(velocity_field, nan=0.0, posinf=0.0, neginf=0.0)

    if u_field_has_nan or u_field_has_inf or velocity_field_has_nan or velocity_field_has_inf:
        print(f"    [Advection DEBUG] u_field input stats AFTER clamp: min={np.min(u_field):.2e}, max={np.max(u_field):.2e}")
        print(f"    [Advection DEBUG] velocity_field input stats AFTER clamp: min={np.min(velocity_field):.2e}, max={np.max(velocity_field):.2e}")

    # Ensure u_field is 3D (scalar component)
    if u_field.ndim != 3:
        raise ValueError("u_field must be a 3D scalar array (e.g., u_x component) including ghost cells.")
    if velocity_field.ndim != 4 or velocity_field.shape[-1] != 3:
        raise ValueError("velocity_field must be a 4D vector array (nx, ny, nz, 3) including ghost cells.")

    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    # Extract advecting velocity components (these are the velocities that are doing the advecting)
    # These are extracted from the full velocity_field.
    vel_x_advecting = velocity_field[..., 0] # u_x component
    vel_y_advecting = velocity_field[..., 1] # u_y component
    vel_z_advecting = velocity_field[..., 2] # u_z component

    # Initialize the advection term for the interior cells
    # This result will be of shape (nx-2, ny-2, nz-2)
    advection_term_interior = np.zeros(u_field[1:-1, 1:-1, 1:-1].shape, dtype=u_field.dtype)

    def upwind_derivative(phi_field, vel_comp_field, axis, spacing):
        """
        Computes the first-order upwind derivative of phi_field along a given axis,
        based on the sign of vel_comp_field.

        Args:
            phi_field (np.ndarray): The scalar field (e.g., u, v, or w component) for which to
                                    compute the derivative. Assumed to include ghost cells.
            vel_comp_field (np.ndarray): The velocity component along the 'axis' direction
                                         used to determine the upwind direction. Assumed to
                                         include ghost cells.
            axis (int): The axis along which to compute the derivative (0 for x, 1 for y, 2 for z).
            spacing (float): The grid spacing (dx, dy, or dz) for the given axis.

        Returns:
            np.ndarray: The upwind derivative of phi_field for the interior cells,
                        shape (nx-2, ny-2, nz-2).
        """
        # Slices for interior cells (where derivative is computed)
        interior_slice = [slice(1, -1)] * phi_field.ndim
        # Slices for neighbor cells (relative to interior_slice along the 'axis')
        prev_slice = list(interior_slice) # slice(0, -2) for 'i-1'
        prev_slice[axis] = slice(0, -2)
        next_slice = list(interior_slice) # slice(2, None) for 'i+1'
        next_slice[axis] = slice(2, None)

        # Get views of the relevant parts of the fields for interior calculations
        phi_interior = phi_field[tuple(interior_slice)]
        vel_comp_interior = vel_comp_field[tuple(interior_slice)]

        # Backward difference (phi_i - phi_{i-1}) / spacing; used when vel_comp >= 0 (flow from i-1 to i)
        backward_diff = (phi_interior - phi_field[tuple(prev_slice)]) / spacing

        # Forward difference (phi_{i+1} - phi_i) / spacing; used when vel_comp < 0 (flow from i+1 to i)
        forward_diff = (phi_field[tuple(next_slice)] - phi_interior) / spacing

        # Determine the upwind direction based on the velocity component's sign
        pos_vel_mask = vel_comp_interior >= 0

        # Apply upwinding:
        deriv_interior = np.where(
            pos_vel_mask,
            backward_diff,
            forward_diff
        )

        # DEBUG: Check for NaNs/Infs in intermediate derivative results
        deriv_interior_has_nan = np.isnan(deriv_interior).any()
        deriv_interior_has_inf = np.isinf(deriv_interior).any()

        if deriv_interior_has_nan or deriv_interior_has_inf:
            print(f"      [Upwind DEBUG] Axis {axis}, phi_field interior min={np.nanmin(phi_interior):.2e}, max={np.nanmax(phi_interior):.2e}")
            print(f"      [Upwind DEBUG] Axis {axis}, vel_comp_field interior min={np.nanmin(vel_comp_interior):.2e}, max={np.nanmax(vel_comp_interior):.2e}")
            print(f"      [Upwind DEBUG] Axis {axis}, backward_diff stats: min={np.nanmin(backward_diff):.2e}, max={np.nanmax(backward_diff):.2e}, has_nan={np.isnan(backward_diff).any()}, has_inf={np.isinf(backward_diff).any()}")
            print(f"      [Upwind DEBUG] Axis {axis}, forward_diff stats: min={np.nanmin(forward_diff):.2e}, max={np.nanmax(forward_diff):.2e}, has_nan={np.isnan(forward_diff).any()}, has_inf={np.isinf(forward_diff).any()}")
            print(f"      [Upwind DEBUG] Axis {axis}, deriv (interior) after upwind: min={np.nanmin(deriv_interior):.2e}, max={np.nanmax(deriv_interior):.2e}, has_nan={deriv_interior_has_nan}, has_inf={deriv_interior_has_inf}")

        # Return only the interior part of the derivative
        return deriv_interior

    # Slices for the interior of the advecting velocity components to match derivative output shape
    vel_x_interior = vel_x_advecting[1:-1, 1:-1, 1:-1]
    vel_y_interior = vel_y_advecting[1:-1, 1:-1, 1:-1]
    vel_z_interior = vel_z_advecting[1:-1, 1:-1, 1:-1]


    # If u_field is a scalar (e.g., for advecting a scalar quantity like temperature)
    # This path is actually for advecting a *component* of velocity, not a generic scalar.
    # The original intent of 'u_field' being a scalar or vector was confusing.
    # In Navier-Stokes, we advect u_x by (u_x, u_y, u_z), u_y by (u_x, u_y, u_z), etc.
    # So u_field should always be a 3D component.

    # We assume u_field represents one of the velocity components (u_x, u_y, or u_z)
    # The loop below handles the case where u_field is part of a 4D velocity array
    # and compute_advection_term is called for each component.

    # Calculate the advection term (u · ∇)u
    # This involves summing the contributions from each direction (x, y, z)
    # for the current component (u_field) being advected.

    # Advection contribution from x-direction velocity (vel_x)
    advection_term_interior += vel_x_interior * upwind_derivative(u_field, vel_x_advecting, axis=0, spacing=dx)
    
    # Advection contribution from y-direction velocity (vel_y)
    advection_term_interior += vel_y_interior * upwind_derivative(u_field, vel_y_advecting, axis=1, spacing=dy)
    
    # Advection contribution from z-direction velocity (vel_z)
    advection_term_interior += vel_z_interior * upwind_derivative(u_field, vel_z_advecting, axis=2, spacing=dz)


    # Final check for NaNs/Infs in the computed advection term.
    # For a stable scheme, these should ideally not appear here.
    advection_final_has_nan = np.isnan(advection_term_interior).any()
    advection_final_has_inf = np.isinf(advection_term_interior).any()

    if advection_final_has_nan or advection_final_has_inf:
        print(f"    Advection stats before final output (interior): min={np.nanmin(advection_term_interior):.2e}, max={np.nanmax(advection_term_interior):.2e}, has_nan={advection_final_has_nan}, has_inf={advection_final_has_inf}")
        # If you truly want to remove all clamping, comment out the line below.
        # However, for robustness against extreme edge cases, keeping it might be safer initially.
        advection_term_interior = np.nan_to_num(advection_term_interior, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"    Advection stats AFTER final output clamp (interior): min={np.min(advection_term_interior):.2e}, max={np.max(advection_term_interior):.2e}")


    # In Navier-Stokes, the advection term (u · ∇)u is typically subtracted in the momentum equation.
    # So, if this function computes (u · ∇)u, we return -(u · ∇)u for the RHS.
    return -advection_term_interior


# The following function 'advect_velocity' is for a different purpose/structure
# and is commented out to avoid confusion and keep the focus on the primary execution path
# used by your main solver.
# def advect_velocity(u, v, w, dx, dy, dz, dt):
#     """
#     Performs forward Euler advection of velocity components using a staggered MAC grid.
#     """
#     interior = (slice(1, -1), slice(1, -1), slice(1, -1))
#     velocity = np.stack([u, v, w], axis=-1)

#     cfl_x = np.max(np.abs(velocity[..., 0])) * dt / dx
#     cfl_y = np.max(np.abs(velocity[..., 1])) * dt / dy
#     cfl_z = np.max(np.abs(velocity[..., 2])) * dt / dz
#     print(f"    CFL conditions: X={cfl_x:.2f}, Y={cfl_y:.2f}, Z={cfl_z:.2f}")

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



