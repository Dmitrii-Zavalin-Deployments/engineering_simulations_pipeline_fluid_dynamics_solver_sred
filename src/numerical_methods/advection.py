# src/numerical_methods/advection.py

import numpy as np
from numba import jit, float64, intc # Import intc for integer types

# --- Numba-jitted helper for upwind derivative ---
@jit(
    float64[:, :, :](
        float64[:, :, :],  # phi_field (e.g., u_x, u_y, u_z component)
        float64[:, :, :],  # vel_comp_field (advecting velocity component along axis)
        float64,           # spacing (dx, dy, or dz)
        intc               # axis (0, 1, or 2) -- CHANGED FROM 'int' to 'intc'
    ),
    nopython=True,
    parallel=False,
    cache=True
)
def _upwind_derivative_kernel(phi_field, vel_comp_field, spacing, axis):
    """
    Numba-jitted kernel to compute the first-order upwind derivative of phi_field
    along a given axis, based on the sign of vel_comp_field.
    Operates on the interior cells of the output array.
    """
    
    # Get interior dimensions for the output array
    nx_interior = phi_field.shape[0] - 2
    ny_interior = phi_field.shape[1] - 2
    nz_interior = phi_field.shape[2] - 2
    
    deriv_interior = np.empty((nx_interior, ny_interior, nz_interior), dtype=phi_field.dtype)

    # Manual iteration for Numba compatibility and performance
    for k in range(nz_interior):
        for j in range(ny_interior):
            for i in range(nx_interior):
                # Map interior indices (i,j,k) to global indices (i+1, j+1, k+1)
                gi, gj, gk = i + 1, j + 1, k + 1

                vel = vel_comp_field[gi, gj, gk]
                phi_current = phi_field[gi, gj, gk]

                if axis == 0: # Derivative with respect to x
                    if vel >= 0: # Upwind from left (i-1)
                        deriv_interior[i, j, k] = (phi_current - phi_field[gi - 1, gj, gk]) / spacing
                    else: # Upwind from right (i+1)
                        deriv_interior[i, j, k] = (phi_field[gi + 1, gj, gk] - phi_current) / spacing
                elif axis == 1: # Derivative with respect to y
                    if vel >= 0: # Upwind from back (j-1)
                        deriv_interior[i, j, k] = (phi_current - phi_field[gi, gj - 1, gk]) / spacing
                    else: # Upwind from front (j+1)
                        deriv_interior[i, j, k] = (phi_field[gi, gj + 1, gk] - phi_current) / spacing
                elif axis == 2: # Derivative with respect to z
                    if vel >= 0: # Upwind from bottom (k-1)
                        deriv_interior[i, j, k] = (phi_current - phi_field[gi, gj, gk - 1]) / spacing
                    else: # Upwind from top (k+1)
                        deriv_interior[i, j, k] = (phi_field[gi, gj, gk + 1] - phi_current) / spacing
    
    return deriv_interior

# --- Main Advection Term Computation Function ---
def compute_advection_term(u_field, velocity_field, mesh_info):
    """
    Computes the advection term - (u · ∇)u using a first-order upwind scheme.

    Args:
        u_field (np.ndarray): The specific scalar field (e.g., u_x, u_y, or u_z component)
                              being advected, including ghost cells. Shape (nx_total, ny_total, nz_total).
        velocity_field (np.ndarray): The full velocity vector field of shape (nx_total, ny_total, nz_total, 3),
                                     including ghost cells. Used to determine the advecting
                                     velocity components (u_x, u_y, u_z).
        mesh_info (dict): Dictionary with:
            - 'grid_shape': (nx_interior, ny_interior, nz_interior)
            - 'dx', 'dy', 'dz': Grid spacing in each direction.

    Returns:
        np.ndarray: The advection term, same shape as the interior of u_field (nx_interior, ny_interior, nz_interior).
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

    # Ensure u_field is 3D (scalar component like u_x, u_y, or u_z)
    if u_field.ndim != 3:
        raise ValueError("u_field must be a 3D scalar array (e.g., u_x component) including ghost cells.")
    if velocity_field.ndim != 4 or velocity_field.shape[-1] != 3:
        raise ValueError("velocity_field must be a 4D vector array (nx_total, ny_total, nz_total, 3) including ghost cells.")

    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    # Extract advecting velocity components (these are the velocities that are doing the advecting)
    vel_x_advecting = velocity_field[..., 0] # u_x component
    vel_y_advecting = velocity_field[..., 1] # u_y component
    vel_z_advecting = velocity_field[..., 2] # u_z component
    
    # Get interior dimensions for the output advection term
    nx_interior, ny_interior, nz_interior = mesh_info['grid_shape']

    # Initialize the advection term for the interior cells
    advection_term_interior = np.zeros((nx_interior, ny_interior, nz_interior), dtype=u_field.dtype)

    # Slices for the interior of the advecting velocity components to match derivative output shape
    vel_x_interior = vel_x_advecting[1:-1, 1:-1, 1:-1]
    vel_y_interior = vel_y_advecting[1:-1, 1:-1, 1:-1]
    vel_z_interior = vel_z_advecting[1:-1, 1:-1, 1:-1]

    # Calculate the advection term (u · ∇)u
    # This involves summing the contributions from each direction (x, y, z)
    # for the current component (u_field) being advected.

    # Advection contribution from x-direction velocity (vel_x)
    advection_term_interior += vel_x_interior * _upwind_derivative_kernel(u_field, vel_x_advecting, dx, 0)
    
    # Advection contribution from y-direction velocity (vel_y)
    advection_term_interior += vel_y_interior * _upwind_derivative_kernel(u_field, vel_y_advecting, dy, 1)
    
    # Advection contribution from z-direction velocity (vel_z)
    advection_term_interior += vel_z_interior * _upwind_derivative_kernel(u_field, vel_z_advecting, dz, 2)

    # Final check for NaNs/Infs in the computed advection term.
    advection_final_has_nan = np.isnan(advection_term_interior).any()
    advection_final_has_inf = np.isinf(advection_term_interior).any()

    if advection_final_has_nan or advection_final_has_inf:
        print(f"    Advection stats before final output (interior): min={np.nanmin(advection_term_interior):.2e}, max={np.nanmax(advection_term_interior):.2e}, has_nan={advection_final_has_nan}, has_inf={advection_final_has_inf}")
        # Clamping here is a safeguard. Ideally, previous steps should prevent these.
        advection_term_interior = np.nan_to_num(advection_term_interior, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"    Advection stats AFTER final output clamp (interior): min={np.min(advection_term_interior):.2e}, max={np.max(advection_term_interior):.2e}")

    # In Navier-Stokes, the advection term (u · ∇)u is typically subtracted in the momentum equation.
    # So, if this function computes (u · ∇)u, we return -(u · ∇)u for the RHS.
    return -advection_term_interior



