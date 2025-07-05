# src/numerical_methods/pressure_divergence.py

import numpy as np

def compute_pressure_divergence(u_field, mesh_info):
    """
    Computes the divergence of the velocity field (∇·u) using central differencing.
    This function calculates the divergence for the interior cells of the domain.

    Args:
        u_field (np.ndarray): Velocity field with ghost cells, shape (nx+2, ny+2, nz+2, 3).
                              Assumed to have correct boundary conditions applied to ghost cells.
        mesh_info (dict): Dictionary containing grid spacing: 'dx', 'dy', 'dz'.

    Returns:
        np.ndarray: Divergence field of shape (nx, ny, nz) (interior cells only).
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # DEBUG: Check incoming u_field for NaNs/Infs
    u_field_has_nan = np.isnan(u_field).any()
    u_field_has_inf = np.isinf(u_field).any()
    if u_field_has_nan or u_field_has_inf:
        print(f"[Divergence DEBUG] Input u_field stats BEFORE component clamp: min={np.nanmin(u_field):.2e}, max={np.nanmax(u_field):.2e}, has_nan={u_field_has_nan}, has_inf={u_field_has_inf}")

    # Extract individual velocity components and defensively clamp any NaNs/Infs
    u = np.nan_to_num(u_field[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(u_field[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(u_field[..., 2], nan=0.0, posinf=0.0, neginf=0.0)

    # DEBUG: Check clamped components
    if u_field_has_nan or u_field_has_inf: # Only print if original had issues
        print(f"[Divergence DEBUG] Clamped u,v,w components stats: u_min={np.nanmin(u):.2e}, u_max={np.nanmax(u):.2e}, v_min={np.nanmin(v):.2e}, v_max={np.nanmax(v):.2e}, w_min={np.nanmin(w):.2e}, w_max={np.nanmax(w):.2e}")


    # Compute partial derivatives using central differences for the interior domain.
    # The slicing [1:-1, 1:-1, 1:-1] implicitly selects the interior cells for which
    # the divergence is calculated. The [2:] and [:-2] access the i+1 and i-1 neighbors.
    du_dx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
    dv_dy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
    dw_dz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)

    # DEBUG: Check partial derivatives
    partial_deriv_has_nan = np.isnan(du_dx).any() or np.isnan(dv_dy).any() or np.isnan(dw_dz).any()
    partial_deriv_has_inf = np.isinf(du_dx).any() or np.isinf(dv_dy).any() or np.isinf(dw_dz).any()
    if partial_deriv_has_nan or partial_deriv_has_inf:
        print(f"[Divergence DEBUG] Partial derivatives stats (before sum):")
        print(f"  du_dx: min={np.nanmin(du_dx):.2e}, max={np.nanmax(du_dx):.2e}, has_nan={np.isnan(du_dx).any()}, has_inf={np.isinf(du_dx).any()}")
        print(f"  dv_dy: min={np.nanmin(dv_dy):.2e}, max={np.nanmax(dv_dy):.2e}, has_nan={np.isnan(dv_dy).any()}, has_inf={np.isinf(dv_dy).any()}")
        print(f"  dw_dz: min={np.nanmin(dw_dz):.2e}, max={np.nanmax(dw_dz):.2e}, has_nan={np.isnan(dw_dz).any()}, has_inf={np.isinf(dw_dz).any()}")

    # Sum the partial derivatives to get the divergence (∇·u)
    divergence = du_dx + dv_dy + dw_dz

    # DEBUG: Check divergence BEFORE final clamp
    div_has_nan_before_clamp = np.isnan(divergence).any()
    div_has_inf_before_clamp = np.isinf(divergence).any()
    if div_has_nan_before_clamp or div_has_inf_before_clamp:
        print(f"❌ Warning: Invalid values in divergence field BEFORE final clamp. Stats: min={np.nanmin(divergence):.2e}, max={np.nanmax(divergence):.2e}, has_nan={div_has_nan_before_clamp}, has_inf={div_has_inf_before_clamp}")
    
    # Final check for NaNs/Infs in the computed divergence and clamp if necessary
    divergence = np.nan_to_num(divergence, nan=0.0, posinf=0.0, neginf=0.0)

    # DEBUG: Check divergence AFTER final clamp (always print for final state)
    print(f"[Divergence DEBUG] Final divergence stats AFTER clamp: min={np.nanmin(divergence):.2e}, max={np.nanmax(divergence):.2e}, has_nan={np.isnan(divergence).any()}, has_inf={np.isinf(divergence).any()}")

    return divergence


def compute_pressure_gradient(p_field, mesh_info):
    """
    Computes the gradient ∇p of a scalar pressure field using central differencing.
    This function calculates the gradient for the interior cells of the domain.

    Args:
        p_field (np.ndarray): Scalar pressure field with ghost cells, shape (nx+2, ny+2, nz+2).
                              Assumed to have correct boundary conditions applied to ghost cells.
        mesh_info (dict): Dictionary containing grid spacing: 'dx', 'dy', 'dz'.

    Returns:
        np.ndarray: Pressure gradient field of shape (nx, ny, nz, 3) (interior cells only).
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]

    # DEBUG: Check incoming p_field for NaNs/Infs
    p_field_has_nan = np.isnan(p_field).any()
    p_field_has_inf = np.isinf(p_field).any()
    if p_field_has_nan or p_field_has_inf:
        print(f"[Pressure Gradient DEBUG] Input p_field stats BEFORE clamp: min={np.nanmin(p_field):.2e}, max={np.nanmax(p_field):.2e}, has_nan={p_field_has_nan}, has_inf={p_field_has_inf}")

    # Defensively clamp any NaNs/Infs in the input pressure field
    p_field_clamped = np.nan_to_num(p_field, nan=0.0, posinf=0.0, neginf=0.0)

    # DEBUG: Check clamped p_field if original had issues
    if p_field_has_nan or p_field_has_inf:
        print(f"[Pressure Gradient DEBUG] p_field stats AFTER clamp: min={np.nanmin(p_field_clamped):.2e}, max={np.nanmax(p_field_clamped):.2e}")


    # Compute partial derivatives of pressure using central differences for the interior domain.
    # Similar to divergence, slicing implicitly selects the interior cells.
    grad_x = (p_field_clamped[2:, 1:-1, 1:-1] - p_field_clamped[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = (p_field_clamped[1:-1, 2:, 1:-1] - p_field_clamped[1:-1, :-2, 1:-1]) / (2 * dy)
    grad_z = (p_field_clamped[1:-1, 1:-1, 2:] - p_field_clamped[1:-1, 1:-1, :-2]) / (2 * dz)

    # DEBUG: Check individual gradient components before stacking
    grad_x_has_nan = np.isnan(grad_x).any()
    grad_y_has_nan = np.isnan(grad_y).any()
    grad_z_has_nan = np.isnan(grad_z).any()
    grad_x_has_inf = np.isinf(grad_x).any()
    grad_y_has_inf = np.isinf(grad_y).any()
    grad_z_has_inf = np.isinf(grad_z).any()

    if grad_x_has_nan or grad_y_has_nan or grad_z_has_nan or \
       grad_x_has_inf or grad_y_has_inf or grad_z_has_inf:
        print(f"[Pressure Gradient DEBUG] Individual gradient components stats (before stack):")
        print(f"  grad_x: min={np.nanmin(grad_x):.2e}, max={np.nanmax(grad_x):.2e}, has_nan={grad_x_has_nan}, has_inf={grad_x_has_inf}")
        print(f"  grad_y: min={np.nanmin(grad_y):.2e}, max={np.nanmax(grad_y):.2e}, has_nan={grad_y_has_nan}, has_inf={grad_y_has_inf}")
        print(f"  grad_z: min={np.nanmin(grad_z):.2e}, max={np.nanmax(grad_z):.2e}, has_nan={grad_z_has_nan}, has_inf={grad_z_has_inf}")


    # Stack the individual gradient components to form a vector field
    grad = np.stack([grad_x, grad_y, grad_z], axis=-1)

    # DEBUG: Check stacked gradient BEFORE final clamp
    grad_has_nan_before_clamp = np.isnan(grad).any()
    grad_has_inf_before_clamp = np.isinf(grad).any()
    if grad_has_nan_before_clamp or grad_has_inf_before_clamp:
        print(f"❌ Warning: Invalid values in pressure gradient BEFORE final clamp. Stats: min={np.nanmin(grad):.2e}, max={np.nanmax(grad):.2e}, has_nan={grad_has_nan_before_clamp}, has_inf={grad_has_inf_before_clamp}")

    # Final check for NaNs/Infs in the computed gradient and clamp if necessary
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    # DEBUG: Check stacked gradient AFTER final clamp (always print for final state)
    print(f"[Pressure Gradient DEBUG] Final gradient stats AFTER clamp: min={np.nanmin(grad):.2e}, max={np.nanmax(grad):.2e}, has_nan={np.isnan(grad).any()}, has_inf={np.isinf(grad).any()}")

    return grad



