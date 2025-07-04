# src/numerical_methods/advection.py

import numpy as np

def compute_advection_term(
    scalar_field: np.ndarray,
    velocity_field: np.ndarray,
    mesh_info: dict,
    should_log_verbose: bool = False # Added verbose logging flag
) -> np.ndarray:
    """
    Computes the advection term (u ⋅ ∇)φ for a scalar field φ (which can be a velocity component)
    using a first-order upwind scheme.

    Args:
        scalar_field (np.ndarray): The scalar field (e.g., u, v, or w component of velocity)
                                   on the full grid including ghost cells (nx+2, ny+2, nz+2).
        velocity_field (np.ndarray): The full 3D velocity field (u, v, w components)
                                     on the full grid including ghost cells (nx+2, ny+2, nz+2, 3).
        mesh_info (dict): Dictionary containing grid spacing: 'dx', 'dy', 'dz'.
        should_log_verbose (bool): If True, print detailed debug logs.

    Returns:
        np.ndarray: The computed advection term (u ⋅ ∇)φ for the interior cells
                    (shape: nx, ny, nz).
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Extract velocity components
    u_vel = velocity_field[..., 0]
    v_vel = velocity_field[..., 1]
    w_vel = velocity_field[..., 2]

    # Initialize advection term for interior cells
    advection_term = np.zeros_like(scalar_field[1:-1, 1:-1, 1:-1])

    # Compute derivatives using upwind scheme
    # For du/dx (advection in x-direction)
    # If u_vel > 0, use backward difference (i - i-1)
    # If u_vel < 0, use forward difference (i+1 - i)
    # This applies to the scalar_field, driven by the velocity components

    # U * d(scalar_field)/dx
    advection_x = np.zeros_like(advection_term)
    u_vel_interior = u_vel[1:-1, 1:-1, 1:-1]
    # Positive u_vel: use backward difference for scalar_field
    advection_x[u_vel_interior >= 0] = u_vel_interior[u_vel_interior >= 0] * \
                                      (scalar_field[1:-1, 1:-1, 1:-1][u_vel_interior >= 0] -
                                       scalar_field[:-2, 1:-1, 1:-1][u_vel_interior >= 0]) / dx
    # Negative u_vel: use forward difference for scalar_field
    advection_x[u_vel_interior < 0] = u_vel_interior[u_vel_interior < 0] * \
                                     (scalar_field[2:, 1:-1, 1:-1][u_vel_interior < 0] -
                                      scalar_field[1:-1, 1:-1, 1:-1][u_vel_interior < 0]) / dx

    # V * d(scalar_field)/dy
    advection_y = np.zeros_like(advection_term)
    v_vel_interior = v_vel[1:-1, 1:-1, 1:-1]
    # Positive v_vel: use backward difference for scalar_field
    advection_y[v_vel_interior >= 0] = v_vel_interior[v_vel_interior >= 0] * \
                                      (scalar_field[1:-1, 1:-1, 1:-1][v_vel_interior >= 0] -
                                       scalar_field[1:-1, :-2, 1:-1][v_vel_interior >= 0]) / dy
    # Negative v_vel: use forward difference for scalar_field
    advection_y[v_vel_interior < 0] = v_vel_interior[v_vel_interior < 0] * \
                                     (scalar_field[1:-1, 2:, 1:-1][v_vel_interior < 0] -
                                      scalar_field[1:-1, 1:-1, 1:-1][v_vel_interior < 0]) / dy

    # W * d(scalar_field)/dz
    advection_z = np.zeros_like(advection_term)
    w_vel_interior = w_vel[1:-1, 1:-1, 1:-1]
    # Positive w_vel: use backward difference for scalar_field
    advection_z[w_vel_interior >= 0] = w_vel_interior[w_vel_interior >= 0] * \
                                      (scalar_field[1:-1, 1:-1, 1:-1][w_vel_interior >= 0] -
                                       scalar_field[1:-1, 1:-1, :-2][w_vel_interior >= 0]) / dz
    # Negative w_vel: use forward difference for scalar_field
    advection_z[w_vel_interior < 0] = w_vel_interior[w_vel_interior < 0] * \
                                     (scalar_field[1:-1, 1:-1, 2:][w_vel_interior < 0] -
                                      scalar_field[1:-1, 1:-1, 1:-1][w_vel_interior < 0]) / dz

    advection_term = advection_x + advection_y + advection_z

    if np.isnan(advection_term).any() or np.isinf(advection_term).any():
        print("❌ Warning: NaN or Inf encountered in advection term. Clamping to zero.")
        advection_term = np.nan_to_num(advection_term, nan=0.0, posinf=0.0, neginf=0.0)

    if should_log_verbose:
        # These are debug prints, only show if verbose logging is on
        print(f"    - Advection Term Max Abs: {np.max(np.abs(advection_term)):.4e}")
        print(f"    - Advection Term Mean Abs: {np.mean(np.abs(advection_term)):.4e}")

    return advection_term



