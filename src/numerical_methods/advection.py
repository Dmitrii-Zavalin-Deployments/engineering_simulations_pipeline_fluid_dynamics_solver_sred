# src/numerical_methods/advection.py

import numpy as np

def compute_advection_term(u_field, velocity_field, mesh_info):
    """
    Computes the advection term - (u · ∇)u using a first-order upwind scheme.

    Args:
        u_field (np.ndarray): Scalar (nx, ny, nz) or vector (nx, ny, nz, 3) field to advect.
        velocity_field (np.ndarray): Velocity vector field of shape (nx, ny, nz, 3).
        mesh_info (dict): Dictionary with:
            - 'grid_shape': (nx, ny, nz)
            - 'dx', 'dy', 'dz': Grid spacing in each direction.

    Returns:
        np.ndarray: Advection term, same shape as u_field.
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    is_scalar = u_field.ndim == 3
    advection = np.zeros_like(u_field)

    vel_x = velocity_field[..., 0]
    vel_y = velocity_field[..., 1]
    vel_z = velocity_field[..., 2]

    def upwind_derivative(u, v, axis, spacing):
        deriv = np.zeros_like(u)
        pos_vel = v > 0  # No extra dimension added

        idx = [slice(None)] * u.ndim
        idx_p = idx.copy(); idx_n = idx.copy()
        idx_p[axis] = slice(1, -1)
        idx_n[axis] = slice(0, -2)
        forward = (u[tuple(idx_p)] - u[tuple(idx_n)]) / spacing

        idx_p2 = idx.copy(); idx_n2 = idx.copy()
        idx_p2[axis] = slice(2, None)
        idx_n2[axis] = slice(1, -1)
        backward = (u[tuple(idx_n2)] - u[tuple(idx_n)]) / spacing

        center = idx.copy(); center[axis] = slice(1, -1)
        deriv[tuple(center)] = np.where(
            pos_vel[tuple(center)],
            backward,
            forward
        )
        return deriv

    if is_scalar:
        advection += vel_x * upwind_derivative(u_field, vel_x, axis=0, spacing=dx)
        advection += vel_y * upwind_derivative(u_field, vel_y, axis=1, spacing=dy)
        advection += vel_z * upwind_derivative(u_field, vel_z, axis=2, spacing=dz)
    else:
        for comp in range(3):
            uc = u_field[..., comp]
            advection[..., comp] += vel_x * upwind_derivative(uc, vel_x, axis=0, spacing=dx)
            advection[..., comp] += vel_y * upwind_derivative(uc, vel_y, axis=1, spacing=dy)
            advection[..., comp] += vel_z * upwind_derivative(uc, vel_z, axis=2, spacing=dz)

    return -advection  # Negative for - (u · ∇)u

def advect_velocity(u, v, w, dx, dy, dz, dt):
    """
    Performs forward Euler advection of velocity components using a staggered MAC grid.

    Args:
        u, v, w (np.ndarray): Velocity components with ghost cells (shape: (nx+2, ny+2, nz+2))
        dx, dy, dz (float): Grid spacing
        dt (float): Time step

    Returns:
        tuple: Advected components (u*, v*, w*) on the interior domain
    """
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    velocity = np.stack([u, v, w], axis=-1)
    mesh_info = {
        "grid_shape": u.shape,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }

    u_star = u[interior] + dt * compute_advection_term(u, velocity, mesh_info)[interior]
    v_star = v[interior] + dt * compute_advection_term(v, velocity, mesh_info)[interior]
    w_star = w[interior] + dt * compute_advection_term(w, velocity, mesh_info)[interior]

    return u_star, v_star, w_star



