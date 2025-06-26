# src/numerical_methods/advection.py

import numpy as np

def compute_advection_term(u_field, velocity_field, mesh_info):
    """
    Computes the advection term - (u · ∇)u using a first-order upwind scheme.

    Args:
        u_field (np.ndarray): Scalar or vector field being advected (shape: (nx, ny, nz) or (nx, ny, nz, 3)).
        velocity_field (np.ndarray): Vector field with shape (nx, ny, nz, 3).
        mesh_info (dict): Contains:
            - 'grid_shape': tuple (nx, ny, nz)
            - 'dx', 'dy', 'dz': Grid spacing

    Returns:
        np.ndarray: Advection term with the same shape as u_field.
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']
    advection = np.zeros_like(u_field)
    is_scalar = (u_field.ndim == 3)

    vel_x, vel_y, vel_z = velocity_field[..., 0], velocity_field[..., 1], velocity_field[..., 2]

    def compute_flux_component(u, v, axis):
        pos = np.maximum(0, v)
        neg = np.minimum(0, v)

        if not is_scalar:
            pos = pos[..., np.newaxis]
            neg = neg[..., np.newaxis]

        F_plus = u * pos
        F_minus = u * neg

        return F_plus, F_minus

    # X-direction
    Fp, Fn = compute_flux_component(u_field, vel_x, axis=0)
    Fx_p = np.zeros_like(u_field)
    Fx_n = np.zeros_like(u_field)
    Fx_p[:-1] = Fp[:-1] + Fn[1:]
    Fx_p[-1] = Fp[-1]
    Fx_n[1:] = Fp[1:] + Fn[:-1]
    Fx_n[0] = Fn[0]
    advection += (Fx_p - Fx_n) / dx

    # Y-direction
    Fp, Fn = compute_flux_component(u_field, vel_y, axis=1)
    Fy_p = np.zeros_like(u_field)
    Fy_n = np.zeros_like(u_field)
    Fy_p[:, :-1] = Fp[:, :-1] + Fn[:, 1:]
    Fy_p[:, -1] = Fp[:, -1]
    Fy_n[:, 1:] = Fp[:, 1:] + Fn[:, :-1]
    Fy_n[:, 0] = Fn[:, 0]
    advection += (Fy_p - Fy_n) / dy

    # Z-direction
    Fp, Fn = compute_flux_component(u_field, vel_z, axis=2)
    Fz_p = np.zeros_like(u_field)
    Fz_n = np.zeros_like(u_field)
    Fz_p[:, :, :-1] = Fp[:, :, :-1] + Fn[:, :, 1:]
    Fz_p[:, :, -1] = Fp[:, :, -1]
    Fz_n[:, :, 1:] = Fp[:, :, 1:] + Fn[:, :, :-1]
    Fz_n[:, :, 0] = Fn[:, :, 0]
    advection += (Fz_p - Fz_n) / dz

    return advection

def advect_velocity(u, v, w, dx, dy, dz, dt):
    """
    Computes the velocity field after one forward Euler advection step.

    Args:
        u, v, w: Velocity components with ghost cells included (shape: (nx+2, ny+2, nz+2))
        dx, dy, dz: Grid spacing
        dt: Time step

    Returns:
        u*, v*, w*: Advected velocity fields on the interior domain
    """
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))

    velocity = np.stack([u, v, w], axis=-1)
    mesh = {
        'grid_shape': u.shape,
        'dx': dx,
        'dy': dy,
        'dz': dz,
    }

    u_star = u[interior] - dt * compute_advection_term(u, velocity, mesh)[interior]
    v_star = v[interior] - dt * compute_advection_term(v, velocity, mesh)[interior]
    w_star = w[interior] - dt * compute_advection_term(w, velocity, mesh)[interior]

    return u_star, v_star, w_star



