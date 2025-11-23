# src/step_2_time_stepping_loop/mac_advection_ops.py
# 🌀 MAC Advection — operator assembly for nonlinear convective terms
# Adv(v) = u ∂v/∂x + v ∂v/∂y + w ∂v/∂z, evaluated at the corresponding MAC face
#
# Boundary Fallback Theory in CFD:
# Same rationale as in mac_advection_gradients.py — when neighbors are missing,
# reuse the central cell’s velocity to enforce a zero-gradient (Neumann) condition.

from src.step_2_time_stepping_loop.mac_interpolation.vx import vx_i_plus_half
from src.step_2_time_stepping_loop.mac_interpolation.vy import vy_j_plus_half
from src.step_2_time_stepping_loop.mac_interpolation.vz import vz_k_plus_half

# Import gradient helpers
from src.step_2_time_stepping_loop.mac_advection_gradients import (
    _grad_vx_at_xface,
    _grad_vy_at_yface,
    _grad_vz_at_zface,
)


def adv_vx(cell_dict, center, dx, dy, dz, timestep=None):
    """Adv(v_x) at the x-face (i+1/2)."""
    u_face = vx_i_plus_half(cell_dict, center, timestep)
    # collocate v and w by sampling at the same face location
    v_face = vy_j_plus_half(cell_dict, center, timestep)
    w_face = vz_k_plus_half(cell_dict, center, timestep)

    grads = _grad_vx_at_xface(cell_dict, center, dx, dy, dz, timestep)
    return u_face * grads["dx"] + v_face * grads["dy"] + w_face * grads["dz"]


def adv_vy(cell_dict, center, dx, dy, dz, timestep=None):
    """Adv(v_y) at the y-face (j+1/2)."""
    v_face = vy_j_plus_half(cell_dict, center, timestep)
    u_face = vx_i_plus_half(cell_dict, center, timestep)
    w_face = vz_k_plus_half(cell_dict, center, timestep)

    grads = _grad_vy_at_yface(cell_dict, center, dx, dy, dz, timestep)
    return u_face * grads["dx"] + v_face * grads["dy"] + w_face * grads["dz"]


def adv_vz(cell_dict, center, dx, dy, dz, timestep=None):
    """Adv(v_z) at the z-face (k+1/2)."""
    w_face = vz_k_plus_half(cell_dict, center, timestep)
    u_face = vx_i_plus_half(cell_dict, center, timestep)
    v_face = vy_j_plus_half(cell_dict, center, timestep)

    grads = _grad_vz_at_zface(cell_dict, center, dx, dy, dz, timestep)
    return u_face * grads["dx"] + v_face * grads["dy"] + w_face * grads["dz"]



