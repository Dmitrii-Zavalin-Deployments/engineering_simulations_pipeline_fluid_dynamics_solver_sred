# src/step_2_time_stepping_loop/mac_advection.py
# 🌀 MAC Advection — compute nonlinear convective terms Adv(v_x), Adv(v_y), Adv(v_z) at face centers
# Adv(v) = u ∂v/∂x + v ∂v/∂y + w ∂v/∂z, evaluated at the corresponding MAC face

from typing import Dict, Any, Optional

# Import face interpolation functions
from src.step_2_time_stepping_loop.mac_interpolation.vx import (
    vx_i_plus_half,
    vx_i_minus_half,
    vx_i_plus_three_half,
)

from src.step_2_time_stepping_loop.mac_interpolation.vy import (
    vy_j_plus_half,
    vy_j_minus_half,
    vy_j_plus_three_half,
)

from src.step_2_time_stepping_loop.mac_interpolation.vz import (
    vz_k_plus_half,
    vz_k_minus_half,
    vz_k_plus_three_half,
)

debug = False  # toggle for verbose logging


# ---------------- Utilities ----------------

def _neighbor_index(cell_dict: Dict[str, Any], center: int, key: str) -> Optional[int]:
    """Return neighbor flat_index via key (e.g., 'flat_index_j_plus_1') or None if missing."""
    d = cell_dict.get(str(center))
    if not d:
        return None
    # Hardened: safe get to avoid KeyError when neighbor keys are absent in mocks/boundaries
    return d.get(key, None)


# ---------------- Gradient helpers ----------------

def _grad_vx_at_xface(cell_dict, center, dx, dy, dz, timestep=None):
    """Gradients of v_x at the x-face (i+1/2)."""
    dvx_dx = (vx_i_plus_three_half(cell_dict, center, timestep) -
              vx_i_minus_half(cell_dict, center, timestep)) / (2.0 * dx)

    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    vx_j_plus = vx_i_plus_half(cell_dict, j_plus, timestep) if j_plus else vx_i_plus_half(cell_dict, center, timestep)
    vx_j_minus = vx_i_plus_half(cell_dict, j_minus, timestep) if j_minus else vx_i_plus_half(cell_dict, center, timestep)
    dvx_dy = (vx_j_plus - vx_j_minus) / (2.0 * dy)

    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    vx_k_plus = vx_i_plus_half(cell_dict, k_plus, timestep) if k_plus else vx_i_plus_half(cell_dict, center, timestep)
    vx_k_minus = vx_i_plus_half(cell_dict, k_minus, timestep) if k_minus else vx_i_plus_half(cell_dict, center, timestep)
    dvx_dz = (vx_k_plus - vx_k_minus) / (2.0 * dz)

    return {"dx": dvx_dx, "dy": dvx_dy, "dz": dvx_dz}


def _grad_vy_at_yface(cell_dict, center, dx, dy, dz, timestep=None):
    """Gradients of v_y at the y-face (j+1/2)."""
    dvy_dy = (vy_j_plus_three_half(cell_dict, center, timestep) -
              vy_j_minus_half(cell_dict, center, timestep)) / (2.0 * dy)

    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    vy_i_plus = vy_j_plus_half(cell_dict, i_plus, timestep) if i_plus else vy_j_plus_half(cell_dict, center, timestep)
    vy_i_minus = vy_j_plus_half(cell_dict, i_minus, timestep) if i_minus else vy_j_plus_half(cell_dict, center, timestep)
    dvy_dx = (vy_i_plus - vy_i_minus) / (2.0 * dx)

    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    vy_k_plus = vy_j_plus_half(cell_dict, k_plus, timestep) if k_plus else vy_j_plus_half(cell_dict, center, timestep)
    vy_k_minus = vy_j_plus_half(cell_dict, k_minus, timestep) if k_minus else vy_j_plus_half(cell_dict, center, timestep)
    dvy_dz = (vy_k_plus - vy_k_minus) / (2.0 * dz)

    return {"dx": dvy_dx, "dy": dvy_dy, "dz": dvy_dz}


def _grad_vz_at_zface(cell_dict, center, dx, dy, dz, timestep=None):
    """Gradients of v_z at the z-face (k+1/2)."""
    dvz_dz = (vz_k_plus_three_half(cell_dict, center, timestep) -
              vz_k_minus_half(cell_dict, center, timestep)) / (2.0 * dz)

    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    vz_i_plus = vz_k_plus_half(cell_dict, i_plus, timestep) if i_plus else vz_k_plus_half(cell_dict, center, timestep)
    vz_i_minus = vz_k_plus_half(cell_dict, i_minus, timestep) if i_minus else vz_k_plus_half(cell_dict, center, timestep)
    dvz_dx = (vz_i_plus - vz_i_minus) / (2.0 * dx)

    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    vz_j_plus = vz_k_plus_half(cell_dict, j_plus, timestep) if j_plus else vz_k_plus_half(cell_dict, center, timestep)
    vz_j_minus = vz_k_plus_half(cell_dict, j_minus, timestep) if j_minus else vz_k_plus_half(cell_dict, center, timestep)
    dvz_dy = (vz_j_plus - vz_j_minus) / (2.0 * dy)

    return {"dx": dvz_dx, "dy": dvz_dy, "dz": dvz_dz}


# ---------------- Advection operators ----------------

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



