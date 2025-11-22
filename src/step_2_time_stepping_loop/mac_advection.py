# src/step_2_time_stepping_loop/mac_advection.py
# 🌀 MAC Advection — compute nonlinear convective terms Adv(v_x), Adv(v_y), Adv(v_z) at face centers
# Adv(v) = u ∂v/∂x + v ∂v/∂y + w ∂v/∂z, evaluated at the corresponding MAC face

from typing import Dict, Any, Optional
from src.step_2_time_stepping_loop.mac_interpolation import (
    # x-faces (u)
    vx_i_plus_half,
    vx_i_minus_half,
    vx_i_plus_three_half,
    # y-faces (v)
    vy_j_plus_half,
    vy_j_minus_half,
    vy_j_plus_three_half,
    # z-faces (w)
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
    return d.get(key)


def _safe_sample_v_at_xface(cell_dict: Dict[str, Any], center: int, timestep: Optional[int]) -> float:
    """
    Collocate v (y-component, vy) to the x-face at i+1/2 by averaging vy on nearby y-faces.
    Strategy: average vy_j_plus_half at {center, j_plus, j_minus} if available.
    """
    samples = []
    # current cell's y-face value
    samples.append(vy_j_plus_half(cell_dict, center, timestep))
    # j+ neighbor
    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    if j_plus is not None:
        samples.append(vy_j_plus_half(cell_dict, j_plus, timestep))
    # j- neighbor
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    if j_minus is not None:
        samples.append(vy_j_plus_half(cell_dict, j_minus, timestep))

    out = sum(samples) / len(samples)
    return out


def _safe_sample_w_at_xface(cell_dict: Dict[str, Any], center: int, timestep: Optional[int]) -> float:
    """
    Collocate w (z-component, vz) to the x-face at i+1/2 by averaging vz on nearby z-faces.
    Strategy: average vz_k_plus_half at {center, k_plus, k_minus} if available.
    """
    samples = []
    samples.append(vz_k_plus_half(cell_dict, center, timestep))
    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    if k_plus is not None:
        samples.append(vz_k_plus_half(cell_dict, k_plus, timestep))
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    if k_minus is not None:
        samples.append(vz_k_plus_half(cell_dict, k_minus, timestep))

    out = sum(samples) / len(samples)
    return out


def _safe_sample_u_at_yface(cell_dict: Dict[str, Any], center: int, timestep: Optional[int]) -> float:
    """
    Collocate u (x-component, vx) to the y-face at j+1/2 by averaging vx on nearby x-faces.
    Strategy: average vx_i_plus_half at {center, i_plus, i_minus} if available.
    """
    samples = []
    samples.append(vx_i_plus_half(cell_dict, center, timestep))
    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    if i_plus is not None:
        samples.append(vx_i_plus_half(cell_dict, i_plus, timestep))
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    if i_minus is not None:
        samples.append(vx_i_plus_half(cell_dict, i_minus, timestep))

    out = sum(samples) / len(samples)
    return out


def _safe_sample_w_at_yface(cell_dict: Dict[str, Any], center: int, timestep: Optional[int]) -> float:
    """
    Collocate w (z-component, vz) to the y-face at j+1/2 by averaging vz on nearby z-faces.
    Strategy: average vz_k_plus_half at {center, k_plus, k_minus} if available.
    """
    samples = []
    samples.append(vz_k_plus_half(cell_dict, center, timestep))
    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    if k_plus is not None:
        samples.append(vz_k_plus_half(cell_dict, k_plus, timestep))
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    if k_minus is not None:
        samples.append(vz_k_plus_half(cell_dict, k_minus, timestep))

    out = sum(samples) / len(samples)
    return out


def _safe_sample_u_at_zface(cell_dict: Dict[str, Any], center: int, timestep: Optional[int]) -> float:
    """
    Collocate u (x-component, vx) to the z-face at k+1/2 by averaging vx on nearby x-faces.
    Strategy: average vx_i_plus_half at {center, i_plus, i_minus} if available.
    """
    samples = []
    samples.append(vx_i_plus_half(cell_dict, center, timestep))
    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    if i_plus is not None:
        samples.append(vx_i_plus_half(cell_dict, i_plus, timestep))
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    if i_minus is not None:
        samples.append(vx_i_plus_half(cell_dict, i_minus, timestep))

    out = sum(samples) / len(samples)
    return out


def _safe_sample_v_at_zface(cell_dict: Dict[str, Any], center: int, timestep: Optional[int]) -> float:
    """
    Collocate v (y-component, vy) to the z-face at k+1/2 by averaging vy on nearby y-faces.
    Strategy: average vy_j_plus_half at {center, j_plus, j_minus} if available.
    """
    samples = []
    samples.append(vy_j_plus_half(cell_dict, center, timestep))
    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    if j_plus is not None:
        samples.append(vy_j_plus_half(cell_dict, j_plus, timestep))
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    if j_minus is not None:
        samples.append(vy_j_plus_half(cell_dict, j_minus, timestep))

    out = sum(samples) / len(samples)
    return out


# ---------------- Face-centered gradients of target component ----------------

def _grad_vx_at_xface(cell_dict: Dict[str, Any], center: int, dx: float, dy: float, dz: float, timestep: Optional[int]) -> Dict[str, float]:
    """
    Gradients of v_x evaluated at the x-face (i+1/2).
    - ∂vx/∂x: centered using vx at i+3/2 and i-1/2.
    - ∂vx/∂y: difference of vx at y-adjacent faces (approx by calling vx_i_plus_half at j± neighbors).
    - ∂vx/∂z: difference of vx at z-adjacent faces (approx by calling vx_i_plus_half at k± neighbors).
    """
    dvx_dx = (vx_i_plus_three_half(cell_dict, center, timestep) - vx_i_minus_half(cell_dict, center, timestep)) / (2.0 * dx)

    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    vx_j_plus = vx_i_plus_half(cell_dict, j_plus, timestep) if j_plus is not None else vx_i_plus_half(cell_dict, center, timestep)
    vx_j_minus = vx_i_plus_half(cell_dict, j_minus, timestep) if j_minus is not None else vx_i_plus_half(cell_dict, center, timestep)
    dvx_dy = (vx_j_plus - vx_j_minus) / (2.0 * dy)

    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    vx_k_plus = vx_i_plus_half(cell_dict, k_plus, timestep) if k_plus is not None else vx_i_plus_half(cell_dict, center, timestep)
    vx_k_minus = vx_i_plus_half(cell_dict, k_minus, timestep) if k_minus is not None else vx_i_plus_half(cell_dict, center, timestep)
    dvx_dz = (vx_k_plus - vx_k_minus) / (2.0 * dz)

    if debug:
        print(f"[grad vx @ x-face {center}] dvx_dx={dvx_dx}, dvx_dy={dvx_dy}, dvx_dz={dvx_dz}")
    return {"dx": dvx_dx, "dy": dvx_dy, "dz": dvx_dz}


def _grad_vy_at_yface(cell_dict: Dict[str, Any], center: int, dx: float, dy: float, dz: float, timestep: Optional[int]) -> Dict[str, float]:
    """
    Gradients of v_y evaluated at the y-face (j+1/2).
    - ∂vy/∂y: centered using vy at j+3/2 and j-1/2.
    - ∂vy/∂x: difference of vy at x-adjacent faces (approx by calling vy_j_plus_half at i± neighbors).
    - ∂vy/∂z: difference of vy at z-adjacent faces (approx by calling vy_j_plus_half at k± neighbors).
    """
    dvy_dy = (vy_j_plus_three_half(cell_dict, center, timestep) - vy_j_minus_half(cell_dict, center, timestep)) / (2.0 * dy)

    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    vy_i_plus = vy_j_plus_half(cell_dict, i_plus, timestep) if i_plus is not None else vy_j_plus_half(cell_dict, center, timestep)
    vy_i_minus = vy_j_plus_half(cell_dict, i_minus, timestep) if i_minus is not None else vy_j_plus_half(cell_dict, center, timestep)
    dvy_dx = (vy_i_plus - vy_i_minus) / (2.0 * dx)

    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    vy_k_plus = vy_j_plus_half(cell_dict, k_plus, timestep) if k_plus is not None else vy_j_plus_half(cell_dict, center, timestep)
    vy_k_minus = vy_j_plus_half(cell_dict, k_minus, timestep) if k_minus is not None else vy_j_plus_half(cell_dict, center, timestep)
    dvy_dz = (vy_k_plus - vy_k_minus) / (2.0 * dz)

    if debug:
        print(f"[grad vy @ y-face {center}] dvy_dx={dvy_dx}, dvy_dy={dvy_dy}, dvy_dz={dvy_dz}")
    return {"dx": dvy_dx, "dy": dvy_dy, "dz": dvy_dz}


def _grad_vz_at_zface(cell_dict: Dict[str, Any], center: int, dx: float, dy: float, dz: float, timestep: Optional[int]) -> Dict[str, float]:
    """
    Gradients of v_z evaluated at the z-face (k+1/2).
    - ∂vz/∂z: centered using vz at k+3/2 and k-1/2.
    - ∂vz/∂x: difference of vz at x-adjacent faces (approx by calling vz_k_plus_half at i± neighbors).
    - ∂vz/∂y: difference of vz at y-adjacent faces (approx by calling vz_k_plus_half at j± neighbors).
    """
    dvz_dz = (vz_k_plus_three_half(cell_dict, center, timestep) - vz_k_minus_half(cell_dict, center, timestep)) / (2.0 * dz)

    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    vz_i_plus = vz_k_plus_half(cell_dict, i_plus, timestep) if i_plus is not None else vz_k_plus_half(cell_dict, center, timestep)
    vz_i_minus = vz_k_plus_half(cell_dict, i_minus, timestep) if i_minus is not None else vz_k_plus_half(cell_dict, center, timestep)
    dvz_dx = (vz_i_plus - vz_i_minus) / (2.0 * dx)

    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    vz_j_plus = vz_k_plus_half(cell_dict, j_plus, timestep) if j_plus is not None else vz_k_plus_half(cell_dict, center, timestep)
    vz_j_minus = vz_k_plus_half(cell_dict, j_minus, timestep) if j_minus is not None else vz_k_plus_half(cell_dict, center, timestep)
    dvz_dy = (vz_j_plus - vz_j_minus) / (2.0 * dy)

    if debug:
        print(f"[grad vz @ z-face {center}] dvz_dx={dvz_dx}, dvz_dy={dvz_dy}, dvz_dz={dvz_dz}")
    return {"dx": dvz_dx, "dy": dvz_dy, "dz": dvz_dz}


# ---------------- Advection operators ----------------

def adv_vx(cell_dict: Dict[str, Any], center: int, dx: float, dy: float, dz: float, timestep: Optional[int] = None) -> float:
    """
    Adv(v_x) at the x-face (i+1/2):
      Adv(v_x) = u_face * ∂(v_x)/∂x + v_face * ∂(v_x)/∂y + w_face * ∂(v_x)/∂z
    """
    # face-centered velocities at i+1/2
    u_face = vx_i_plus_half(cell_dict, center, timestep)
    v_face = _safe_sample_v_at_xface(cell_dict, center, timestep)
    w_face = _safe_sample_w_at_xface(cell_dict, center, timestep)

    grads = _grad_vx_at_xface(cell_dict, center, dx, dy, dz, timestep)
    out = u_face * grads["dx"] + v_face * grads["dy"] + w_face * grads["dz"]

    if debug:
        print(f"[adv vx @ x-face {center}] u={u_face}, v={v_face}, w={w_face}, "
              f"dvx_dx={grads['dx']}, dvx_dy={grads['dy']}, dvx_dz={grads['dz']}, Adv={out}")
    return out


def adv_vy(cell_dict: Dict[str, Any], center: int, dx: float, dy: float, dz: float, timestep: Optional[int] = None) -> float:
    """
    Adv(v_y) at the y-face (j+1/2):
      Adv(v_y) = u_face * ∂(v_y)/∂x + v_face * ∂(v_y)/∂y + w_face * ∂(v_y)/∂z
    """
    # face-centered velocities at j+1/2
    v_face = vy_j_plus_half(cell_dict, center, timestep)
    u_face = _safe_sample_u_at_yface(cell_dict, center, timestep)
    w_face = _safe_sample_w_at_yface(cell_dict, center, timestep)

    grads = _grad_vy_at_yface(cell_dict, center, dx, dy, dz, timestep)
    out = u_face * grads["dx"] + v_face * grads["dy"] + w_face * grads["dz"]

    if debug:
        print(f"[adv vy @ y-face {center}] u={u_face}, v={v_face}, w={w_face}, "
              f"dvy_dx={grads['dx']}, dvy_dy={grads['dy']}, dvy_dz={grads['dz']}, Adv={out}")
    return out


def adv_vz(cell_dict: Dict[str, Any], center: int, dx: float, dy: float, dz: float, timestep: Optional[int] = None) -> float:
    """
    Adv(v_z) at the z-face (k+1/2):
      Adv(v_z) = u_face * ∂(v_z)/∂x + v_face * ∂(v_z)/∂y + w_face * ∂(v_z)/∂z
    """
    # face-centered velocities at k+1/2
    w_face = vz_k_plus_half(cell_dict, center, timestep)
    u_face = _safe_sample_u_at_zface(cell_dict, center, timestep)
    v_face = _safe_sample_v_at_zface(cell_dict, center, timestep)

    grads = _grad_vz_at_zface(cell_dict, center, dx, dy, dz, timestep)
    out = u_face * grads["dx"] + v_face * grads["dy"] + w_face * grads["dz"]

    if debug:
        print(f"[adv vz @ z-face {center}] u={u_face}, v={v_face}, w={w_face}, "
              f"dvz_dx={grads['dx']}, dvz_dy={grads['dy']}, dvz_dz={grads['dz']}, Adv={out}")
    return out



