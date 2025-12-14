# src/step_2_time_stepping_loop/mac_interpolation/vz.py
# ⬇️ VZ Interpolation — Convert cell-centered vz to face-centered values
#
# Hardened against missing neighbor keys:
# - Uses .get() instead of direct indexing
# - Falls back to central cell velocity when neighbor is missing
# - This enforces a zero-gradient (Neumann) boundary condition

from typing import Dict, Any
from .base import _get_velocity

debug = False  # toggle to True for verbose GitHub Action logs


def vz_k_plus_half(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vz at k+1/2 face (between central and above neighbor).
    """
    v_k = _get_velocity(cell_dict, k_cell, timestep, "vz")
    kp1 = cell_dict[str(k_cell)].get("flat_index_k_plus_1")
    if kp1 is None:
        return v_k  # Neumann fallback
    v_kp1 = _get_velocity(cell_dict, kp1, timestep, "vz")
    out = 0.5 * (v_k + v_kp1)
    if debug:
        print(f"vz_k+1/2 between {k_cell} and {kp1} -> {out}")
    return out


def vz_k_minus_half(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vz at k-1/2 face (between central and below neighbor).
    """
    v_k = _get_velocity(cell_dict, k_cell, timestep, "vz")
    km1 = cell_dict[str(k_cell)].get("flat_index_k_minus_1")
    if km1 is None:
        return v_k  # Neumann fallback
    v_km1 = _get_velocity(cell_dict, km1, timestep, "vz")
    out = 0.5 * (v_k + v_km1)
    if debug:
        print(f"vz_k-1/2 between {km1} and {k_cell} -> {out}")
    return out


def vz_k_plus_three_half(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vz at k+3/2 face.
    Start from central -> above -> above+1.
    """
    kp1 = cell_dict[str(k_cell)].get("flat_index_k_plus_1")
    if kp1 is None:
        return _get_velocity(cell_dict, k_cell, timestep, "vz")  # Neumann fallback
    kp2 = cell_dict[str(kp1)].get("flat_index_k_plus_1")
    if kp2 is None:
        return _get_velocity(cell_dict, kp1, timestep, "vz")  # Neumann fallback
    v_kp1 = _get_velocity(cell_dict, kp1, timestep, "vz")
    v_kp2 = _get_velocity(cell_dict, kp2, timestep, "vz")
    out = 0.5 * (v_kp1 + v_kp2)
    if debug:
        print(f"vz_k+3/2 between {kp1} and {kp2} -> {out}")
    return out


def vz_k_minus_three_half(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vz at k-3/2 face.
    Start from central -> below -> below-1.
    """
    km1 = cell_dict[str(k_cell)].get("flat_index_k_minus_1")
    if km1 is None:
        return _get_velocity(cell_dict, k_cell, timestep, "vz")  # Neumann fallback
    km2 = cell_dict[str(km1)].get("flat_index_k_minus_1")
    if km2 is None:
        return _get_velocity(cell_dict, km1, timestep, "vz")  # Neumann fallback
    v_km1 = _get_velocity(cell_dict, km1, timestep, "vz")
    v_km2 = _get_velocity(cell_dict, km2, timestep, "vz")
    out = 0.5 * (v_km1 + v_km2)
    if debug:
        print(f"vz_k-3/2 between {km1} and {km2} -> {out}")
    return out

def vz_i_plus_one(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vz at (i+1, j, k+1/2).
    """
    v_k = _get_velocity(cell_dict, k_cell, timestep, "vz")
    ip1 = cell_dict[str(k_cell)].get("flat_index_i_plus_1")
    if ip1 is None:
        return v_k  # Neumann fallback
    v_ip1 = _get_velocity(cell_dict, ip1, timestep, "vz")
    out = 0.5 * (v_k + v_ip1)
    if debug:
        print(f"vz_i+1 between {k_cell} and {ip1} -> {out}")
    return out


def vz_i_minus_one(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vz at (i-1, j, k+1/2).
    """
    v_k = _get_velocity(cell_dict, k_cell, timestep, "vz")
    im1 = cell_dict[str(k_cell)].get("flat_index_i_minus_1")
    if im1 is None:
        return v_k  # Neumann fallback
    v_im1 = _get_velocity(cell_dict, im1, timestep, "vz")
    out = 0.5 * (v_k + v_im1)
    if debug:
        print(f"vz_i-1 between {im1} and {k_cell} -> {out}")
    return out


def vz_j_plus_one(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vz at (i, j+1, k+1/2).
    """
    v_k = _get_velocity(cell_dict, k_cell, timestep, "vz")
    jp1 = cell_dict[str(k_cell)].get("flat_index_j_plus_1")
    if jp1 is None:
        return v_k  # Neumann fallback
    v_jp1 = _get_velocity(cell_dict, jp1, timestep, "vz")
    out = 0.5 * (v_k + v_jp1)
    if debug:
        print(f"vz_j+1 between {k_cell} and {jp1} -> {out}")
    return out


def vz_j_minus_one(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vz at (i, j-1, k+1/2).
    """
    v_k = _get_velocity(cell_dict, k_cell, timestep, "vz")
    jm1 = cell_dict[str(k_cell)].get("flat_index_j_minus_1")
    if jm1 is None:
        return v_k  # Neumann fallback
    v_jm1 = _get_velocity(cell_dict, jm1, timestep, "vz")
    out = 0.5 * (v_k + v_jm1)
    if debug:
        print(f"vz_j-1 between {jm1} and {k_cell} -> {out}")
    return out



