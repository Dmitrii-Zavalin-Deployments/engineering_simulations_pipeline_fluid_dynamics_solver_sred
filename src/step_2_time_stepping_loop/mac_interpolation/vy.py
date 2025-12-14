# src/step_2_time_stepping_loop/mac_interpolation/vy.py
# ⬆️ VY Interpolation — Convert cell-centered vy to face-centered values
#
# Hardened against missing neighbor keys:
# - Uses .get() instead of direct indexing
# - Falls back to central cell velocity when neighbor is missing
# - This enforces a zero-gradient (Neumann) boundary condition

from typing import Dict, Any
from .base import _get_velocity

debug = False  # toggle to True for verbose GitHub Action logs


def vy_j_plus_half(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vy at j+1/2 face (between central and up neighbor).
    """
    v_j = _get_velocity(cell_dict, j_cell, timestep, "vy")
    jp1 = cell_dict[str(j_cell)].get("flat_index_j_plus_1")
    if jp1 is None:
        return v_j  # Neumann fallback
    v_jp1 = _get_velocity(cell_dict, jp1, timestep, "vy")
    out = 0.5 * (v_j + v_jp1)
    if debug:
        print(f"vy_j+1/2 between {j_cell} and {jp1} -> {out}")
    return out


def vy_j_minus_half(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vy at j-1/2 face (between central and down neighbor).
    """
    v_j = _get_velocity(cell_dict, j_cell, timestep, "vy")
    jm1 = cell_dict[str(j_cell)].get("flat_index_j_minus_1")
    if jm1 is None:
        return v_j  # Neumann fallback
    v_jm1 = _get_velocity(cell_dict, jm1, timestep, "vy")
    out = 0.5 * (v_j + v_jm1)
    if debug:
        print(f"vy_j-1/2 between {jm1} and {j_cell} -> {out}")
    return out


def vy_j_plus_three_half(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vy at j+3/2 face.
    Start from central -> up -> up+1.
    """
    jp1 = cell_dict[str(j_cell)].get("flat_index_j_plus_1")
    if jp1 is None:
        return _get_velocity(cell_dict, j_cell, timestep, "vy")  # Neumann fallback
    jp2 = cell_dict[str(jp1)].get("flat_index_j_plus_1")
    if jp2 is None:
        return _get_velocity(cell_dict, jp1, timestep, "vy")  # Neumann fallback
    v_jp1 = _get_velocity(cell_dict, jp1, timestep, "vy")
    v_jp2 = _get_velocity(cell_dict, jp2, timestep, "vy")
    out = 0.5 * (v_jp1 + v_jp2)
    if debug:
        print(f"vy_j+3/2 between {jp1} and {jp2} -> {out}")
    return out


def vy_j_minus_three_half(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vy at j-3/2 face.
    Start from central -> down -> down-1.
    """
    jm1 = cell_dict[str(j_cell)].get("flat_index_j_minus_1")
    if jm1 is None:
        return _get_velocity(cell_dict, j_cell, timestep, "vy")  # Neumann fallback
    jm2 = cell_dict[str(jm1)].get("flat_index_j_minus_1")
    if jm2 is None:
        return _get_velocity(cell_dict, jm1, timestep, "vy")  # Neumann fallback
    v_jm1 = _get_velocity(cell_dict, jm1, timestep, "vy")
    v_jm2 = _get_velocity(cell_dict, jm2, timestep, "vy")
    out = 0.5 * (v_jm1 + v_jm2)
    if debug:
        print(f"vy_j-3/2 between {jm1} and {jm2} -> {out}")
    return out

def vy_i_plus_one(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vy at (i+1, j+1/2, k).
    """
    v_j = _get_velocity(cell_dict, j_cell, timestep, "vy")
    ip1 = cell_dict[str(j_cell)].get("flat_index_i_plus_1")
    if ip1 is None:
        return v_j  # Neumann fallback
    v_ip1 = _get_velocity(cell_dict, ip1, timestep, "vy")
    out = 0.5 * (v_j + v_ip1)
    if debug:
        print(f"vy_i+1 between {j_cell} and {ip1} -> {out}")
    return out


def vy_i_minus_one(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vy at (i-1, j+1/2, k).
    """
    v_j = _get_velocity(cell_dict, j_cell, timestep, "vy")
    im1 = cell_dict[str(j_cell)].get("flat_index_i_minus_1")
    if im1 is None:
        return v_j  # Neumann fallback
    v_im1 = _get_velocity(cell_dict, im1, timestep, "vy")
    out = 0.5 * (v_j + v_im1)
    if debug:
        print(f"vy_i-1 between {im1} and {j_cell} -> {out}")
    return out


def vy_k_plus_one(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vy at (i, j+1/2, k+1).
    """
    v_j = _get_velocity(cell_dict, j_cell, timestep, "vy")
    kp1 = cell_dict[str(j_cell)].get("flat_index_k_plus_1")
    if kp1 is None:
        return v_j  # Neumann fallback
    v_kp1 = _get_velocity(cell_dict, kp1, timestep, "vy")
    out = 0.5 * (v_j + v_kp1)
    if debug:
        print(f"vy_k+1 between {j_cell} and {kp1} -> {out}")
    return out


def vy_k_minus_one(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vy at (i, j+1/2, k-1).
    """
    v_j = _get_velocity(cell_dict, j_cell, timestep, "vy")
    km1 = cell_dict[str(j_cell)].get("flat_index_k_minus_1")
    if km1 is None:
        return v_j  # Neumann fallback
    v_km1 = _get_velocity(cell_dict, km1, timestep, "vy")
    out = 0.5 * (v_j + v_km1)
    if debug:
        print(f"vy_k-1 between {km1} and {j_cell} -> {out}")
    return out



