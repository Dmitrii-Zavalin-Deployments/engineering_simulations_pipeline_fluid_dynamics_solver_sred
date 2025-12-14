# src/step_2_time_stepping_loop/mac_interpolation/vx.py
# ➡️ VX Interpolation — Convert cell-centered vx to face-centered values
#
# Hardened against missing neighbor keys:
# - Uses .get() instead of direct indexing
# - Falls back to central cell velocity when neighbor is missing
# - This enforces a zero-gradient (Neumann) boundary condition

from typing import Dict, Any
from .base import _get_velocity

debug = False  # toggle to True for verbose GitHub Action logs


def vx_i_plus_half(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vx at i+1/2 face (between central and right neighbor).
    """
    v_i = _get_velocity(cell_dict, i_cell, timestep, "vx")
    ip1 = cell_dict[str(i_cell)].get("flat_index_i_plus_1")
    if ip1 is None:
        return v_i  # Neumann fallback
    v_ip1 = _get_velocity(cell_dict, ip1, timestep, "vx")
    out = 0.5 * (v_i + v_ip1)
    if debug:
        print(f"vx_i+1/2 between {i_cell} and {ip1} -> {out}")
    return out


def vx_i_minus_half(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vx at i-1/2 face (between central and left neighbor).
    """
    v_i = _get_velocity(cell_dict, i_cell, timestep, "vx")
    im1 = cell_dict[str(i_cell)].get("flat_index_i_minus_1")
    if im1 is None:
        return v_i  # Neumann fallback
    v_im1 = _get_velocity(cell_dict, im1, timestep, "vx")
    out = 0.5 * (v_i + v_im1)
    if debug:
        print(f"vx_i-1/2 between {im1} and {i_cell} -> {out}")
    return out


def vx_i_plus_three_half(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vx at i+3/2 face.
    Start from central -> right -> right+1.
    """
    ip1 = cell_dict[str(i_cell)].get("flat_index_i_plus_1")
    if ip1 is None:
        return _get_velocity(cell_dict, i_cell, timestep, "vx")  # Neumann fallback
    ip2 = cell_dict[str(ip1)].get("flat_index_i_plus_1")
    if ip2 is None:
        return _get_velocity(cell_dict, ip1, timestep, "vx")  # Neumann fallback
    v_ip1 = _get_velocity(cell_dict, ip1, timestep, "vx")
    v_ip2 = _get_velocity(cell_dict, ip2, timestep, "vx")
    out = 0.5 * (v_ip1 + v_ip2)
    if debug:
        print(f"vx_i+3/2 between {ip1} and {ip2} -> {out}")
    return out


def vx_i_minus_three_half(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vx at i-3/2 face.
    Start from central -> left -> left-1.
    """
    im1 = cell_dict[str(i_cell)].get("flat_index_i_minus_1")
    if im1 is None:
        return _get_velocity(cell_dict, i_cell, timestep, "vx")  # Neumann fallback
    im2 = cell_dict[str(im1)].get("flat_index_i_minus_1")
    if im2 is None:
        return _get_velocity(cell_dict, im1, timestep, "vx")  # Neumann fallback
    v_im1 = _get_velocity(cell_dict, im1, timestep, "vx")
    v_im2 = _get_velocity(cell_dict, im2, timestep, "vx")
    out = 0.5 * (v_im1 + v_im2)
    if debug:
        print(f"vx_i-3/2 between {im1} and {im2} -> {out}")
    return out

def vx_j_plus_one(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vx at (i+1/2, j+1, k).
    """
    v_i = _get_velocity(cell_dict, i_cell, timestep, "vx")
    jp1 = cell_dict[str(i_cell)].get("flat_index_j_plus_1")
    if jp1 is None:
        return v_i  # Neumann fallback
    v_jp1 = _get_velocity(cell_dict, jp1, timestep, "vx")
    out = 0.5 * (v_i + v_jp1)
    if debug:
        print(f"vx_j+1 between {i_cell} and {jp1} -> {out}")
    return out


def vx_j_minus_one(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vx at (i+1/2, j-1, k).
    """
    v_i = _get_velocity(cell_dict, i_cell, timestep, "vx")
    jm1 = cell_dict[str(i_cell)].get("flat_index_j_minus_1")
    if jm1 is None:
        return v_i  # Neumann fallback
    v_jm1 = _get_velocity(cell_dict, jm1, timestep, "vx")
    out = 0.5 * (v_i + v_jm1)
    if debug:
        print(f"vx_j-1 between {jm1} and {i_cell} -> {out}")
    return out


def vx_k_plus_one(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vx at (i+1/2, j, k+1).
    """
    v_i = _get_velocity(cell_dict, i_cell, timestep, "vx")
    kp1 = cell_dict[str(i_cell)].get("flat_index_k_plus_1")
    if kp1 is None:
        return v_i  # Neumann fallback
    v_kp1 = _get_velocity(cell_dict, kp1, timestep, "vx")
    out = 0.5 * (v_i + v_kp1)
    if debug:
        print(f"vx_k+1 between {i_cell} and {kp1} -> {out}")
    return out


def vx_k_minus_one(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vx at (i+1/2, j, k-1).
    """
    v_i = _get_velocity(cell_dict, i_cell, timestep, "vx")
    km1 = cell_dict[str(i_cell)].get("flat_index_k_minus_1")
    if km1 is None:
        return v_i  # Neumann fallback
    v_km1 = _get_velocity(cell_dict, km1, timestep, "vx")
    out = 0.5 * (v_i + v_km1)
    if debug:
        print(f"vx_k-1 between {km1} and {i_cell} -> {out}")
    return out


