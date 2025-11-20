# src/step_2_time_stepping_loop/mac_interpolation/vx.py
# ➡️ VX Interpolation — Convert cell-centered vx to face-centered values

from typing import Dict, Any
from .base import _get_velocity

debug = False  # toggle to True for verbose GitHub Action logs


def vx_i_plus_half(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vx at i+1/2 face (between central and right neighbor).
    """
    v_i = _get_velocity(cell_dict, i_cell, timestep, "vx")
    ip1 = cell_dict[str(i_cell)]["flat_index_i_plus_1"]
    if ip1 is None:
        return v_i
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
    im1 = cell_dict[str(i_cell)]["flat_index_i_minus_1"]
    if im1 is None:
        return v_i
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
    ip1 = cell_dict[str(i_cell)]["flat_index_i_plus_1"]
    if ip1 is None:
        return _get_velocity(cell_dict, i_cell, timestep, "vx")
    ip2 = cell_dict[str(ip1)]["flat_index_i_plus_1"]
    if ip2 is None:
        return _get_velocity(cell_dict, ip1, timestep, "vx")
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
    im1 = cell_dict[str(i_cell)]["flat_index_i_minus_1"]
    if im1 is None:
        return _get_velocity(cell_dict, i_cell, timestep, "vx")
    im2 = cell_dict[str(im1)]["flat_index_i_minus_1"]
    if im2 is None:
        return _get_velocity(cell_dict, im1, timestep, "vx")
    v_im1 = _get_velocity(cell_dict, im1, timestep, "vx")
    v_im2 = _get_velocity(cell_dict, im2, timestep, "vx")
    out = 0.5 * (v_im1 + v_im2)
    if debug:
        print(f"vx_i-3/2 between {im1} and {im2} -> {out}")
    return out



