# src/step_2_time_stepping_loop/mac_interpolation/vz.py
# ⬇️ VZ Interpolation — Convert cell-centered vz to face-centered values

from typing import Dict, Any
from .base import _get_velocity

debug = False  # toggle to True for verbose GitHub Action logs


def vz_k_plus_half(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
    """
    Interpolate vz at k+1/2 face (between central and above neighbor).
    """
    v_k = _get_velocity(cell_dict, k_cell, timestep, "vz")
    kp1 = cell_dict[str(k_cell)]["flat_index_k_plus_1"]
    if kp1 is None:
        return v_k
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
    km1 = cell_dict[str(k_cell)]["flat_index_k_minus_1"]
    if km1 is None:
        return v_k
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
    kp1 = cell_dict[str(k_cell)]["flat_index_k_plus_1"]
    if kp1 is None:
        return _get_velocity(cell_dict, k_cell, timestep, "vz")
    kp2 = cell_dict[str(kp1)]["flat_index_k_plus_1"]
    if kp2 is None:
        return _get_velocity(cell_dict, kp1, timestep, "vz")
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
    km1 = cell_dict[str(k_cell)]["flat_index_k_minus_1"]
    if km1 is None:
        return _get_velocity(cell_dict, k_cell, timestep, "vz")
    km2 = cell_dict[str(km1)]["flat_index_k_minus_1"]
    if km2 is None:
        return _get_velocity(cell_dict, km1, timestep, "vz")
    v_km1 = _get_velocity(cell_dict, km1, timestep, "vz")
    v_km2 = _get_velocity(cell_dict, km2, timestep, "vz")
    out = 0.5 * (v_km1 + v_km2)
    if debug:
        print(f"vz_k-3/2 between {km1} and {km2} -> {out}")
    return out



