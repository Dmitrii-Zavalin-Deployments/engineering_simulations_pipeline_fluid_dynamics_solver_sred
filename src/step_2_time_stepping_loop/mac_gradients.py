# src/step_2_time_stepping_loop/mac_gradients.py
# 🧮 Step 2: MAC Gradients — Compute ∇ operators using face-centered values

from typing import Dict, Any
from src.step_2_time_stepping_loop.mac_interpolation import (
    vx_i_plus_half,
    vx_i_minus_half,
    vy_j_plus_half,
    vy_j_minus_half,
    vz_k_plus_half,
    vz_k_minus_half,
)

debug = False  # toggle for verbose logging


def _resolve_pressure(cell_dict: Dict[str, Any], flat_index: int, timestep: int | None) -> float:
    """
    Helper to fetch pressure from a cell at a given timestep.
    Defaults to latest timestep if None is provided.
    """
    history_keys = list(cell_dict[str(flat_index)]["time_history"].keys())
    if not history_keys:
        raise ValueError(f"No time_history available for cell {flat_index}")
    if timestep is None:
        timestep = max(map(int, history_keys))
    state = cell_dict[str(flat_index)]["time_history"].get(str(timestep))
    if state is None:
        raise ValueError(f"No time_history for timestep {timestep} in cell {flat_index}")
    return float(state["pressure"])


# ---------------- Pressure Gradients ----------------

def grad_p_x(cell_dict: Dict[str, Any], i_cell: int, dx: float, timestep: int | None = None) -> float:
    """
    Compute ∂p/∂x at the face between i and i+1 (i+1/2).
    If no neighbor exists, assume ghost cell pressure = current cell pressure.
    """
    p_i = _resolve_pressure(cell_dict, i_cell, timestep)
    ip1 = cell_dict[str(i_cell)].get("flat_index_i_plus_1")
    if ip1 is None:
        p_ip1 = p_i  # ghost cell assumption
    else:
        p_ip1 = _resolve_pressure(cell_dict, ip1, timestep)
    out = (p_ip1 - p_i) / dx
    if debug:
        print(f"∂p/∂x at i+1/2 between {i_cell} and {ip1} -> {out}")
    return out


def grad_p_y(cell_dict: Dict[str, Any], j_cell: int, dy: float, timestep: int | None = None) -> float:
    """
    Compute ∂p/∂y at the face between j and j+1 (j+1/2).
    If no neighbor exists, assume ghost cell pressure = current cell pressure.
    """
    p_j = _resolve_pressure(cell_dict, j_cell, timestep)
    jp1 = cell_dict[str(j_cell)].get("flat_index_j_plus_1")
    if jp1 is None:
        p_jp1 = p_j
    else:
        p_jp1 = _resolve_pressure(cell_dict, jp1, timestep)
    out = (p_jp1 - p_j) / dy
    if debug:
        print(f"∂p/∂y at j+1/2 between {j_cell} and {jp1} -> {out}")
    return out


def grad_p_z(cell_dict: Dict[str, Any], k_cell: int, dz: float, timestep: int | None = None) -> float:
    """
    Compute ∂p/∂z at the face between k and k+1 (k+1/2).
    If no neighbor exists, assume ghost cell pressure = current cell pressure.
    """
    p_k = _resolve_pressure(cell_dict, k_cell, timestep)
    kp1 = cell_dict[str(k_cell)].get("flat_index_k_plus_1")
    if kp1 is None:
        p_kp1 = p_k
    else:
        p_kp1 = _resolve_pressure(cell_dict, kp1, timestep)
    out = (p_kp1 - p_k) / dz
    if debug:
        print(f"∂p/∂z at k+1/2 between {k_cell} and {kp1} -> {out}")
    return out


# ---------------- Divergence ----------------

def divergence(cell_dict: Dict[str, Any], center: int, dx: float, dy: float, dz: float, timestep: int | None = None) -> float:
    """
    Compute ∇·v at the central cell using MAC face velocities.
    """
    # x-direction
    vx_plus = vx_i_plus_half(cell_dict, center, timestep)
    vx_minus = vx_i_minus_half(cell_dict, center, timestep)
    dvx_dx = (vx_plus - vx_minus) / dx

    # y-direction
    vy_plus = vy_j_plus_half(cell_dict, center, timestep)
    vy_minus = vy_j_minus_half(cell_dict, center, timestep)
    dvy_dy = (vy_plus - vy_minus) / dy

    # z-direction
    vz_plus = vz_k_plus_half(cell_dict, center, timestep)
    vz_minus = vz_k_minus_half(cell_dict, center, timestep)
    dvz_dz = (vz_plus - vz_minus) / dz

    out = dvx_dx + dvy_dy + dvz_dz
    if debug:
        print(f"∇·v at cell {center} -> {out}")
    return out



