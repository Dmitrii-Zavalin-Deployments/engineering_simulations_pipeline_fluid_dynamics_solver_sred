# src/step_2_time_stepping_loop/mac_interpolation.py
# 🧱 Step 2: MAC Interpolation — Convert cell-centered velocities to face-centered values

from typing import Dict, Any

debug = False  # toggle to True for verbose GitHub Action logs


def _resolve_timestep(cell_dict: Dict[str, Any], flat_index: int, timestep: int | None) -> int:
    """
    Resolve timestep: if None, pick the latest available in time_history.
    """
    history_keys = list(cell_dict[str(flat_index)]["time_history"].keys())
    if not history_keys:
        raise ValueError(f"No time_history available for cell {flat_index}")
    if timestep is None:
        # default to latest
        timestep = max(map(int, history_keys))
        if debug:
            print(f"ℹ️ Using latest timestep {timestep} for cell {flat_index}")
    return timestep


def _get_velocity(cell_dict: Dict[str, Any], flat_index: int, timestep: int | None, comp: str) -> float:
    """
    Helper to fetch velocity component from a cell at a given timestep.
    """
    timestep = _resolve_timestep(cell_dict, flat_index, timestep)
    state = cell_dict[str(flat_index)]["time_history"].get(str(timestep))
    if state is None:
        raise ValueError(f"No time_history for timestep {timestep} in cell {flat_index}")
    return float(state["velocity"][comp])


# ---------------- VX Interpolation ----------------

def vx_i_plus_half(cell_dict: Dict[str, Any], i_cell: int, timestep: int | None = None) -> float:
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


# ---------------- VY Interpolation ----------------

def vy_j_plus_half(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    v_j = _get_velocity(cell_dict, j_cell, timestep, "vy")
    jp1 = cell_dict[str(j_cell)]["flat_index_j_plus_1"]
    if jp1 is None:
        return v_j
    v_jp1 = _get_velocity(cell_dict, jp1, timestep, "vy")
    out = 0.5 * (v_j + v_jp1)
    if debug:
        print(f"vy_j+1/2 between {j_cell} and {jp1} -> {out}")
    return out


def vy_j_minus_half(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    v_j = _get_velocity(cell_dict, j_cell, timestep, "vy")
    jm1 = cell_dict[str(j_cell)]["flat_index_j_minus_1"]
    if jm1 is None:
        return v_j
    v_jm1 = _get_velocity(cell_dict, jm1, timestep, "vy")
    out = 0.5 * (v_j + v_jm1)
    if debug:
        print(f"vy_j-1/2 between {jm1} and {j_cell} -> {out}")
    return out


def vy_j_plus_three_half(cell_dict: Dict[str, Any], j_cell: int, timestep: int | None = None) -> float:
    jp1 = cell_dict[str(j_cell)]["flat_index_j_plus_1"]
    if jp1 is None:
        return _get_velocity(cell_dict, j_cell, timestep, "vy")
    jp2 = cell_dict[str(jp1)]["flat_index_j_plus_1"]
    if jp2 is None:
        return _get_velocity(cell_dict, jp1, timestep, "vy")
    v_jp1 = _get_velocity(cell_dict, jp1, timestep, "vy")
    v_jp2 = _get_velocity(cell_dict, jp2, timestep, "vy")
    out = 0.5 * (v_jp1 + v_jp2)
    if debug:
        print(f"vy_j+3/2 between {jp1} and {jp2} -> {out}")
    return out


# ---------------- VZ Interpolation ----------------

def vz_k_plus_half(cell_dict: Dict[str, Any], k_cell: int, timestep: int | None = None) -> float:
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



