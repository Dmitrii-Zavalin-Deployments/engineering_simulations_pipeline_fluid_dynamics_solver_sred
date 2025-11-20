# src/step_2_time_stepping_loop/mac_interpolation/base.py
# ⚙️ Shared helpers for MAC Interpolation modules

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
    Helper to fetch velocity component (vx, vy, vz) from a cell at a given timestep.
    """
    timestep = _resolve_timestep(cell_dict, flat_index, timestep)
    state = cell_dict[str(flat_index)]["time_history"].get(str(timestep))
    if state is None:
        raise ValueError(f"No time_history for timestep {timestep} in cell {flat_index}")
    value = float(state["velocity"][comp])
    if debug:
        print(f"🔎 _get_velocity: cell={flat_index}, timestep={timestep}, comp={comp}, value={value}")
    return value



