# src/step_2_time_stepping_loop/driver_loop.py
# 🚀 Driver Loop Skeleton — Navier–Stokes timestep orchestration
#
# Phases:
#   1. Velocity Prediction (v*)
#   2. Pressure Correction (p^{n+1})
#   3. Velocity Correction (v^{n+1})
#
# Currently: Phase 1 implemented, Phases 2 & 3 are placeholders.

from typing import Dict, Any
from src.step_2_time_stepping_loop.mac_update_velocity import (
    update_velocity_x,
    update_velocity_y,
    update_velocity_z,
)
from src.step_2_time_stepping_loop.boundary_utils import enforce_boundary

debug = False


def timestep_driver(cell_dict: Dict[str, Any], config: Dict[str, Any], timestep: int) -> None:
    """
    Orchestrate one full timestep of the solver.
    Currently implements Phase 1 (velocity prediction).
    Phases 2 and 3 are placeholders.
    """
    next_timestep = timestep + 1

    # ---------------- Phase 1: Velocity Prediction ----------------
    for flat_idx_str, cell in cell_dict.items():
        flat_idx = int(flat_idx_str)

        vx_star = update_velocity_x(cell_dict, flat_idx, config, timestep)
        vy_star = update_velocity_y(cell_dict, flat_idx, config, timestep)
        vz_star = update_velocity_z(cell_dict, flat_idx, config, timestep)

        prev_state = cell["time_history"].get(str(timestep))
        if prev_state is None:
            raise ValueError(f"No time_history for timestep {timestep} in cell {flat_idx}")

        # Stage predictor velocities (not final!)
        new_state = {
            "pressure": prev_state["pressure"],  # pressure unchanged until Phase 2
            "velocity": {"vx": vx_star, "vy": vy_star, "vz": vz_star},
        }

        # Enforce boundary overrides
        new_state = enforce_boundary(new_state, cell, config)

        # Store provisional state under a staging key
        cell["time_history"][f"{next_timestep}_predictor"] = new_state

        if debug and flat_idx < 5:
            print(f"[Phase 1] cell={flat_idx}, v*={new_state['velocity']}")

    # ---------------- Phase 2: Pressure Correction ----------------
    # TODO: assemble RHS from divergence of v*, solve Poisson equation for p^{n+1}
    # cell["time_history"][f"{next_timestep}_pressure"] = {...}

    # ---------------- Phase 3: Velocity Correction ----------------
    # TODO: subtract ∇p^{n+1} from v*, produce divergence-free v^{n+1}
    # cell["time_history"][str(next_timestep)] = {...}
    # apply enforce_boundary here for the second time

    if debug:
        print(f"✅ Phase 1 complete for timestep {timestep} → {next_timestep}")



