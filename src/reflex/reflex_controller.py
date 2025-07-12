# src/reflex/reflex_controller.py
# 🔧 Reflex controller — gathers diagnostics and applies reflex flags like damping, overflow detection, and CFL monitoring

from typing import List
from src.grid_modules.cell import Cell
from src.reflex.reflex_logic import should_dampen, should_flag_overflow, adjust_time_step
from src.metrics.velocity_metrics import compute_max_velocity
from src.metrics.cfl_controller import compute_global_cfl

def apply_reflex(grid: List[Cell], input_data: dict, step: int) -> dict:
    """
    Applies reflex logic based on velocity diagnostics, CFL condition, and configurable thresholds.

    Args:
        grid (List[Cell]): Simulation grid at current step
        input_data (dict): Full simulation configuration and parameters
        step (int): Simulation step index

    Returns:
        dict: Reflex flags and metrics including:
            - "damping_enabled" (bool)
            - "overflow_detected" (bool)
            - "adjusted_time_step" (float)
            - "max_velocity" (float)
            - "global_cfl" (float)
    """
    # 📊 Extract domain and time_step
    domain = input_data["domain_definition"]
    time_step = input_data["simulation_parameters"]["time_step"]

    # 📊 Collect key metrics
    max_velocity = compute_max_velocity(grid)
    global_cfl = compute_global_cfl(grid, time_step, domain)

    # 🧠 Evaluate reflex conditions
    damping_enabled = should_dampen(grid)
    overflow_detected = should_flag_overflow(grid)
    adjusted_time_step = adjust_time_step(grid, input_data)

    # 📦 Package diagnostics for step controller
    reflex_flags = {
        "damping_enabled": damping_enabled,
        "overflow_detected": overflow_detected,
        "adjusted_time_step": adjusted_time_step,
        "max_velocity": max_velocity,
        "global_cfl": global_cfl
    }

    return reflex_flags



