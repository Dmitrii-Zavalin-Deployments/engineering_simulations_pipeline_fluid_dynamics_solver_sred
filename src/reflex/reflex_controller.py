# src/reflex/reflex_controller.py
# 🔧 Reflex controller — gathers diagnostics and applies reflex flags and metrics:
# damping, overflow detection, CFL monitoring, divergence, and projection estimates

from typing import List
from src.grid_modules.cell import Cell
from src.reflex.reflex_logic import should_flag_overflow, adjust_time_step
from src.metrics.velocity_metrics import compute_max_velocity
from src.metrics.cfl_controller import compute_global_cfl
from src.metrics.divergence_metrics import compute_max_divergence
from src.metrics.projection_evaluator import calculate_projection_passes
from src.metrics.overflow_monitor import detect_overflow
from src.metrics.damping_manager import should_dampen as damping_metric

def apply_reflex(grid: List[Cell], input_data: dict, step: int) -> dict:
    """
    Applies reflex diagnostics including velocity, divergence, CFL, overflow,
    damping logic, time-step adaptation, and pressure projection pass estimation.

    Args:
        grid (List[Cell]): Simulation grid for current time step
        input_data (dict): Simulation configuration and physical parameters
        step (int): Current simulation step index

    Returns:
        dict: Reflex metadata containing stability flags and physics metrics:
            - "max_velocity" (float)
            - "max_divergence" (float)
            - "global_cfl" (float)
            - "overflow_detected" (bool)
            - "damping_enabled" (bool)
            - "adjusted_time_step" (float)
            - "projection_passes" (int)
    """
    domain = input_data["domain_definition"]
    time_step = input_data["simulation_parameters"]["time_step"]

    max_velocity = compute_max_velocity(grid)
    max_divergence = compute_max_divergence(grid, domain)
    global_cfl = compute_global_cfl(grid, time_step, domain)
    overflow_detected = detect_overflow(grid)
    damping_enabled = damping_metric(grid, time_step)
    adjusted_time_step = adjust_time_step(grid, input_data)
    projection_passes = calculate_projection_passes(grid)

    return {
        "max_velocity": max_velocity,
        "max_divergence": max_divergence,
        "global_cfl": global_cfl,
        "overflow_detected": overflow_detected,
        "damping_enabled": damping_enabled,
        "adjusted_time_step": adjusted_time_step,
        "projection_passes": projection_passes
    }



