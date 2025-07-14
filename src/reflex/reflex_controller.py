# src/reflex/reflex_controller.py
# 🔧 Reflex controller — gathers diagnostics and applies reflex flags and metrics:
# damping, overflow detection, CFL monitoring, divergence, and projection estimates

from typing import List, Optional
from src.grid_modules.cell import Cell
from src.reflex.reflex_logic import should_flag_overflow, adjust_time_step
from src.metrics.velocity_metrics import compute_max_velocity
from src.metrics.cfl_controller import compute_global_cfl
from src.metrics.divergence_metrics import compute_max_divergence
from src.metrics.projection_evaluator import calculate_projection_passes
from src.metrics.overflow_monitor import detect_overflow
from src.metrics.damping_manager import should_dampen as damping_metric

def apply_reflex(
    grid: List[Cell],
    input_data: dict,
    step: int,
    ghost_influence_count: Optional[int] = None,
    config: Optional[dict] = None
) -> dict:
    """
    Applies reflex diagnostics including velocity, divergence, CFL, overflow,
    damping logic, time-step adaptation, pressure projection estimation,
    and ghost influence propagation tracking.

    Args:
        grid (List[Cell]): Simulation grid for current time step
        input_data (dict): Simulation configuration and physical parameters
        step (int): Current simulation step index
        ghost_influence_count (int, optional): Fluid cells modified via ghost influence
        config (dict, optional): Reflex verbosity and diagnostic toggles

    Returns:
        dict: Reflex metadata containing stability flags, physics metrics, and ghost interaction stats
    """
    verbosity = (config or {}).get("reflex_verbosity", "medium")
    include_div_delta = (config or {}).get("include_divergence_delta", False)
    include_pressure_map = (config or {}).get("include_pressure_mutation_map", False)
    log_projection_trace = (config or {}).get("log_projection_trace", False)

    if verbosity == "high":
        print(f"[DEBUG] Step {step} → Reflex diagnostics active")

    domain = input_data["domain_definition"]
    time_step = input_data["simulation_parameters"]["time_step"]

    max_velocity = compute_max_velocity(grid)
    max_divergence = compute_max_divergence(grid, domain)
    global_cfl = compute_global_cfl(grid, time_step, domain)
    overflow_detected = detect_overflow(grid)
    damping_enabled = damping_metric(grid, time_step)
    adjusted_time_step = adjust_time_step(grid, input_data)
    projection_passes = calculate_projection_passes(grid)

    divergence_zero = max_divergence < 1e-8
    projection_skipped = divergence_zero or projection_passes == 0

    if verbosity != "low":
        if divergence_zero:
            print(f"⚠️ [reflex] Step {step}: Zero divergence — projection may be skipped.")
        else:
            print(f"📊 [reflex] Step {step}: Max divergence = {max_divergence:.6e}")

        if log_projection_trace:
            print(f"🔄 [reflex] Step {step}: Projection passes estimated → {projection_passes}")

    influence_tagged = sum(
        1 for c in grid
        if getattr(c, "fluid_mask", False) and getattr(c, "influenced_by_ghost", False)
    )

    # Optional mutation causality tagging
    triggered_by = []
    if ghost_influence_count and ghost_influence_count > 0:
        triggered_by.append("ghost_influence")
    if overflow_detected:
        triggered_by.append("overflow_detected")
    if damping_enabled:
        triggered_by.append("damping_enabled")
    # "boundary_override" can optionally be added externally by main_solver or step_controller

    reflex_data = {
        "max_velocity": max_velocity,
        "max_divergence": max_divergence,
        "global_cfl": global_cfl,
        "overflow_detected": overflow_detected,
        "damping_enabled": damping_enabled,
        "adjusted_time_step": adjusted_time_step,
        "projection_passes": projection_passes,
        "divergence_zero": divergence_zero,
        "projection_skipped": projection_skipped,
        "ghost_influence_count": ghost_influence_count if ghost_influence_count is not None else 0,
        "fluid_cells_modified_by_ghost": influence_tagged,
        "triggered_by": triggered_by  # ✅ pressure mutation causality tags
    }

    if verbosity == "high" and include_div_delta:
        print(f"[DEBUG] Step {step} → Divergence delta tracking enabled")
        # placeholder for divergence delta logic

    if verbosity == "high" and include_pressure_map:
        print(f"[DEBUG] Step {step} → Pressure mutation map tracing enabled")
        # placeholder for pressure diff logic

    return reflex_data



