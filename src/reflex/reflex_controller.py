# src/reflex/reflex_controller.py
# 🔧 Reflex controller — gathers diagnostics and applies reflex flags and metrics:
# damping, overflow detection, CFL monitoring, divergence tracking, pressure correction, mutation causality

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
    config: Optional[dict] = None,
    pressure_solver_invoked: Optional[bool] = None,
    pressure_mutated: Optional[bool] = None,
    post_projection_divergence: Optional[float] = None
) -> dict:
    """
    Applies reflex diagnostics including velocity, divergence, CFL, overflow,
    damping logic, time-step adaptation, projection estimation, and mutation cause tracking.

    Args:
        grid (List[Cell]): Simulation grid
        input_data (dict): Full simulation config
        step (int): Current simulation step index
        ghost_influence_count (Optional[int]): Fluid cells modified by ghosts
        config (Optional[dict]): Reflex diagnostic flags
        pressure_solver_invoked (Optional[bool]): True if pressure projection solver ran
        pressure_mutated (Optional[bool]): True if pressure field was updated
        post_projection_divergence (Optional[float]): Divergence after projection

    Returns:
        dict: Flattened reflex metadata fields for snapshot
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
    projection_skipped = projection_passes == 0  # ✅ Corrected logic

    influence_tagged = sum(
        1 for c in grid
        if getattr(c, "fluid_mask", False) and getattr(c, "influenced_by_ghost", False)
    )

    # Mutation causality tagging
    triggered_by = []
    if ghost_influence_count and ghost_influence_count > 0:
        triggered_by.append("ghost_influence")
    if overflow_detected:
        triggered_by.append("overflow_detected")
    if damping_enabled:
        triggered_by.append("damping_enabled")
    # "boundary_override" may be appended externally

    if verbosity != "low":
        print(f"📊 [reflex] Step {step}: Max velocity = {max_velocity:.3e}")
        print(f"📊 [reflex] Step {step}: Max divergence = {max_divergence:.3e}")
        if post_projection_divergence is not None:
            print(f"📊 [reflex] Step {step}: Post-projection divergence = {post_projection_divergence:.3e}")
        if log_projection_trace:
            print(f"🔄 [reflex] Step {step}: Projection passes = {projection_passes}")
        if projection_skipped:
            print(f"⚠️ [reflex] Step {step}: Projection skipped (passes = 0)")
        if pressure_mutated:
            print(f"✅ [reflex] Step {step}: Pressure field mutated.")
        elif pressure_solver_invoked:
            print(f"ℹ️ [reflex] Step {step}: Solver invoked but pressure unchanged.")

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
        "triggered_by": triggered_by,
        "pressure_solver_invoked": pressure_solver_invoked if pressure_solver_invoked is not None else False,
        "pressure_mutated": pressure_mutated if pressure_mutated is not None else False,
        "post_projection_divergence": post_projection_divergence if post_projection_divergence is not None else None
    }

    if verbosity == "high" and include_div_delta:
        print(f"[DEBUG] Step {step} → Divergence delta tracking enabled")

    if verbosity == "high" and include_pressure_map:
        print(f"[DEBUG] Step {step} → Pressure mutation map tracing enabled")

    return reflex_data



