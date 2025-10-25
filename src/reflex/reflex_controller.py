# src/reflex/reflex_controller.py
# 🔧 Reflex Controller — gathers diagnostics and applies reflex flags and metrics:
# damping, overflow detection, CFL monitoring, divergence tracking, pressure correction, mutation causality
# 📌 This module computes per-step reflex diagnostics and mutation traceability.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

from typing import List, Optional
from src.grid_modules.cell import Cell
from src.initialization.fluid_mask_initializer import build_simulation_grid
from src.config.config_validator import validate_config
from src.reflex.reflex_logic import adjust_time_step
from src.metrics.velocity_metrics import compute_max_velocity
from src.metrics.cfl_controller import compute_global_cfl
from src.metrics.divergence_metrics import compute_max_divergence
from src.metrics.projection_evaluator import calculate_projection_passes
from src.metrics.overflow_monitor import detect_overflow
from src.metrics.damping_manager import should_dampen as damping_metric
from src.metrics.reflex_score_evaluator import compute_score
from src.reflex.spatial_tagging.ghost_face_mapper import tag_ghost_adjacency
from src.reflex.spatial_tagging.suppression_zones import detect_suppression_zones, extract_mutated_coordinates
from src.utils.ghost_registry import build_ghost_registry, extract_ghost_coordinates

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def apply_reflex(
    grid: Optional[List[Cell]] = None,
    input_data: dict = {},
    step: int = 0,
    ghost_influence_count: Optional[int] = None,
    config: Optional[dict] = None,
    pressure_solver_invoked: Optional[bool] = None,
    pressure_mutated: Optional[bool] = None,
    post_projection_divergence: Optional[float] = None,
    ghost_registry: Optional[dict] = None
) -> dict:
    """
    Computes per-step diagnostics and reflex metrics for solver integrity and mutation traceability.
    Accepts external grid injection for test-safe diagnostics.
    """
    verbosity = (config or {}).get("reflex_verbosity", "medium")
    include_div_delta = (config or {}).get("include_divergence_delta", False)
    include_pressure_map = (config or {}).get("include_pressure_mutation_map", False)
    log_projection_trace = (config or {}).get("log_projection_trace", False)

    if debug and verbosity == "high":
        print(f"[REFLEX] Step {step} → Reflex diagnostics active")

    validate_config(config)
    if grid is None:
        grid = build_simulation_grid(config)

    domain = input_data["domain_definition"]
    time_step = input_data["simulation_parameters"]["time_step"]
    spacing = (
        (domain["max_x"] - domain["min_x"]) / domain["nx"],
        (domain["max_y"] - domain["min_y"]) / domain["ny"],
        (domain["max_z"] - domain["min_z"]) / domain["nz"]
    )

    max_velocity = compute_max_velocity(grid)
    max_divergence = compute_max_divergence(grid, domain)
    global_cfl = compute_global_cfl(grid, time_step, domain)
    overflow_detected = detect_overflow(grid)
    damping_enabled = damping_metric(grid, time_step)
    adjusted_time_step = adjust_time_step(grid, input_data)
    projection_passes = calculate_projection_passes(grid)

    divergence_zero = post_projection_divergence is not None and post_projection_divergence < 1e-8
    projection_skipped = projection_passes == 0

    influence_tagged = sum(
        1 for c in grid
        if getattr(c, "fluid_mask", False) and getattr(c, "influenced_by_ghost", False)
    )

    ghost_registry = ghost_registry or build_ghost_registry(grid)
    ghost_coords = extract_ghost_coordinates(ghost_registry)
    adjacency_zones = tag_ghost_adjacency(grid, ghost_coords, spacing)

    mutated_coords = extract_mutated_coordinates(input_data.get("mutated_cells", [])) if "mutated_cells" in input_data else set()
    suppression_zones = detect_suppression_zones(grid, ghost_coords, mutated_coords, spacing)

    fluid_count = sum(1 for c in grid if getattr(c, "fluid_mask", False))
    mutation_count = len(input_data.get("mutated_cells", [])) if "mutated_cells" in input_data else 0
    mutation_density = mutation_count / fluid_count if fluid_count > 0 else 0.0
    damping_triggered_count = sum(1 for c in grid if getattr(c, "damping_triggered", False))
    overflow_triggered_count = sum(1 for c in grid if getattr(c, "overflow_triggered", False))
    cfl_exceeded_count = sum(1 for c in grid if getattr(c, "cfl_exceeded", False))

    triggered_by = []
    if ghost_influence_count and ghost_influence_count > 0:
        triggered_by.append("ghost_influence")
    if overflow_detected:
        triggered_by.append("overflow_detected")
    if damping_enabled:
        triggered_by.append("damping_enabled")

    if debug and verbosity != "low":
        print(f"[REFLEX] Step {step}: Max velocity = {max_velocity:.3e}")
        print(f"[REFLEX] Step {step}: Max divergence = {max_divergence:.3e}")
        if post_projection_divergence is not None:
            print(f"[REFLEX] Step {step}: Post-projection divergence = {post_projection_divergence:.3e}")
        if log_projection_trace:
            print(f"[REFLEX] Step {step}: Projection passes = {projection_passes}")
        if projection_skipped:
            print(f"[REFLEX] Step {step}: Projection skipped (passes = 0)")
        if pressure_mutated:
            print(f"[REFLEX] Step {step}: Pressure field mutated.")
        elif pressure_solver_invoked:
            print(f"[REFLEX] Step {step}: Solver invoked but pressure unchanged.")

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
        "post_projection_divergence": post_projection_divergence,
        "adjacency_zones": adjacency_zones,
        "suppression_zones": suppression_zones,
        "mutation_density": round(mutation_density, 6),
        "mutated_cells": input_data.get("mutated_cells", []),
        "mutation_count": mutation_count,
        "damping_triggered_count": damping_triggered_count,
        "overflow_triggered_count": overflow_triggered_count,
        "cfl_exceeded_count": cfl_exceeded_count,
        "ghost_registry": ghost_registry
    }

    score_inputs = {
        "influence": reflex_data["ghost_influence_count"],
        "adjacency": len(adjacency_zones),
        "mutation": reflex_data["pressure_mutated"]
    }

    if debug:
        print(f"[REFLEX] Step {step} → reflex scoring inputs: {score_inputs}")
    score = compute_score(score_inputs)
    if debug:
        print(f"[REFLEX] Step {step} → computed reflex score: {score}")
    reflex_data["reflex_score"] = score if isinstance(score, (int, float)) else 0.0

    if debug:
        print(f"[REFLEX] reflex_data from reflex controller {reflex_data}")

    if (
        reflex_data["pressure_mutated"]
        and reflex_data["ghost_influence_count"] > 0
        and reflex_data["fluid_cells_modified_by_ghost"] == 0
    ):
        if debug:
            print(f"[SUPPRESSION] Step {step}: ghost influence present but no fluid cells tagged — check tagging logic")

    if debug and verbosity == "high" and include_div_delta:
        print(f"[REFLEX] Step {step} → Divergence delta tracking enabled")

    if debug and verbosity == "high" and include_pressure_map:
        print(f"[REFLEX] Step {step} → Pressure mutation map tracing enabled")

    return reflex_data



