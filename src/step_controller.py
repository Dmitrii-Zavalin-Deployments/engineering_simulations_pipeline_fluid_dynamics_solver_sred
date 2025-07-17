# src/step_controller.py
# 🚀 Simulation Step Controller — orchestrates velocity, pressure, ghost logic, and reflex updates

import logging
from typing import List, Tuple, Optional
from src.grid_modules.cell import Cell
from src.physics.ghost_cell_generator import generate_ghost_cells
from src.physics.boundary_condition_solver import apply_boundary_conditions
from src.physics.ghost_influence_applier import apply_ghost_influence
from src.solvers.momentum_solver import apply_momentum_update
from src.solvers.pressure_solver import apply_pressure_correction
from src.physics.velocity_projection import apply_pressure_velocity_projection
from src.reflex.reflex_controller import apply_reflex
from src.utils.ghost_diagnostics import log_ghost_summary, inject_diagnostics
from src.utils.divergence_tracker import compute_divergence_stats
from src.adaptive.timestep_controller import suggest_timestep  # ✅ NEW IMPORT

def evolve_step(
    grid: List[Cell],
    input_data: dict,
    step: int,
    config: Optional[dict] = None,
    reflex_score: Optional[int] = None  # ✅ Optional score input
) -> Tuple[List[Cell], dict]:
    """
    Evolves the fluid grid by one simulation step using:
    - Ghost cell padding and boundary enforcement
    - Ghost influence propagation (velocity/pressure transfer)
    - Momentum update (advection + viscosity)
    - Pressure correction (divergence and projection)
    - Velocity projection via pressure gradient
    - Reflex logic (damping, overflow, CFL diagnostics)

    Args:
        grid (List[Cell]): Current grid state
        input_data (dict): Full input configuration
        step (int): Current simulation step index
        config (dict, optional): Reflex and diagnostic config values

    Returns:
        Tuple[List[Cell], dict]: Updated grid and reflex metadata for snapshot
    """
    logging.info(f"🌀 [evolve_step] Step {step}: Starting evolution")

    domain = input_data["domain_definition"]
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    spacing = (dx, dy, dz)

    output_folder = "data/testing-input-output/navier_stokes_output"
    padded_grid, ghost_registry = generate_ghost_cells(grid, input_data)
    logging.debug(f"🧱 Generated {len(ghost_registry)} ghost cells")
    log_ghost_summary(ghost_registry)

    boundary_tagged_grid = apply_boundary_conditions(padded_grid, ghost_registry, input_data)
    boundary_applied = True

    influence_count = apply_ghost_influence(
        boundary_tagged_grid,
        spacing,
        verbose=(config or {}).get("reflex_verbosity", "") == "high",
        radius=(config or {}).get("ghost_adjacency_depth", 1)
    )
    logging.debug(f"👣 Ghost influence applied to {influence_count} fluid cells")

    stats_before = compute_divergence_stats(
        boundary_tagged_grid, spacing,
        label="before projection", step_index=step,
        output_folder=output_folder, config=config
    )
    divergence_before = stats_before["max"]

    base_dt = input_data.get("default_timestep", 0.01)
    delta_path = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
    trace_path = "data/testing-input-output/navier_stokes_output/mutation_pathways_log.json"
    dt = suggest_timestep(delta_path, trace_path, base_dt=base_dt, reflex_score=reflex_score)

    velocity_updated_grid = apply_momentum_update(boundary_tagged_grid, input_data, step)

    try:
        pressure_corrected_grid, pressure_has_changed, projection_passes, pressure_metadata = apply_pressure_correction(
            velocity_updated_grid, input_data, step
        )
    except ValueError as e:
        logging.error(f"[evolve_step] Pressure solver did not return expected values: {e}")
        raise

    velocity_projected_grid = apply_pressure_velocity_projection(pressure_corrected_grid, input_data)

    stats_after = compute_divergence_stats(
        velocity_projected_grid, spacing,
        label="after projection", step_index=step,
        output_folder=output_folder, config=config
    )
    divergence_after = stats_after["max"]

    reflex_metadata = apply_reflex(
        velocity_projected_grid,
        input_data,
        step,
        ghost_influence_count=influence_count,
        config=config,
        pressure_solver_invoked=True,
        pressure_mutated=pressure_has_changed,
        post_projection_divergence=divergence_after
    )
    logging.debug(f"📋 Reflex Flags: {reflex_metadata}")

    reflex_metadata["ghost_influence_count"] = influence_count
    reflex_metadata["ghost_registry"] = ghost_registry
    reflex_metadata["boundary_condition_applied"] = boundary_applied
    reflex_metadata["projection_passes"] = projection_passes
    reflex_metadata["adaptive_timestep"] = dt  # ✅ Log selected dt
    reflex_metadata["reflex_score"] = reflex_score  # ✅ Patch: propagate reflex score

    if isinstance(pressure_metadata, dict):
        reflex_metadata.update(pressure_metadata)

    reflex_metadata = inject_diagnostics(
        reflex_metadata, ghost_registry,
        grid=velocity_projected_grid,
        spacing=spacing
    )

    logging.info(f"✅ [evolve_step] Step {step}: Evolution complete")
    return velocity_projected_grid, reflex_metadata



