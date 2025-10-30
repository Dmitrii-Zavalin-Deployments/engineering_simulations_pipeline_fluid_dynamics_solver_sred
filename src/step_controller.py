# src/step_controller.py
# 🚀 Simulation Step Controller — executes Navier-Stokes evolution per timestep
# 📌 This module coordinates ghost logic, boundary enforcement, divergence tracking, and reflex scoring.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

import logging
from typing import List, Tuple, Optional
from src.grid_modules.cell import Cell
from src.physics.ghost_cell_generator import generate_ghost_cells
from src.physics.boundary_condition_solver import apply_boundary_conditions
from src.physics.ghost_influence_applier import apply_ghost_influence
from src.solvers.navier_stokes_solver import solve_navier_stokes_step
from src.reflex.reflex_controller import apply_reflex
from src.utils.ghost_diagnostics import log_ghost_summary, inject_diagnostics
from src.utils.divergence_tracker import compute_divergence_stats
from src.adaptive.timestep_controller import suggest_timestep
from src.utils.grid_spacing import compute_grid_spacing

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def evolve_step(
    grid: List[Cell],
    input_data: dict,
    step: int,
    config: Optional[dict] = None,
    sim_config: Optional[dict] = None,
    reflex_score: Optional[int] = None
) -> Tuple[List[Cell], dict]:
    """
    Evolves the fluid grid by one simulation step using the full Navier-Stokes formulation.
    """
    logging.info(f"🌀 [evolve_step] Step {step}: Starting evolution")

    if "domain_definition" not in input_data:
        raise KeyError("Missing required 'domain_definition' in input_data")
    domain = input_data["domain_definition"]
    dx, dy, dz = compute_grid_spacing(domain)
    spacing = (dx, dy, dz)

    output_folder = "data/testing-input-output/navier_stokes_output"

    padded_grid, ghost_registry = generate_ghost_cells(grid, input_data)
    if debug:
        logging.debug(f"🧱 Generated {len(ghost_registry)} ghost cells")
        log_ghost_summary(ghost_registry)

    boundary_tagged_grid = apply_boundary_conditions(padded_grid, ghost_registry, input_data)
    boundary_applied = True

    if config is None:
        raise ValueError("Missing required 'config' dictionary")
    if "ghost_adjacency_depth" not in config:
        raise KeyError("Missing required 'ghost_adjacency_depth' in config")
    if "reflex_verbosity" not in config:
        raise KeyError("Missing required 'reflex_verbosity' in config")

    influence_count = apply_ghost_influence(
        boundary_tagged_grid,
        spacing,
        verbose=config["reflex_verbosity"] == "high",
        radius=config["ghost_adjacency_depth"]
    )
    if debug:
        logging.debug(f"👣 Ghost influence applied to {influence_count} fluid cells")

    stats_before = compute_divergence_stats(
        boundary_tagged_grid, spacing,
        label="before projection", step_index=step,
        output_folder=output_folder, config=config
    )
    stats_before["max"]

    if "simulation_parameters" not in input_data:
        raise KeyError("Missing 'simulation_parameters' in input_data")
    sim_params = input_data["simulation_parameters"]
    if "time_step" not in sim_params:
        raise KeyError("Missing 'time_step' in simulation_parameters")
    base_dt = sim_params["time_step"]

    delta_path = f"data/snapshots/pressure_delta_map_step_{step:04d}.json"
    trace_path = "data/testing-input-output/navier_stokes_output/mutation_pathways_log.json"
    dt = suggest_timestep(delta_path, trace_path, base_dt=base_dt, reflex_score=reflex_score)

    velocity_projected_grid, ns_metadata = solve_navier_stokes_step(boundary_tagged_grid, input_data, step)

    stats_after = compute_divergence_stats(
        velocity_projected_grid, spacing,
        label="after projection", step_index=step,
        output_folder=output_folder, config=config
    )
    divergence_after = stats_after["max"]

    if "pressure_mutated" not in ns_metadata:
        raise KeyError("Missing required 'pressure_mutated' in Navier-Stokes metadata")
    if "projection_passes" not in ns_metadata:
        raise KeyError("Missing required 'projection_passes' in Navier-Stokes metadata")

    reflex_metadata = apply_reflex(
        velocity_projected_grid,
        input_data,
        step,
        ghost_influence_count=influence_count,
        config=config,
        sim_config=sim_config,
        pressure_solver_invoked=True,
        pressure_mutated=ns_metadata["pressure_mutated"],
        post_projection_divergence=divergence_after
    )
    if debug:
        print(f"[DEBUG] 📋 Reflex Flags: {reflex_metadata}")

    reflex_metadata["ghost_influence_count"] = influence_count
    reflex_metadata["ghost_registry"] = ghost_registry
    reflex_metadata["boundary_condition_applied"] = boundary_applied
    reflex_metadata["projection_passes"] = ns_metadata["projection_passes"]
    reflex_metadata["adaptive_timestep"] = dt
    reflex_metadata.update(ns_metadata)

    reflex_metadata = inject_diagnostics(
        reflex_metadata, ghost_registry,
        grid=velocity_projected_grid,
        spacing=spacing
    )

    logging.info(f"✅ [evolve_step] Step {step}: Evolution complete")
    return velocity_projected_grid, reflex_metadata
