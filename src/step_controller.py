# src/step_controller.py
# 🚀 Simulation Step Controller — orchestrates velocity, pressure, ghost logic, and reflex updates

import logging
from typing import List, Tuple
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

def evolve_step(grid: List[Cell], input_data: dict, step: int) -> Tuple[List[Cell], dict]:
    """
    Evolves the fluid grid by one simulation step using:
    - Ghost cell padding and boundary enforcement
    - Momentum update (advection + viscosity)
    - Ghost influence propagation (velocity/pressure transfer)
    - Pressure correction (divergence and projection)
    - Velocity projection via pressure gradient
    - Reflex logic (damping, overflow, CFL diagnostics)

    Args:
        grid (List[Cell]): Current grid state
        input_data (dict): Full input configuration
        step (int): Current simulation step index

    Returns:
        Tuple[List[Cell], dict]: Updated grid and reflex metadata for snapshot
    """
    logging.info(f"🌀 [evolve_step] Step {step}: Starting evolution")

    dx = (input_data["domain_definition"]["max_x"] - input_data["domain_definition"]["min_x"]) / input_data["domain_definition"]["nx"]
    dy = (input_data["domain_definition"]["max_y"] - input_data["domain_definition"]["min_y"]) / input_data["domain_definition"]["ny"]
    dz = (input_data["domain_definition"]["max_z"] - input_data["domain_definition"]["min_z"]) / input_data["domain_definition"]["nz"]
    spacing = (dx, dy, dz)

    # 🧱 Step 0a: Generate ghost cell padding from tagged boundary faces
    padded_grid, ghost_registry = generate_ghost_cells(grid, input_data)
    logging.debug(f"🧱 Generated {len(ghost_registry)} ghost cells")
    log_ghost_summary(ghost_registry)

    # 🧪 Step 0b: Enforce boundary conditions for both ghost and adjacent edge cells
    boundary_tagged_grid = apply_boundary_conditions(padded_grid, ghost_registry, input_data)

    # 👣 Step 0c: Apply ghost influence to nearby fluid cells
    influence_count = apply_ghost_influence(boundary_tagged_grid, spacing, verbose=True)
    logging.debug(f"👣 Ghost influence applied to {influence_count} fluid cells")

    # 📈 Step 0d: Compute divergence BEFORE projection
    compute_divergence_stats(boundary_tagged_grid, spacing, label="before projection")

    # 💨 Step 1: Apply momentum update to evolve velocity fields
    velocity_updated_grid = apply_momentum_update(boundary_tagged_grid, input_data, step)

    # 💧 Step 2: Apply pressure correction to maintain incompressibility
    pressure_corrected_grid = apply_pressure_correction(velocity_updated_grid, input_data, step)

    # 💨 Step 2.5: Project velocity using pressure gradient (∇p)
    velocity_projected_grid = apply_pressure_velocity_projection(pressure_corrected_grid, input_data)

    # 📈 Step 2.6: Compute divergence AFTER projection
    compute_divergence_stats(velocity_projected_grid, spacing, label="after projection")

    # 🔄 Step 3: Evaluate reflex metrics, flags, and diagnostics
    reflex_metadata = apply_reflex(velocity_projected_grid, input_data, step)
    logging.debug(f"📋 Reflex Flags: {reflex_metadata}")

    # 📦 Inject optional ghost diagnostics into reflex metadata
    reflex_metadata = inject_diagnostics(reflex_metadata, ghost_registry, grid=velocity_projected_grid, spacing=spacing)

    logging.info(f"✅ [evolve_step] Step {step}: Evolution complete")
    return velocity_projected_grid, reflex_metadata



