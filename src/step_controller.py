# src/step_controller.py
# 🚀 Simulation Step Controller — orchestrates velocity, pressure, ghost logic, and reflex updates

import logging
from typing import List, Tuple
from src.grid_modules.cell import Cell
from src.physics.ghost_cell_generator import generate_ghost_cells
from src.physics.boundary_condition_solver import apply_boundary_conditions
from src.solvers.momentum_solver import apply_momentum_update
from src.solvers.pressure_solver import apply_pressure_correction
from src.reflex.reflex_controller import apply_reflex

def evolve_step(grid: List[Cell], input_data: dict, step: int) -> Tuple[List[Cell], dict]:
    """
    Evolves the fluid grid by one simulation step using:
    - Ghost cell padding and boundary enforcement
    - Momentum update (advection + viscosity)
    - Pressure correction (divergence and projection)
    - Reflex logic (damping, overflow, CFL diagnostics)

    Args:
        grid (List[Cell]): Current grid state
        input_data (dict): Full input configuration
        step (int): Current simulation step index

    Returns:
        Tuple[List[Cell], dict]: Updated grid and reflex metadata for snapshot
    """
    logging.info(f"🌀 [evolve_step] Step {step}: Starting evolution")

    # 🧱 Step 0a: Generate ghost cell padding
    grid, ghost_registry = generate_ghost_cells(grid, input_data)

    # 🧪 Step 0b: Apply boundary conditions to ghost cells and edge cells
    grid = apply_boundary_conditions(grid, ghost_registry, input_data)

    # 💨 Step 1: Apply momentum update to evolve velocity
    grid = apply_momentum_update(grid, input_data, step)

    # 💧 Step 2: Apply pressure correction to enforce incompressibility
    grid = apply_pressure_correction(grid, input_data, step)

    # 🔄 Step 3: Apply reflex diagnostics and flag logic
    reflex_metadata = apply_reflex(grid, input_data, step)

    logging.info(f"✅ [evolve_step] Step {step}: Evolution complete")
    logging.debug(f"📋 Reflex Flags: {reflex_metadata}")

    return grid, reflex_metadata



