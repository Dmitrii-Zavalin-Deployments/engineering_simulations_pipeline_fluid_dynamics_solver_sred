# src/step_controller.py
# 🚀 Simulation Step Controller — orchestrates velocity, pressure, and reflex updates

import logging
from typing import List, Tuple
from src.grid_modules.cell import Cell
from src.solvers.momentum_solver import apply_momentum_update
from src.solvers.pressure_solver import apply_pressure_correction
from src.reflex.reflex_controller import apply_reflex

def evolve_step(grid: List[Cell], input_data: dict, step: int) -> Tuple[List[Cell], dict]:
    """
    Evolves the fluid grid by one simulation step using:
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

    # 1️⃣ Apply momentum update to evolve velocity
    grid = apply_momentum_update(grid, input_data, step)

    # 2️⃣ Apply pressure correction to enforce incompressibility
    grid = apply_pressure_correction(grid, input_data, step)

    # 3️⃣ Apply reflex diagnostics and flag logic
    reflex_metadata = apply_reflex(grid, input_data, step)

    logging.info(f"✅ [evolve_step] Step {step}: Evolution complete")
    logging.debug(f"📋 Reflex Flags: {reflex_metadata}")

    return grid, reflex_metadata



