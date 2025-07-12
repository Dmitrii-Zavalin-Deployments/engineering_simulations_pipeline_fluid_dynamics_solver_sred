# src/step_controller.py

import logging
from typing import List
from src.grid_modules.cell import Cell
from src.solvers.momentum_solver import apply_momentum_update
from src.solvers.pressure_solver import apply_pressure_correction
from src.reflex.reflex_controller import apply_reflex

def evolve_step(grid: List[Cell], input_data: dict, step: int) -> List[Cell]:
    """
    Evolves the fluid grid by one time step using momentum, pressure, and reflex updates.
    """
    logging.info(f"🌀 [evolve_step] Step {step}: Beginning evolution")

    grid = apply_momentum_update(grid, input_data, step)
    grid = apply_pressure_correction(grid, input_data, step)
    grid = apply_reflex(grid, input_data, step)

    logging.info(f"✅ [evolve_step] Step {step}: Completed evolution")
    return grid



