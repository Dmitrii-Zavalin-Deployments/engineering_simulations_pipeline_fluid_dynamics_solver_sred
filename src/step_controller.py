# src/step_controller.py

import logging
from typing import List
from src.grid_modules.cell import Cell

def evolve_step(grid: List[Cell], input_data: dict, step: int) -> List[Cell]:
    """
    Evolves the fluid grid by one time step.
    This stub returns the grid unchanged but logs simulation progression.

    Args:
        grid (List[Cell]): Current fluid state grid at time t
        input_data (dict): Full simulation configuration and parameters
        step (int): Current time step index

    Returns:
        List[Cell]: Updated grid after one time step (currently unchanged)
    """
    logging.info(f"🌀 [evolve_step] Step {step}: Evolution placeholder activated")

    # ✅ This is where future modules will update velocity, pressure, boundaries
    # For now, the stub simply returns the grid as-is
    return grid



