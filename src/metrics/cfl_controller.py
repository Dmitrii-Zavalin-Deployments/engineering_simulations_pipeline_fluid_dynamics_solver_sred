# src/metrics/cfl_controller.py

import math
from src.grid_modules.cell import Cell
from typing import List, Dict

def compute_global_cfl(grid: List[Cell], time_step: float, domain: Dict, cfl_threshold: float = 1.0) -> float:
    """
    Computes the global CFL number for the simulation grid using velocity magnitudes.
    Annotates each cell with its local CFL value and flags overflow for reflex diagnostics.

    Roadmap Alignment:
    Continuity Enforcement:
    - CFL tracking supports timestep adaptation
    - Reflex scoring uses per-cell CFL diagnostics

    Args:
        grid (List[Cell]): Simulation grid cells
        time_step (float): Time step duration
        domain (Dict): Contains nx, min_x, max_x for dx calculation
        cfl_threshold (float): Threshold for CFL overflow tagging

    Returns:
        float: Maximum CFL value across the grid
    """
    if not grid or "nx" not in domain or "min_x" not in domain or "max_x" not in domain:
        return 0.0

    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    max_cfl = 0.0

    for cell in grid:
        velocity = cell.velocity
        if cell.fluid_mask and isinstance(velocity, list) and len(velocity) == 3:
            magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            cfl = magnitude * time_step / dx
            cell.local_cfl = round(cfl, 6)
            if cfl > cfl_threshold:
                cell.cfl_exceeded = True
                cell.mutation_source = "cfl_violation"
            max_cfl = max(max_cfl, cfl)

    return round(max_cfl, 5)



