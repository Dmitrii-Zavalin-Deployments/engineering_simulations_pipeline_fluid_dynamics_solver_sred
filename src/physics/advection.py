# src/physics/advection.py
# 🌀 Stub: Advection module for velocity evolution

from src.grid_modules.cell import Cell
from typing import List

def compute_advection(grid: List[Cell], dt: float, config: dict) -> List[Cell]:
    """
    Placeholder for velocity advection logic.

    Args:
        grid (List[Cell]): List of Cell objects with current velocity
        dt (float): Time step (delta t)
        config (dict): Simulation config (e.g., domain, fluid properties)

    Returns:
        List[Cell]: Grid with updated velocity values (only fluid cells)

    Notes:
        This stub assumes no velocity change and simply returns a deep copy.
        Will later implement semi-Lagrangian or Euler-based advection.
    """
    updated = []
    for cell in grid:
        if cell.fluid_mask:
            updated_cell = Cell(
                x=cell.x,
                y=cell.y,
                z=cell.z,
                velocity=cell.velocity[:],  # shallow copy
                pressure=cell.pressure,
                fluid_mask=True
            )
        else:
            updated_cell = Cell(
                x=cell.x,
                y=cell.y,
                z=cell.z,
                velocity=None,
                pressure=None,
                fluid_mask=False
            )
        updated.append(updated_cell)

    return updated



