# src/physics/advection_methods/euler.py
# 🔄 Stub: Forward Euler advection method for velocity evolution

from src.grid_modules.cell import Cell
from typing import List
from src.physics.advection_methods.helpers import copy_cell

def compute_euler_velocity(grid: List[Cell], dt: float, config: dict) -> List[Cell]:
    """
    Applies a placeholder Forward Euler update to fluid cell velocities.

    Args:
        grid (List[Cell]): Simulation grid
        dt (float): Time step
        config (dict): Simulation configuration

    Returns:
        List[Cell]: Grid with updated fluid velocities
    """
    updated = []
    for cell in grid:
        if cell.fluid_mask and isinstance(cell.velocity, list):
            vx, vy, vz = cell.velocity
            # ⚠️ Stub: No actual advection logic yet; velocity unchanged
            new_velocity = [vx, vy, vz]
            updated.append(copy_cell(cell, velocity=new_velocity))
        else:
            updated.append(copy_cell(cell))
    return updated



