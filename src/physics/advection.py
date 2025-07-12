# src/physics/advection.py
# 🌀 Velocity advection module — uses default Euler method

from src.grid_modules.cell import Cell
from typing import List
from src.physics.advection_methods.euler import compute_euler_velocity

def compute_advection(grid: List[Cell], dt: float, config: dict) -> List[Cell]:
    """
    Evolves fluid velocity fields using the default Forward Euler method.

    Args:
        grid (List[Cell]): List of Cell objects with current velocity
        dt (float): Time step (delta t)
        config (dict): Simulation config (e.g., domain, fluid properties)

    Returns:
        List[Cell]: Grid with updated velocity values (fluid cells evolved, solid unchanged)

    Notes:
        This version uses a fixed advection method (Euler).
        TODO: Refactor to dispatch multiple methods (e.g., semi-Lagrangian) if needed.
    """
    return compute_euler_velocity(grid, dt, config)



