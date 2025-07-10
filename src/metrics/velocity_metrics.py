# src/metrics/velocity_metrics.py

import math
from src.grid_modules.cell import Cell

def compute_max_velocity(grid: list[Cell]) -> float:
    """
    Computes the maximum velocity magnitude in the simulation grid.
    Measures Euclidean norm of each velocity vector.

    Args:
        grid (list[Cell]): A list of Cell objects representing the simulation grid

    Returns:
        float: Maximum velocity magnitude across the grid
    """
    if not grid:
        return 0.0

    max_magnitude = 0.0

    for cell in grid:
        velocity = cell.velocity
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            max_magnitude = max(max_magnitude, magnitude)

    return round(max_magnitude, 5)



