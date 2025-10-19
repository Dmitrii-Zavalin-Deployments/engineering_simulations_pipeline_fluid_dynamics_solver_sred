# src/metrics/velocity_metrics.py

import math
from src.grid_modules.cell import Cell
from typing import List

def compute_max_velocity(grid: List[Cell], overflow_threshold: float = 10.0) -> float:
    """
    Computes the maximum velocity magnitude in the simulation grid.
    Measures Euclidean norm of each velocity vector.
    Annotates cells with velocity magnitude and overflow flags for reflex diagnostics.

    Roadmap Alignment:
    Momentum Enforcement:
    - Tracks velocity magnitude for reflex scoring
    - Flags overflow for mutation diagnostics

    Args:
        grid (List[Cell]): A list of Cell objects representing the simulation grid
        overflow_threshold (float): Threshold for overflow tagging

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
            cell.velocity_magnitude = round(magnitude, 6)
            if cell.fluid_mask and magnitude > overflow_threshold:
                cell.overflow_triggered = True
                cell.mutation_source = "velocity_overflow"
            max_magnitude = max(max_magnitude, magnitude)

    return round(max_magnitude, 5)



