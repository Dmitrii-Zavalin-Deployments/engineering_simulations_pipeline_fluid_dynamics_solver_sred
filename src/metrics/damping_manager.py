# src/metrics/damping_manager.py

import math
from src.grid_modules.cell import Cell

def should_dampen(grid: list[Cell], time_step: float) -> bool:
    """
    Determines whether flow damping should be enabled based on velocity volatility.
    Checks for spikes exceeding 50% above average magnitude.

    Args:
        grid (list[Cell]): Grid of simulation cells
        time_step (float): Simulation time step

    Returns:
        bool: True if damping should be applied, False otherwise
    """
    if not grid or time_step <= 0.0:
        return False

    velocity_magnitudes = []

    for cell in grid:
        velocity = cell.velocity
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            velocity_magnitudes.append(magnitude)

    if not velocity_magnitudes:
        return False

    avg_velocity = sum(velocity_magnitudes) / len(velocity_magnitudes)
    max_velocity = max(velocity_magnitudes)
    volatility = max_velocity - avg_velocity

    # Trigger if max velocity exceeds 50% above average
    return volatility > (0.5 * avg_velocity)



