# src/metrics/projection_evaluator.py

from src.grid_modules.cell import Cell
import math

def calculate_projection_passes(grid: list[Cell]) -> int:
    """
    Estimates how many projection passes are needed based on velocity variability.
    Greater velocity irregularity may require more iterations for pressure convergence.

    Args:
        grid (list[Cell]): Grid of simulation cells

    Returns:
        int: Estimated projection passes (minimum of 1)
    """
    if not grid:
        return 1

    velocity_magnitudes = []

    for cell in grid:
        velocity = cell.velocity
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            velocity_magnitudes.append(magnitude)

    if not velocity_magnitudes:
        return 1

    max_v = max(velocity_magnitudes)
    avg_v = sum(velocity_magnitudes) / len(velocity_magnitudes)
    variation = max_v - avg_v

    # Heuristic: more variation → more projection depth
    passes = 1 + int(variation // 0.5)

    return max(passes, 1)



