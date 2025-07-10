# src/metrics/overflow_monitor.py

from src.grid_modules.cell import Cell
import math

def detect_overflow(grid: list[Cell]) -> bool:
    """
    Detects whether any velocity magnitude in the grid exceeds the overflow threshold.
    Helps catch unstable or non-physical flow values.

    Args:
        grid (list[Cell]): Structured simulation grid

    Returns:
        bool: True if overflow is detected, False otherwise
    """
    if not grid:
        return False

    overflow_threshold = 10.0  # Units/sec

    for cell in grid:
        velocity = cell.velocity
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            if magnitude > overflow_threshold:
                return True

    return False



