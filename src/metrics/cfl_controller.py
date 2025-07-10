# src/metrics/cfl_controller.py

import math

def compute_global_cfl(grid: list, time_step: float, domain: dict) -> float:
    """
    Computes the global CFL number for the simulation grid.

    Args:
        grid (list): Grid cells as [x, y, z, velocity_vector, pressure]
        time_step (float): Time step duration
        domain (dict): Contains nx, min_x, max_x for dx calculation

    Returns:
        float: Maximum CFL value across the grid
    """
    if not grid or "nx" not in domain or "min_x" not in domain or "max_x" not in domain:
        return 0.0

    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    max_cfl = 0.0

    for cell in grid:
        velocity = cell[3]
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            cfl = magnitude * time_step / dx
            max_cfl = max(max_cfl, cfl)

    return round(max_cfl, 5)



