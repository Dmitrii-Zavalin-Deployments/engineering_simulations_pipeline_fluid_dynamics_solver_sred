# src/metrics/divergence_metrics.py

from src.grid_modules.cell import Cell

def compute_max_divergence(grid: list[Cell]) -> float:
    """
    Computes the maximum divergence across the simulation grid.
    Currently simplified to adjacent x-velocity differences.

    Args:
        grid (list[Cell]): Grid of simulation cells

    Returns:
        float: Maximum divergence value detected
    """
    if not grid:
        return 0.0

    divergence_values = []

    for i in range(len(grid) - 1):
        cell_a = grid[i]
        cell_b = grid[i + 1]

        vx_a = cell_a.velocity[0] if isinstance(cell_a.velocity, list) and len(cell_a.velocity) == 3 else 0.0
        vx_b = cell_b.velocity[0] if isinstance(cell_b.velocity, list) and len(cell_b.velocity) == 3 else 0.0

        divergence = abs(vx_b - vx_a)  # Simplified 1D divergence
        divergence_values.append(divergence)

    return round(max(divergence_values), 5) if divergence_values else 0.0



