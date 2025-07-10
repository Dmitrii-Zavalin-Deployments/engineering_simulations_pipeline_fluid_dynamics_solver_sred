# src/metrics/divergence_metrics.py

def compute_max_divergence(grid: list) -> float:
    """
    Computes the maximum divergence across the simulation grid.
    Real implementation analyzes the velocity change around each cell.
    Currently simplified to use adjacent velocity differences in x-direction.

    Args:
        grid (list): Grid cells as [x, y, z, velocity_vector, pressure]

    Returns:
        float: Maximum divergence value detected
    """
    if not grid:
        return 0.0

    divergence_values = []

    for i in range(len(grid) - 1):
        cell_a = grid[i]
        cell_b = grid[i + 1]

        vx_a = cell_a[3][0] if isinstance(cell_a[3], list) and len(cell_a[3]) == 3 else 0.0
        vx_b = cell_b[3][0] if isinstance(cell_b[3], list) and len(cell_b[3]) == 3 else 0.0

        # Simplified 1D divergence: change in x-velocity between neighboring cells
        divergence = abs(vx_b - vx_a)
        divergence_values.append(divergence)

    return round(max(divergence_values), 5) if divergence_values else 0.0



