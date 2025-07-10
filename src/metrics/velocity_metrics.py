# src/metrics/velocity_metrics.py

def compute_max_velocity(grid: list) -> float:
    """
    Computes the maximum velocity magnitude in the grid.
    Stub implementation returns the first velocity's X component.

    Args:
        grid (list): A list of grid cells, each in format [x, y, z, velocity_vector, pressure]

    Returns:
        float: Maximum velocity magnitude (stubbed)
    """
    if not grid:
        return 0.0

    first_velocity_vector = grid[0][3]  # grid[0][3] is velocity_vector [vx, vy, vz]
    return abs(first_velocity_vector[0])



