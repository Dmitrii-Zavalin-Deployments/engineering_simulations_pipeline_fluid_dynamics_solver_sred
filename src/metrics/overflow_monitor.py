# src/metrics/overflow_monitor.py

def detect_overflow(grid: list) -> bool:
    """
    Detects whether any velocity magnitude in the grid exceeds a defined overflow threshold.
    Real implementation checks for extreme, likely unstable flow values.

    Args:
        grid (list): Grid cells formatted as [x, y, z, velocity_vector, pressure]

    Returns:
        bool: True if overflow is detected, False otherwise
    """
    if not grid:
        return False

    overflow_threshold = 10.0  # Define overflow threshold (units/sec)

    for cell in grid:
        velocity = cell[3]
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = sum(v**2 for v in velocity) ** 0.5
            if magnitude > overflow_threshold:
                return True

    return False



