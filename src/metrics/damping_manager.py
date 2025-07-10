# src/metrics/damping_manager.py

import math

def should_dampen(grid: list, time_step: float) -> bool:
    """
    Determines whether flow damping should be enabled based on velocity volatility.
    Real implementation checks for velocity spikes exceeding a stability threshold.

    Args:
        grid (list): Grid cells as [x, y, z, velocity_vector, pressure]
        time_step (float): Simulation time step

    Returns:
        bool: True if damping should be applied, False otherwise
    """
    if not grid or time_step <= 0.0:
        return False

    velocity_magnitudes = []

    for cell in grid:
        velocity = cell[3]
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            velocity_magnitudes.append(magnitude)

    if not velocity_magnitudes:
        return False

    avg_velocity = sum(velocity_magnitudes) / len(velocity_magnitudes)
    max_velocity = max(velocity_magnitudes)
    volatility = max_velocity - avg_velocity

    # Damping threshold: triggers if spike exceeds 50% above average
    return volatility > (0.5 * avg_velocity)



