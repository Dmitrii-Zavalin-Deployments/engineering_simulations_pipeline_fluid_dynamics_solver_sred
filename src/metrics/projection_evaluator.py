# src/metrics/projection_evaluator.py

def calculate_projection_passes(grid: list) -> int:
    """
    Estimates how many passes a projection solver would need based on velocity variability.
    Real implementation calculates a rough count based on flow irregularity.

    Args:
        grid (list): Grid cells as [x, y, z, velocity_vector, pressure]

    Returns:
        int: Estimated projection passes (minimum of 1)
    """
    if not grid:
        return 1

    velocity_magnitudes = []

    for cell in grid:
        velocity = cell[3]
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = sum(v**2 for v in velocity) ** 0.5
            velocity_magnitudes.append(magnitude)

    if not velocity_magnitudes:
        return 1

    max_v = max(velocity_magnitudes)
    avg_v = sum(velocity_magnitudes) / len(velocity_magnitudes)
    variation = max_v - avg_v

    # Higher variation suggests more iterations needed for convergence
    passes = 1 + int(variation // 0.5)

    return max(passes, 1)



