# src/metrics/velocity_metrics.py
# 💨 Velocity Metrics — computes max velocity magnitude across fluid cells for reflex diagnostics and overflow tagging
# 📌 This module supports momentum enforcement and mutation traceability.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip cells based on adjacency, boundary proximity, or pressure anomalies.

import math
from src.grid_modules.cell import Cell
from typing import List

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def compute_max_velocity(grid: List[Cell], overflow_threshold: float = 10.0) -> float:
    """
    Computes the maximum velocity magnitude in the simulation grid.
    Measures Euclidean norm of each velocity vector.
    Flags overflow for reflex diagnostics.

    Roadmap Alignment:
    Momentum Enforcement:
    - Tracks velocity magnitude for reflex scoring
    - Flags overflow for mutation diagnostics

    Args:
        grid (List[Cell]): A list of Cell objects representing the simulation grid
        overflow_threshold (float): Threshold for overflow tagging

    Returns:
        float: Maximum velocity magnitude across the grid
    """
    if not grid:
        return 0.0

    max_magnitude = 0.0
    overflow_count = 0

    for cell in grid:
        if not cell.fluid_mask:
            continue  # ❌ Explicit exclusion: solid or ghost cell

        velocity = cell.velocity
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            if magnitude > overflow_threshold:
                cell.overflow_triggered = True
                cell.mutation_source = "velocity_overflow"
                overflow_count += 1
                if debug:
                    print(f"[VELOCITY] ⚠️ Overflow @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f}) → magnitude = {magnitude:.2f}")
            max_magnitude = max(max_magnitude, magnitude)

    if debug:
        print(f"[VELOCITY] Max velocity magnitude across fluid cells: {max_magnitude:.5f}")
        if overflow_count > 0:
            print(f"[VELOCITY] ⚠️ {overflow_count} cells exceeded overflow threshold ({overflow_threshold})")

    return round(max_magnitude, 5)



