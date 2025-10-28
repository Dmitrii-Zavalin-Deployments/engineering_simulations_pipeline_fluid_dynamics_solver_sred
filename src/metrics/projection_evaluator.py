# src/metrics/projection_evaluator.py
# 📐 Projection Evaluator — estimates projection depth based on velocity variability
# across fluid cells
# 📌 This module supports reflex scoring for ∇²P enforcement fidelity.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip cells based on adjacency, boundary proximity, or pressure
# anomalies.

from src.grid_modules.cell import Cell
import math
from typing import List

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


def calculate_projection_passes(grid: List[Cell]) -> int:
    """
    Estimates how many projection passes are needed based on velocity variability.
    Greater velocity irregularity may require more iterations for pressure convergence.

    Roadmap Alignment:
    Reflex Scoring:
    - Projection depth → ∇²P enforcement fidelity
    - Supports reflex scoring and solver traceability

    Args:
        grid (List[Cell]): Grid of simulation cells

    Returns:
        int: Estimated projection passes (minimum of 1)
    """
    if not grid:
        return 1

    velocity_magnitudes = []

    for cell in grid:
        if not cell.fluid_mask:
            continue  # ❌ Explicit exclusion: solid or ghost cell

        velocity = cell.velocity
        if isinstance(velocity, list) and len(velocity) == 3:
            magnitude = math.sqrt(
                velocity[0]**2 + velocity[1]**2 + velocity[2]**2
            )
            velocity_magnitudes.append(magnitude)

    if not velocity_magnitudes:
        return 1

    max_v = max(velocity_magnitudes)
    avg_v = sum(velocity_magnitudes) / len(velocity_magnitudes)
    variation = max_v - avg_v

    # Heuristic: more variation → more projection depth
    passes = 1 + int(variation // 0.5)

    if debug:
        print(
            f"[PROJECTION] Velocity variation → max={max_v:.4f}, "
            f"avg={avg_v:.4f}, passes={passes}"
        )

    return max(passes, 1)
