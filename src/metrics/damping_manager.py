# src/metrics/damping_manager.py
# 🌊 Damping Manager — evaluates velocity volatility to trigger reflex-aware flow
# damping
# 📌 This module analyzes fluid cell velocity magnitudes to detect instability.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip cells based on adjacency, boundary proximity, or pressure
# anomalies.

import math
from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


def should_dampen(grid: list[Cell], time_step: float) -> bool:
    """
    Determines whether flow damping should be enabled based on velocity volatility.
    Checks for spikes exceeding 50% above average magnitude.

    This function performs evaluation only. It does NOT mutate grid state.

    Args:
        grid (list[Cell]): Grid of simulation cells (original or cloned)
        time_step (float): Simulation time step

    Returns:
        bool: True if damping should be applied, False otherwise
    """
    if not grid or time_step <= 0.0:
        return False

    velocity_magnitudes = []

    for cell in grid:
        if not cell.fluid_mask:
            continue  # ❌ Explicit exclusion: solid or ghost cell

        velocity = cell.velocity
        if isinstance(velocity, list) and len(velocity) == 3:
            try:
                magnitude = math.sqrt(sum(v**2 for v in velocity))
                velocity_magnitudes.append(magnitude)
            except (TypeError, ValueError):
                continue  # Gracefully skip malformed velocity

    if not velocity_magnitudes:
        return False

    avg_velocity = sum(velocity_magnitudes) / len(velocity_magnitudes)
    max_velocity = max(velocity_magnitudes)
    volatility = max_velocity - avg_velocity

    if debug:
        print(
            f"[DAMPING] Avg velocity: {avg_velocity:.4f}, "
            f"Max velocity: {max_velocity:.4f}, Volatility: {volatility:.4f}"
        )

    # Trigger if max velocity exceeds 50% above average
    trigger = volatility > (0.5 * avg_velocity)

    if debug and trigger:
        print("[DAMPING] ⚠️ Damping triggered due to high volatility")

    return trigger
