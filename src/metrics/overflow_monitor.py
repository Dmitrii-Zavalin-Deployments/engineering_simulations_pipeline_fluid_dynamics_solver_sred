# src/metrics/overflow_monitor.py
# 🚨 Overflow Monitor — detects velocity spikes that exceed physical thresholds
# for reflex diagnostics
# 📌 This module scans fluid cells for non-physical velocity magnitudes.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip cells based on adjacency, boundary proximity, or pressure
# anomalies.

from src.grid_modules.cell import Cell
import math

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


def detect_overflow(grid: list[Cell]) -> bool:
    """
    Detects whether any velocity magnitude in the grid exceeds the overflow threshold.
    Helps catch unstable or non-physical flow values.

    This function performs detection only. It does NOT mutate cell flags.

    Args:
        grid (list[Cell]): Structured simulation grid

    Returns:
        bool: True if overflow is detected, False otherwise
    """
    if not grid:
        return False

    overflow_threshold = 10.0  # Units/sec
    overflow_detected = False

    for cell in grid:
        if not cell.fluid_mask:
            continue  # ❌ Explicit exclusion: solid or ghost cell

        velocity = cell.velocity
        if isinstance(velocity, list) and len(velocity) == 3:
            try:
                magnitude = math.sqrt(sum(v**2 for v in velocity))
                if magnitude > overflow_threshold:
                    overflow_detected = True
                    if debug:
                        print(
                            f"[OVERFLOW] ⚠️ Cell @ ({cell.x:.2f}, {cell.y:.2f}, "
                            f"{cell.z:.2f}) → velocity magnitude = {magnitude:.2f} "
                            f"exceeds threshold"
                        )
            except (TypeError, ValueError):
                continue  # Gracefully skip malformed velocity

    if debug and not overflow_detected:
        print("[OVERFLOW] ✅ No overflow detected across fluid cells")

    return overflow_detected
