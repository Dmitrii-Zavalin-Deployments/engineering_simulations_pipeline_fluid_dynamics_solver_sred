# src/metrics/overflow_monitor.py
# 🚨 Overflow Monitor — detects velocity spikes that exceed physical thresholds for reflex diagnostics
# 📌 This module scans fluid cells for non-physical velocity magnitudes.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip cells based on adjacency, boundary proximity, or pressure anomalies.

from src.grid_modules.cell import Cell
import math

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def detect_overflow(grid: list[Cell]) -> bool:
    """
    Detects whether any velocity magnitude in the grid exceeds the overflow threshold.
    Helps catch unstable or non-physical flow values.

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
            magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            if magnitude > overflow_threshold:
                overflow_detected = True
                if debug:
                    print(f"[OVERFLOW] ⚠️ Cell @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f}) → velocity magnitude = {magnitude:.2f} exceeds threshold")

    if debug and not overflow_detected:
        print("[OVERFLOW] ✅ No overflow detected across fluid cells")

    return overflow_detected



