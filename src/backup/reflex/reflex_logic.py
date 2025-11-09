# src/reflex/reflex_logic.py
# 🧠 Reflex logic module — real-time flow diagnostics based on velocity and configuration
# 📌 This module supports damping triggers and timestep adjustment based on mutation density and CFL stability.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is geometry-mask-driven.

from src.grid_modules.cell import Cell
from typing import List

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def should_dampen(grid: List[Cell], volatility_threshold: float = 0.5) -> bool:
    """
    Determines whether damping should be enabled by assessing velocity volatility.

    Args:
        grid (List[Cell]): Grid of simulation cells
        volatility_threshold (float): Trigger ratio (e.g., 0.5 for 50% spike above average)

    Returns:
        bool: True if damping should be triggered, False otherwise
    """
    magnitudes = []
    for cell in grid:
        v = cell.velocity
        if cell.fluid_mask and isinstance(v, list) and len(v) == 3:
            mag = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
            magnitudes.append(mag)

    if not magnitudes:
        return False

    avg_mag = sum(magnitudes) / len(magnitudes)
    max_mag = max(magnitudes)

    triggered = (max_mag - avg_mag) > (volatility_threshold * avg_mag)
    if debug:
        print(f"[DAMPING] avg_mag={avg_mag:.3e}, max_mag={max_mag:.3e}, triggered={triggered}")
    return triggered


def adjust_time_step(grid: List[Cell], config: dict, cfl_limit: float = 1.0) -> float:
    """
    Adjusts time step based on CFL stability condition and reflex mutation density.

    Roadmap Alignment:
    - CFL enforcement → ∂u/∂t stability
    - Reflex scoring → mutation density and solver health

    Purpose:
    - Reduce timestep if CFL exceeds limit
    - Further reduce timestep if mutation density is high
    - Support reflex diagnostics and adaptive control

    Returns:
        float: Adapted time step
    """
    dt = config.get("simulation_parameters", {}).get("time_step", 0.1)
    domain = config.get("domain_definition", {})
    nx = domain.get("nx", 1)
    min_x = domain.get("min_x", 0.0)
    max_x = domain.get("max_x", 1.0)

    dx = (max_x - min_x) / nx if nx > 0 else 1.0

    max_velocity = 0.0
    for cell in grid:
        v = cell.velocity
        if cell.fluid_mask and isinstance(v, list) and len(v) == 3:
            mag = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
            max_velocity = max(max_velocity, mag)

    cfl = (max_velocity * dt) / dx if dx > 0.0 else 0.0

    # 🧠 Reflex-aware mutation density
    mutation_count = sum(
        1 for cell in grid
        if getattr(cell, "pressure_mutated", False)
        or getattr(cell, "damping_triggered", False)
        or getattr(cell, "transport_triggered", False)
    )
    fluid_count = sum(1 for cell in grid if getattr(cell, "fluid_mask", False))
    mutation_ratio = mutation_count / fluid_count if fluid_count > 0 else 0.0

    # 📉 Adjust timestep based on CFL and mutation activity
    if cfl > cfl_limit and max_velocity > 0.0:
        dt = (cfl_limit * dx) / max_velocity
        if debug:
            print(f"[TIMESTEP] CFL exceeded → cfl={cfl:.3e}, dt adjusted to {dt:.6f}")

    if mutation_ratio > 0.2:
        dt *= 0.75
        if debug:
            print(f"[TIMESTEP] High mutation ratio ({mutation_ratio:.3f}) → dt scaled to 75%")
    elif mutation_ratio > 0.1:
        dt *= 0.9
        if debug:
            print(f"[TIMESTEP] Moderate mutation ratio ({mutation_ratio:.3f}) → dt scaled to 90%")

    return round(dt, 6)



