# src/reflex/reflex_logic.py
# 🧠 Reflex logic module — real-time flow diagnostics based on velocity and configuration

from src.grid_modules.cell import Cell
from typing import List

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

    return (max_mag - avg_mag) > (volatility_threshold * avg_mag)

def should_flag_overflow(grid: List[Cell], threshold: float = 10.0) -> bool:
    """
    Flags overflow if any fluid velocity magnitude exceeds the specified threshold.

    Args:
        grid (List[Cell]): Grid of simulation cells
        threshold (float): Velocity magnitude threshold for overflow detection

    Returns:
        bool: True if overflow is detected, False otherwise
    """
    for cell in grid:
        v = cell.velocity
        if cell.fluid_mask and isinstance(v, list) and len(v) == 3:
            mag = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
            if mag > threshold:
                return True
    return False

def adjust_time_step(grid: List[Cell], config: dict, cfl_limit: float = 1.0) -> float:
    """
    Adjusts time step based on CFL stability condition.

    Args:
        grid (List[Cell]): Grid of simulation cells
        config (dict): Simulation configuration
        cfl_limit (float): CFL upper bound for stability (default 1.0)

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

    # Reduce time step if CFL exceeds limit
    if cfl > cfl_limit and max_velocity > 0.0:
        return (cfl_limit * dx) / max_velocity

    return dt



